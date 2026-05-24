"""
Zarel Sevan Chatbot — AI Assistant
FastAPI backend: Query Expansion + Hybrid Search + Cross-Encoder +
Self-Eval + Follow-ups + Conversation Memory +
Claude (Anthropic) + Mesolitica Embeddings (CPU)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
import hashlib
import time
import re
import csv
import fcntl
import json
import base64
import io
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("zs-chatbot")

from agency_config import (
    AGENCY_ID, AGENCY_NAME, AGENCY_ACRONYM,
    INTERNAL_KEYWORDS, EXTERNAL_KEYWORDS,
    SYSTEM_PROMPT, PORT,
    INSTALL_DIR, LOG_DIR,
)
from providers import (
    get_providers, get_mode_info, ingest_knowledge_to_chroma, DEVICE,
    search_web,
)

# Configuration
CSV_LOG_PATH = os.environ.get("CSV_LOG_PATH", f"{LOG_DIR}/conversations.csv")
FEEDBACK_CSV_PATH = os.environ.get("FEEDBACK_CSV_PATH", f"{LOG_DIR}/feedback.csv")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "600"))


# ============================================================================
# REDIS CACHE — shared across all workers
# ============================================================================
import redis as _redis

_redis_client = None

def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            redis_url = os.environ.get("REDIS_URL", "")
            if redis_url:
                _redis_client = _redis.from_url(
                    redis_url, db=int(os.environ.get("REDIS_DB", "2")),
                    decode_responses=True, socket_connect_timeout=2,
                )
            else:
                _redis_client = _redis.Redis(
                    host=os.environ.get("REDIS_HOST", "localhost"),
                    port=int(os.environ.get("REDIS_PORT", "6379")),
                    db=int(os.environ.get("REDIS_DB", "2")),
                    decode_responses=True,
                    socket_connect_timeout=2,
                )
            _redis_client.ping()
            print("[ZS] Redis connected for caching")
        except Exception as e:
            print(f"[ZS] Redis unavailable ({e}), falling back to in-memory")
            _redis_client = None
    return _redis_client


class ResponseCache:
    """Redis-backed cache with in-memory fallback. Shared across all workers."""

    def __init__(self, ttl: int = 600, prefix: str = "zs:cache:"):
        self._ttl = ttl
        self._prefix = prefix
        # In-memory fallback
        self._mem: Dict[str, tuple[Any, float]] = {}

    def _key(self, k: str) -> str:
        return self._prefix + hashlib.md5(k.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        r = _get_redis()
        if r:
            try:
                val = r.get(self._key(key))
                if val:
                    return json.loads(val)
            except Exception:
                pass
        # Fallback
        hk = self._key(key)
        if hk in self._mem:
            val, ts = self._mem[hk]
            if time.time() - ts < self._ttl:
                return val
            del self._mem[hk]
        return None

    def set(self, key: str, value: Any):
        r = _get_redis()
        if r:
            try:
                r.setex(self._key(key), self._ttl, json.dumps(value, default=str))
                return
            except Exception:
                pass
        self._mem[self._key(key)] = (value, time.time())

    def stats(self) -> Dict[str, Any]:
        r = _get_redis()
        if r:
            try:
                keys = r.keys(self._prefix + "*")
                return {"entries": len(keys), "ttl_seconds": self._ttl, "backend": "redis"}
            except Exception:
                pass
        now = time.time()
        active = sum(1 for _, (_, ts) in self._mem.items() if now - ts < self._ttl)
        return {"entries": active, "ttl_seconds": self._ttl, "backend": "memory"}

    def clear(self):
        r = _get_redis()
        if r:
            try:
                keys = r.keys(self._prefix + "*")
                if keys:
                    r.delete(*keys)
                return
            except Exception:
                pass
        self._mem.clear()


response_cache = ResponseCache(ttl=CACHE_TTL_SECONDS)


# ============================================================================
# RATE LIMITING
# ============================================================================
_rate_limit_store: Dict[str, list] = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 30

def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_store[client_ip].append(now)
    return True


# ============================================================================
# QUERY CLASSIFICATION
# ============================================================================
def classify_query(query: str) -> str:
    query_lower = query.lower()

    internal_score = sum(1 for kw in INTERNAL_KEYWORDS if kw in query_lower)
    external_score = sum(1 for kw in EXTERNAL_KEYWORDS if kw in query_lower)

    if internal_score >= 2 and external_score == 0:
        return "internal"
    if external_score >= 2 and internal_score == 0:
        return "external"

    internal_patterns = [
        r"apakah (prosedur|sop|polisi|spesifikasi|keperluan)",
        r"bagaimana (untuk|cara|proses)",
        r"apa (maksud|definisi|keperluan)",
        r"senarai(kan)? (peralatan|bahan|dokumen)",
        r"berapa (kekerapan|masa|bilangan)",
        r"siapa (bertanggungjawab|perlu)",
    ]
    for pattern in internal_patterns:
        if re.search(pattern, query_lower):
            return "internal"

    external_patterns = [
        r"(terkini|berita|rencana|kemaskini).*(zarel|produk|tiktok)",
        r"(zarel|tiktok).*(terkini|berita|kemaskini|semasa)",
        r"apa.*(berlaku|terjadi).*(terkini|semasa)",
    ]
    for pattern in external_patterns:
        if re.search(pattern, query_lower):
            return "external"

    return "hybrid"


# ============================================================================
# ENHANCED PROMPT — Chain-of-Thought
# ============================================================================
def build_prompt(query: str, contexts: List[str], history: List[Dict[str, str]],
                 query_type: str = "hybrid", num_query_variants: int = 1,
                 web_contexts: List[str] = None) -> str:
    context_text = "\n\n---\n\n".join(contexts) if contexts else ""
    web_text = "\n\n---\n\n".join(web_contexts) if web_contexts else ""

    history_text = ""
    if len(history) > 1:
        for msg in history[:-1]:
            role = "Pengguna" if msg["role"] == "user" else "Pembantu"
            history_text += f"{role}: {msg['content']}\n"

    context_section = ""
    if context_text:
        context_section = f"""
=== DOKUMEN RUJUKAN (SUMBER UTAMA — WAJIB DIUTAMAKAN) ===
{context_text}
"""
    else:
        context_section = "Tiada dokumen berkaitan dijumpai dalam pangkalan data."

    if web_text:
        context_section += f"""

=== MAKLUMAT WEB (SUMBER TAMBAHAN — untuk maklumat terkini/konteks luar) ===
{web_text}
"""

    # Extract URLs from web results for safe citation
    urls_section = ""
    available_urls = []
    if web_text:
        for url_match in re.finditer(r'https?://[^\s\])<>"\']+', web_text):
            url = url_match.group()
            if url not in available_urls and not url.endswith(('.css', '.js', '.png', '.jpg', '.gif', '.ico')):
                available_urls.append(url)
    if available_urls:
        urls_list = "\n".join(f"- {u}" for u in available_urls[:10])
        urls_section = f"""

=== URL YANG SAH UNTUK RUJUKAN ===
{urls_list}

PERATURAN URL — WAJIB DIPATUHI:
- Guna format markdown: [Tajuk](URL) untuk pautan.
- HANYA guna URL dari senarai di atas. JANGAN SEKALI-KALI reka, teka, atau ubah suai URL.
- Jika tiada URL yang sesuai, nyatakan sumber tanpa pautan (contoh: "Menurut laman web rasmi...")."""
    else:
        urls_section = """

PERATURAN URL — WAJIB DIPATUHI:
- JANGAN letak sebarang URL atau pautan dalam jawapan kerana tiada sumber web yang disahkan.
- Rujuk seksyen dokumen sahaja (contoh: "Menurut Seksyen 23.4...")."""

    # Transparency about query expansion
    expansion_note = ""
    if num_query_variants > 1:
        expansion_note = f"\n[Sistem telah mencari dengan {num_query_variants} variasi soalan untuk ketepatan maksimum]\n"

    # Chain-of-thought instruction
    cot_instruction = """
PROSES PEMIKIRAN (gunakan secara dalaman, JANGAN tunjuk kepada pengguna):
1. FAHAMI — Apa sebenarnya yang ditanya?
2. ANALISIS — Dokumen mana yang relevan?
3. HUBUNGKAN — Kaitkan maklumat dari pelbagai sumber.
4. JAWAB — Beri jawapan yang jelas dan lengkap.
5. SUMBER — Nyatakan seksyen/bahagian atau URL yang dirujuk.

Tulis jawapan akhir sahaja — jangan tunjuk proses berfikir."""

    answer_instruction = "Jawab berdasarkan dokumen rujukan di atas. Nyatakan seksyen/bahagian yang dirujuk."
    if web_text:
        answer_instruction = "Jawab berdasarkan dokumen rujukan dan maklumat web di atas. Utamakan dokumen rasmi. Nyatakan seksyen/bahagian atau sumber web yang dirujuk."

    if history_text:
        prompt = f"""{SYSTEM_PROMPT}
{cot_instruction}

SEJARAH PERBUALAN:
{history_text}

{context_section}
{urls_section}
{expansion_note}
SOALAN TERKINI: {query}

{answer_instruction}"""
    else:
        prompt = f"""{SYSTEM_PROMPT}
{cot_instruction}

{context_section}
{urls_section}
{expansion_note}
SOALAN: {query}

{answer_instruction}"""

    return prompt


# ============================================================================
# CSV LOGGING
# ============================================================================
def log_conversation(query: str, reply: str, query_type: str,
                     num_sources: int, response_time_ms: int,
                     client_ip: str = "", user_agent: str = "",
                     cache_hit: bool = False):
    try:
        os.makedirs(os.path.dirname(CSV_LOG_PATH), exist_ok=True)
        file_exists = os.path.exists(CSV_LOG_PATH)
        with open(CSV_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "query", "reply", "query_type",
                    "num_sources", "response_time_ms",
                    "client_ip", "user_agent", "cache_hit"
                ])
            writer.writerow([
                datetime.now().isoformat(),
                query,
                reply,
                query_type,
                num_sources,
                response_time_ms,
                client_ip,
                user_agent[:200],
                cache_hit,
            ])
    except Exception as e:
        print(f"[ZS] CSV log error: {e}")


# ============================================================================
# CHAT LOGIC — ULTRA MODE
# ============================================================================
async def handle_chat(messages_payload: List[Dict[str, str]], latest_message: str) -> Dict[str, Any]:
    providers = get_providers()
    start_time = time.time()

    query_type = classify_query(latest_message)
    print(f"[ZS] Query classified as: {query_type}")

    # Step 1: Retrieve (hybrid: vector + BM25 + query expansion)
    retriever = providers["retriever"]
    contexts, sources = await retriever.retrieve(latest_message)

    # Step 1b: Web search for external/hybrid queries
    web_contexts, web_sources = [], []
    if query_type in ("external", "hybrid"):
        print(f"[ZS] Query is {query_type}, searching web...")
        web_search_fn = providers.get("web_search")
        if web_search_fn:
            web_contexts, web_sources = await asyncio.to_thread(web_search_fn, latest_message)

    # Step 2: Rerank with cross-encoder
    reranker = providers["reranker"]
    if reranker and len(sources) > 5:
        sources = await asyncio.to_thread(reranker.rerank, latest_message, sources, 7)
        contexts = [
            f"[SUMBER - {s.get('filename', '?')}]\n{s.get('page_content', '')}"
            for s in sources
        ]

    # Step 3: Generate with Claude (enhanced CoT prompt)
    generator = providers["generator"]
    num_variants = len(getattr(retriever, '_last_expanded', [latest_message]))
    prompt = build_prompt(latest_message, contexts, messages_payload, query_type, num_variants, web_contexts)
    reply = await asyncio.to_thread(generator.generate, prompt)

    # Step 4: Ultra extras — self-eval + follow-ups IN PARALLEL
    enhancer = providers.get("enhancer")
    extras = {}
    if enhancer:
        eval_task = asyncio.to_thread(
            enhancer.self_evaluate, latest_message, reply, contexts
        )
        followup_task = asyncio.to_thread(
            enhancer.suggest_followups, latest_message, reply
        )
        eval_data, followups = await asyncio.gather(eval_task, followup_task)
        extras["self_evaluation"] = eval_data
        extras["followup_suggestions"] = followups

        # Save to conversation memory
        memory = providers.get("memory")
        if memory:
            session_id = hashlib.md5(
                json.dumps(messages_payload[:1]).encode()
            ).hexdigest()[:12]
            memory.save_turn(session_id, "user", latest_message)
            memory.save_turn(session_id, "assistant", reply[:2000])

    # Combine sources
    all_sources = sources + web_sources

    response_time_ms = int((time.time() - start_time) * 1000)

    return {
        "reply": reply,
        "retrieval": all_sources if all_sources else None,
        "query_type": query_type,
        "response_time_ms": response_time_ms,
        **extras,
    }


async def handle_chat_stream(messages_payload: List[Dict[str, str]], latest_message: str):
    """Streaming version — returns (sources, query_type, stream_gen, enhancer, contexts)."""
    providers = get_providers()

    query_type = classify_query(latest_message)

    retriever = providers["retriever"]
    contexts, sources = await retriever.retrieve(latest_message)

    # Web search for external/hybrid queries
    web_contexts, web_sources = [], []
    if query_type in ("external", "hybrid"):
        web_search_fn = providers.get("web_search")
        if web_search_fn:
            web_contexts, web_sources = await asyncio.to_thread(web_search_fn, latest_message)

    reranker = providers["reranker"]
    if reranker and len(sources) > 5:
        sources = await asyncio.to_thread(reranker.rerank, latest_message, sources, 7)
        contexts = [
            f"[SUMBER - {s.get('filename', '?')}]\n{s.get('page_content', '')}"
            for s in sources
        ]

    generator = providers["generator"]
    num_variants = len(getattr(retriever, '_last_expanded', [latest_message]))
    prompt = build_prompt(latest_message, contexts, messages_payload, query_type, num_variants, web_contexts)

    all_sources = sources + web_sources
    enhancer = providers.get("enhancer")
    return all_sources, query_type, generator.generate_stream(prompt), enhancer, contexts, latest_message


# ============================================================================
# VOICE: STT (Whisper) + TTS (MMS-TTS Malay) — Local GPU
# ============================================================================
_whisper_model = None
_tts_model = None
_tts_tokenizer = None

def get_whisper_model():
    """Lazy-load Faster-Whisper large-v3 on GPU."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        device = os.environ.get("WHISPER_DEVICE", "cuda")
        print(f"[ZS] Loading Whisper {model_size} on {device}...")
        _whisper_model = WhisperModel(
            model_size, device=device, compute_type="float16"
        )
        print(f"[ZS] Whisper ready on {device}")
    return _whisper_model


def get_tts_model():
    """Lazy-load Facebook MMS-TTS Malay on GPU."""
    global _tts_model, _tts_tokenizer
    if _tts_model is None:
        import torch
        from transformers import VitsModel, AutoTokenizer
        model_name = os.environ.get("TTS_LOCAL_MODEL", "facebook/mms-tts-zlm")
        device = os.environ.get("TTS_LOCAL_DEVICE", "cuda")
        print(f"[ZS] Loading TTS {model_name} on {device}...")
        _tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tts_model = VitsModel.from_pretrained(model_name).to(device)
        print(f"[ZS] TTS ready on {device}")
    return _tts_model, _tts_tokenizer


def transcribe_audio_local(audio_bytes: bytes) -> str:
    """Transcribe audio using local Faster-Whisper on GPU."""
    import tempfile
    model = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, info = model.transcribe(
            tmp.name, beam_size=5, language="ms",
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join(seg.text.strip() for seg in segments)
    print(f"[ZS] Whisper transcribed: {text[:100]}...")
    return text


def synthesize_speech_local(text: str) -> bytes:
    """Synthesize speech using local MMS-TTS Malay on GPU."""
    import torch
    import struct
    model, tokenizer = get_tts_model()
    device = os.environ.get("TTS_LOCAL_DEVICE", "cuda")

    # Limit text length
    text = text[:3000]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    waveform = output.waveform[0].cpu().numpy()

    # Get actual sample rate from model config (MMS-TTS uses 16000Hz)
    sample_rate = getattr(model.config, 'sampling_rate', 16000)

    # Convert to 16-bit PCM WAV
    import numpy as np
    buf = io.BytesIO()
    num_samples = len(waveform)
    data_size = num_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<HHIIHH', 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    pcm = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
    buf.write(pcm.tobytes())

    print(f"[ZS] TTS synthesized: {len(text)} chars -> {data_size} bytes @ {sample_rate}Hz")
    return buf.getvalue()


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Zarel Sevan Chatbot — AI Assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(_frontend_dir, "index.html"))

    @app.get("/admin")
    async def serve_admin():
        return FileResponse(os.path.join(_frontend_dir, "admin.html"))


# ============================================================================
# MODELS
# ============================================================================
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    reply: str
    retrieval: Optional[List[Dict[str, Any]]] = None
    query_type: Optional[str] = None
    cache_hit: bool = False
    self_evaluation: Optional[Dict[str, Any]] = None
    followup_suggestions: Optional[List[str]] = None

class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int
    comment: Optional[str] = ""

class TranscribeRequest(BaseModel):
    audio_data: str  # Base64 encoded audio

class SynthesizeRequest(BaseModel):
    text: str


# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    client_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    user_agent = request.headers.get("user-agent", "")
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Terlalu banyak permintaan. Sila cuba semula.")

    messages_payload = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]
    latest_message = messages_payload[-1]["content"] if messages_payload else ""

    if not latest_message:
        raise HTTPException(status_code=400, detail="Mesej kosong")

    # Check cache — include conversation history to avoid cross-user collisions
    history_hash = hashlib.md5(json.dumps(messages_payload).encode()).hexdigest()[:8]
    cache_key = f"response:{history_hash}:{latest_message}"
    cached = response_cache.get(cache_key)
    if cached:
        await asyncio.to_thread(
            log_conversation, latest_message, cached["reply"],
            cached.get("query_type", ""), 0, 0, client_ip, user_agent, True
        )
        return ChatResponse(
            reply=cached["reply"],
            retrieval=cached["retrieval"],
            query_type=cached.get("query_type"),
            cache_hit=True,
            self_evaluation=cached.get("self_evaluation"),
            followup_suggestions=cached.get("followup_suggestions"),
        )

    try:
        result = await handle_chat(messages_payload, latest_message)

        response_cache.set(cache_key, result)

        await asyncio.to_thread(
            log_conversation, latest_message, result["reply"],
            result.get("query_type", ""), len(result.get("retrieval") or []),
            result.get("response_time_ms", 0), client_ip, user_agent, False
        )

        return ChatResponse(
            reply=result["reply"],
            retrieval=result["retrieval"],
            query_type=result.get("query_type"),
            cache_hit=False,
            self_evaluation=result.get("self_evaluation"),
            followup_suggestions=result.get("followup_suggestions"),
        )
    except Exception as exc:
        print(f"[ZS] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Ralat sistem: {str(exc)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    client_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
    user_agent = request.headers.get("user-agent", "")
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Terlalu banyak permintaan.")

    messages_payload = [
        {"role": msg.role, "content": msg.content}
        for msg in req.messages
    ]
    latest_message = messages_payload[-1]["content"] if messages_payload else ""

    if not latest_message:
        raise HTTPException(status_code=400, detail="Mesej kosong")

    try:
        sources, query_type, stream_gen, enhancer, contexts, query = await handle_chat_stream(
            messages_payload, latest_message
        )

        import orjson

        async def event_stream():
            # Send sources first
            sources_event = {
                "type": "sources",
                "data": {
                    "retrieval": sources,
                    "query_type": query_type,
                }
            }
            yield f"data: {orjson.dumps(sources_event).decode()}\n\n"

            # Stream LLM response — run sync generator in a thread
            # so each chunk flushes immediately without blocking the event loop
            q: asyncio.Queue = asyncio.Queue()
            _sentinel = object()

            def _drain_sync_gen():
                try:
                    for chunk in stream_gen:
                        q.put_nowait(chunk)
                except Exception as exc:
                    q.put_nowait(exc)
                finally:
                    q.put_nowait(_sentinel)

            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, _drain_sync_gen)

            full_reply = []
            while True:
                item = await q.get()
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    print(f"[ZS] Stream gen error: {item}")
                    break
                full_reply.append(item)
                chunk_event = {"type": "chunk", "data": item}
                yield f"data: {orjson.dumps(chunk_event).decode()}\n\n"

            full_text = "".join(full_reply)

            # Run self-eval + follow-ups in parallel after streaming completes
            extras = {}
            if enhancer:
                try:
                    eval_task = asyncio.to_thread(
                        enhancer.self_evaluate, query, full_text, contexts
                    )
                    followup_task = asyncio.to_thread(
                        enhancer.suggest_followups, query, full_text
                    )
                    eval_data, followups = await asyncio.gather(eval_task, followup_task)
                    extras["self_evaluation"] = eval_data
                    extras["followup_suggestions"] = followups
                except Exception as e:
                    print(f"[ZS] Stream extras error: {e}")

            # Done event with extras
            # Log the conversation
            await asyncio.to_thread(
                log_conversation, latest_message, full_text,
                query_type, len(sources), 0, client_ip, user_agent, False
            )

            done_event = {
                "type": "done",
                "data": {
                    "full_reply": full_text,
                    **extras,
                }
            }
            yield f"data: {orjson.dumps(done_event).decode()}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    except Exception as exc:
        print(f"[ZS] Stream ERROR: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Ralat sistem: {str(exc)}")


@app.get("/api/health")
async def health():
    from providers import get_chroma_collection
    try:
        collection = get_chroma_collection()
        doc_count = collection.count()
    except Exception:
        doc_count = -1

    return {
        "status": "ok",
        "service": "zs-chatbot",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "documents_indexed": doc_count,
        "device": DEVICE,
        "features": ["query_expansion", "hybrid_search", "cross_encoder",
                      "self_evaluation", "followup_suggestions",
                      "conversation_memory"],
    }


@app.get("/api/mode")
async def mode_info():
    return get_mode_info()


@app.get("/api/cache/stats")
async def cache_stats():
    return {"response_cache": response_cache.stats()}


@app.post("/api/cache/clear")
async def cache_clear():
    response_cache.clear()
    return {"status": "cleared"}


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    try:
        client_ip = request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")
        os.makedirs(os.path.dirname(FEEDBACK_CSV_PATH), exist_ok=True)
        file_exists = os.path.exists(FEEDBACK_CSV_PATH)
        with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "query", "response_preview", "rating", "comment", "client_ip"])
            writer.writerow([
                datetime.now().isoformat(),
                req.query[:500],
                req.response[:200],
                req.rating,
                req.comment[:500] if req.comment else "",
                client_ip,
            ])
        return {"status": "ok", "message": "Terima kasih atas maklum balas anda."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/stats")
async def feedback_stats():
    try:
        if not os.path.exists(FEEDBACK_CSV_PATH):
            return {"total": 0, "average_rating": 0}
        with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            ratings = [int(row["rating"]) for row in reader if row.get("rating")]
        if not ratings:
            return {"total": 0, "average_rating": 0}
        return {
            "total": len(ratings),
            "average_rating": round(sum(ratings) / len(ratings), 2),
            "distribution": {str(i): ratings.count(i) for i in range(1, 6)}
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# ADMIN — LOGS & AUDIT TRAIL
# ============================================================================
@app.get("/api/admin/conversations")
async def admin_conversations(limit: int = 200, offset: int = 0):
    """Return conversation logs as JSON for admin dashboard."""
    try:
        if not os.path.exists(CSV_LOG_PATH):
            return {"rows": [], "total": 0}
        rows = []
        with open(CSV_LOG_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        rows.reverse()  # newest first
        total = len(rows)
        return {"rows": rows[offset:offset + limit], "total": total}
    except Exception as e:
        return {"error": str(e), "rows": [], "total": 0}


@app.get("/api/admin/feedback")
async def admin_feedback(limit: int = 200, offset: int = 0):
    """Return feedback logs as JSON for admin dashboard."""
    try:
        if not os.path.exists(FEEDBACK_CSV_PATH):
            return {"rows": [], "total": 0}
        rows = []
        with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        rows.reverse()
        total = len(rows)
        return {"rows": rows[offset:offset + limit], "total": total}
    except Exception as e:
        return {"error": str(e), "rows": [], "total": 0}


@app.get("/api/admin/summary")
async def admin_summary():
    """Return summary stats for admin dashboard."""
    try:
        conv_count = 0
        unique_ips = set()
        query_types = {}
        total_response_time = 0
        cache_hits = 0

        if os.path.exists(CSV_LOG_PATH):
            with open(CSV_LOG_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    conv_count += 1
                    ip = row.get("client_ip", "")
                    if ip:
                        unique_ips.add(ip)
                    qt = row.get("query_type", "unknown")
                    query_types[qt] = query_types.get(qt, 0) + 1
                    try:
                        total_response_time += int(row.get("response_time_ms", 0))
                    except (ValueError, TypeError):
                        pass
                    if row.get("cache_hit", "").lower() == "true":
                        cache_hits += 1

        fb_count = 0
        fb_ratings = []
        if os.path.exists(FEEDBACK_CSV_PATH):
            with open(FEEDBACK_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fb_count += 1
                    try:
                        fb_ratings.append(int(row.get("rating", 0)))
                    except (ValueError, TypeError):
                        pass

        avg_response = round(total_response_time / conv_count) if conv_count else 0
        avg_rating = round(sum(fb_ratings) / len(fb_ratings), 1) if fb_ratings else 0

        return {
            "conversations": conv_count,
            "unique_ips": len(unique_ips),
            "feedback_count": fb_count,
            "average_rating": avg_rating,
            "average_response_ms": avg_response,
            "cache_hits": cache_hits,
            "query_types": query_types,
            "rating_distribution": {str(i): fb_ratings.count(i) for i in range(1, 6)} if fb_ratings else {},
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# VOICE ENDPOINTS
# ============================================================================
@app.post("/api/voice/transcribe/local")
async def voice_transcribe(req: TranscribeRequest):
    """Transcribe audio using local Faster-Whisper on GPU."""
    try:
        audio_bytes = base64.b64decode(req.audio_data)
        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio terlalu pendek")
        if len(audio_bytes) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Audio melebihi 25MB")

        text = await asyncio.to_thread(transcribe_audio_local, audio_bytes)
        return {"text": text, "engine": "whisper-large-v3-local"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ZS] Transcribe error: {e}")
        raise HTTPException(status_code=500, detail=f"Ralat transkripsi: {str(e)}")


@app.post("/api/voice/synthesize/local")
async def voice_synthesize(req: SynthesizeRequest):
    """Synthesize speech using local MMS-TTS Malay on GPU."""
    try:
        text = req.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Teks kosong")

        audio_bytes = await asyncio.to_thread(synthesize_speech_local, text)
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=pt-tts.wav"}
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ZS] TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"Ralat TTS: {str(e)}")


# Startup: pre-warm models
@app.on_event("startup")
async def startup():
    print(f"[ZS Chatbot] Starting on port {PORT}...")
    print(f"[ZS Chatbot] Device: {DEVICE}")
    print(f"[ZS Chatbot] Features: Query Expansion, Hybrid Search, Cross-Encoder, Self-Eval, Follow-ups, Memory")
    asyncio.create_task(asyncio.to_thread(get_providers))
