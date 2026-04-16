<p align="center">
  <img src="logo/pt.jpg" alt="Pendakwah Teknologi" width="120">
</p>

<h1 align="center">PT Chatbot</h1>

<p align="center">
  <strong>Production-grade RAG chatbot with hybrid search, self-evaluation, voice I/O, and real-time streaming.</strong>
</p>

<p align="center">
  <a href="https://pendakwah.tech">Website</a> &bull;
  <a href="#use-this-as-a-template">Use as Template</a> &bull;
  <a href="#the-full-pipeline-explained">How It Works</a> &bull;
  <a href="#setup">Setup Guide</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI">
  <img src="https://img.shields.io/badge/ChromaDB-vector%20search-orange?style=flat-square" alt="ChromaDB">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="GPU">
</p>

---

## What Is This?

This is **not** a toy chatbot. This is a full-stack, production-ready AI assistant that:

- **Reads your documents** (PDF, DOCX, TXT, MD) and answers questions about them
- **Searches the web** when it needs current information
- **Scores its own answers** so you know when to trust them
- **Streams responses** in real-time like ChatGPT
- **Speaks and listens** with built-in voice (STT + TTS)
- **Logs everything** with a full admin dashboard for audit trails

Built for [Pendakwah Teknologi](https://pendakwah.tech) — a digital transformation company specializing in AI training, cybersecurity, video production, and tech consulting. But the architecture is **completely generic**. Swap out the company config and documents, and you have your own enterprise chatbot.

---

## Use This as a Template

This repo is designed to be forked and rebranded. Here's how to make it yours in 10 minutes:

### Step 1: Fork & Clone

```bash
# Fork this repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/YOUR_CHATBOT.git
cd YOUR_CHATBOT
```

### Step 2: Edit `backend/agency_config.py`

This is the **only file you must change**. It controls your chatbot's entire identity:

```python
AGENCY_ID = "your-company"          # Used in paths, service names, cache keys
AGENCY_NAME = "Your Company Name"    # Displayed in system prompts
AGENCY_ACRONYM = "YC"               # Short form
AGENCY_WEBSITE = "https://yoursite.com"
CONTACT_EMAIL = "hello@yoursite.com"

INTERNAL_KEYWORDS = [...]            # Words that trigger document search
EXTERNAL_KEYWORDS = [...]            # Words that trigger web search
WEB_SEARCH_PREFIX = "Your Company context keywords"

SYSTEM_PROMPT = """..."""            # The personality of your chatbot
```

### Step 3: Drop Your Documents

Put your PDFs, DOCX, or text files into the `knowledge/` folder. The system will automatically chunk them, embed them, and index them.

### Step 4: Set Your API Key

```bash
cp configs/backend.env.template backend.env
# Edit backend.env and add your OpenAI key:
# OPENAI_API_KEYS=sk-your-key-here
```

### Step 5: Run

```bash
bash scripts/setup.sh      # One-time setup
bash scripts/ingest.sh     # Ingest your documents
sudo systemctl start pt-chatbot
```

That's it. Your chatbot is live.

### What to Customize

| What | Where | Difficulty |
|------|-------|-----------|
| Company name, prompt, keywords | `backend/agency_config.py` | Easy |
| Logo | `frontend/pt.jpg` and `logo/pt.jpg` | Easy |
| Chat UI text, chips, colors | `frontend/index.html` | Easy |
| LLM provider (swap OpenAI for Anthropic, local models, etc.) | `backend/providers.py` | Medium |
| Add new API endpoints | `backend/app.py` | Medium |
| Nginx domain, SSL | `configs/pt-chatbot.conf` | Medium |

---

## The Full Pipeline — Explained

Every time a user asks a question, it goes through **8 stages** before they get an answer. Here's what each one does and why it exists.

```
User Query
    |
    v
[1] Query Classification -----> "Should I search docs, web, or both?"
    |
    v
[2] Query Expansion ----------> "Let me rephrase this 3 ways for better search"
    |
    v
[3] Hybrid Retrieval ---------> "Search by meaning AND by keywords, then fuse"
    |               \
    v                v
[4] Web Search      [5] Cross-Encoder Reranking
    (if needed)          "Re-score every result for real relevance"
    |               /
    v              v
[6] LLM Generation ----------> "Generate the answer with sources"
    |
    v
[7] Self-Evaluation ----------> "How good was my own answer? Score 1-5"
    |
    v
[8] Follow-up Suggestions ----> "Here are 3 things you might ask next"
```

### Stage 1: Query Classification

**File:** `backend/app.py` > `classify_query()`

Before doing anything, the system figures out *what kind of question* this is:

- **Internal** — The answer is probably in your documents. Example: "What training courses do you offer?"
- **External** — The answer needs current web info. Example: "What's the latest news about AI in Malaysia?"
- **Hybrid** — Needs both. Example: "How does your AI training compare to current market trends?"

**How it works:** The system checks the question against two keyword lists (`INTERNAL_KEYWORDS` and `EXTERNAL_KEYWORDS` in `agency_config.py`). It also uses regex patterns for common question structures. If the question scores high on internal keywords, it skips web search entirely (faster). If it scores high on external, it prioritizes web results.

**Why it matters:** Without this, every question would trigger both document search AND web search, wasting time and potentially polluting answers with irrelevant web results.

---

### Stage 2: Query Expansion

**File:** `backend/providers.py` > `HybridRetriever.expand_query()`

The user types one question. The system turns it into **4 questions** (the original + 3 LLM-generated variants).

**Example:**
- Original: "Kursus keselamatan siber"
- Variant 1: "Cybersecurity training programs and workshops"
- Variant 2: "Latihan keselamatan siber untuk organisasi"
- Variant 3: "Bengkel cybersecurity certification"

**How it works:** The fast LLM (GPT-4o-mini) takes your question and rewrites it 3 different ways — different languages, different terminology, different angles. All 4 versions are then searched in parallel.

**Why it matters:** Your document might say "cybersecurity workshop" but the user typed "kursus keselamatan siber". Without expansion, the search misses it. This is the single biggest improvement for recall (finding relevant documents).

---

### Stage 3: Hybrid Retrieval (Vector + BM25 + RRF)

**File:** `backend/providers.py` > `HybridRetriever.retrieve()`

This is the core search engine. It uses **two completely different search methods** and combines them:

#### Vector Search (Semantic)
- Converts the question into a 1024-dimensional number array (called an "embedding") using the Mesolitica model
- Finds document chunks whose embeddings are closest in meaning
- Understands that "kursus" and "training" mean the same thing
- Powered by ChromaDB with HNSW indexing (a fast nearest-neighbor algorithm)

#### BM25 Search (Keyword)
- Classic keyword matching — counts how many words overlap between query and document
- Good at finding exact terms, acronyms, specific names
- Uses BM25Okapi scoring (a proven formula from information retrieval research)

#### Reciprocal Rank Fusion (RRF)
- Takes the ranked results from both methods
- Assigns each result a score based on its rank position: `1/(60 + rank)`
- Adds up scores for documents that appear in both lists
- Documents found by BOTH methods get boosted to the top

**Why two methods?** Vector search is great at understanding meaning but sometimes misses exact keywords. BM25 is great at exact matching but doesn't understand synonyms. Together, they cover each other's blind spots.

---

### Stage 4: Web Search

**File:** `backend/providers.py` > `search_web()`

For external/hybrid queries, the system searches the live internet using **two providers** with automatic fallback:

1. **Tavily** (primary) — An AI-focused search API that returns clean, structured results. Uses "advanced" search depth for comprehensive results.
2. **Brave Search** (fallback) — If Tavily fails or has no key configured, Brave Search kicks in as backup.

**How it works:** The query is automatically prefixed with your company context (from `WEB_SEARCH_PREFIX` in config) so results are relevant. Top 5 results are returned with title, URL, and content snippets.

**Safety feature:** All URLs from web results are extracted and given to the LLM as an explicit allowlist. The LLM is instructed to ONLY link to these verified URLs — it cannot fabricate links.

---

### Stage 5: Cross-Encoder Reranking

**File:** `backend/providers.py` > `CrossEncoderReranker.rerank()`

The initial search returns ~15 document chunks. Most are relevant, some are noise. The cross-encoder **re-scores every single one** for true relevance and keeps the top 7.

**How it works:** Unlike the embedding model (which encodes query and document separately), the cross-encoder looks at the query AND document **together** as a pair. It's dramatically more accurate but slower — that's why we only run it on the shortlisted candidates, not the entire database.

**Model:** `ms-marco-MiniLM-L-6-v2` — a lightweight but effective cross-encoder trained on the MS MARCO passage ranking dataset (millions of real search queries from Bing).

**Why it matters:** This is the difference between "kinda relevant" and "exactly what you asked for". It turns a decent search into a precise one.

---

### Stage 6: LLM Generation

**File:** `backend/providers.py` > `OpenAIGenerator.generate_stream()`

The main LLM (GPT-4o) receives:
- The system prompt (your chatbot's personality and rules from `agency_config.py`)
- The top 7 document chunks from retrieval
- Any web search results
- The conversation history
- Chain-of-thought instructions

It generates the answer and **streams it token by token** via Server-Sent Events (SSE) — so the user sees text appearing in real-time, just like ChatGPT.

**Key design choices:**
- `temperature=0.2` — Low creativity, high accuracy. We want factual answers, not creative writing.
- `max_tokens=4000` — Generous limit for detailed technical answers.
- Round-robin key rotation — If you have multiple API keys, they're used in rotation to distribute rate limits.
- Automatic retry — If a request fails, it retries up to 2 times with a different API key.

---

### Stage 7: Self-Evaluation

**File:** `backend/providers.py` > `UltraEnhancer.self_evaluate()`

After the answer is generated, a **separate LLM call** (using the fast model) evaluates the answer on 3 dimensions:

| Dimension | What It Measures | Score |
|-----------|-----------------|-------|
| **Relevan** (Relevant) | Does the answer actually address the question? | 1-5 |
| **Tepat** (Accurate) | Is the answer grounded in the provided documents? | 1-5 |
| **Lengkap** (Complete) | Is the answer thorough enough? | 1-5 |

**How it works:** The evaluator LLM receives the original question, the retrieved documents, and the generated answer. It scores each dimension and provides a one-line note explaining its assessment.

**Why it matters:** This is your automatic quality check. If the score is low, the user (or admin) knows to double-check the answer. It also shows up in the admin dashboard for monitoring overall system quality.

---

### Stage 8: Follow-up Suggestions

**File:** `backend/providers.py` > `UltraEnhancer.suggest_followups()`

The fast LLM generates **3 contextual follow-up questions** based on the conversation. These appear as clickable chips in the chat UI.

**Why it matters:** Most users don't know what to ask next. Follow-up suggestions keep the conversation flowing and help users discover information they didn't know to ask about.

---

## Every Component Explained

### Backend Stack

| Component | What It Is | Why We Use It |
|-----------|-----------|---------------|
| **[FastAPI](https://fastapi.tiangolo.com/)** | A modern Python web framework | Async support, automatic API docs, type validation. The fastest Python framework available. |
| **[Uvicorn](https://www.uvicorn.org/)** | ASGI server that runs FastAPI | Runs 4 worker processes with uvloop (a fast event loop written in C). Handles hundreds of concurrent connections. |
| **[OpenAI API](https://platform.openai.com/)** | LLM provider (GPT-4o, GPT-4o-mini) | Best-in-class language models. GPT-4o for main answers, GPT-4o-mini for fast utility tasks. Easily swappable for any OpenAI-compatible API. |
| **[ChromaDB](https://www.trychroma.com/)** | Vector database | Stores document embeddings on disk. Uses HNSW algorithm for fast nearest-neighbor search. Zero config, runs embedded in the Python process. |
| **[Mesolitica Embeddings](https://huggingface.co/mesolitica/mistral-embedding-191m-8k-contrastive)** | Text-to-vector model | Specifically trained for Bahasa Melayu. Converts text into 1024-dimensional vectors that capture semantic meaning. Runs locally on GPU. |
| **[BM25Okapi](https://github.com/dorianbrown/rank_bm25)** | Keyword search algorithm | Classic information retrieval scoring. Complements vector search by catching exact keyword matches that semantic search might miss. |
| **[Cross-Encoder](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)** | Reranking model | Takes (query, document) pairs and scores relevance directly. Much more accurate than embedding similarity alone. Runs on GPU. |
| **[Redis](https://redis.io/)** | In-memory cache | Caches LLM responses for 10 minutes. Shared across all 4 worker processes. Falls back to per-process memory cache if Redis is unavailable. |
| **[SQLite](https://sqlite.org/)** | Conversation memory | Stores conversation history for multi-turn context. Lightweight, zero-config, file-based. |

### Voice Stack

| Component | What It Is | Why We Use It |
|-----------|-----------|---------------|
| **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** | Speech-to-text engine | OpenAI's Whisper model, re-implemented in CTranslate2 for 4x faster inference. Runs locally on GPU in float16. Supports Malay and English. |
| **[MMS-TTS](https://huggingface.co/facebook/mms-tts-zlm)** | Text-to-speech engine | Meta's Massively Multilingual Speech model, specifically the Malay variant. Generates natural-sounding WAV audio locally on GPU. |

### Frontend Stack

| Component | What It Is | Why We Use It |
|-----------|-----------|---------------|
| **Vanilla JavaScript** | No framework — pure JS | Zero dependencies, zero build step, loads instantly. The chat UI is a single HTML file with embedded CSS and JS. |
| **Server-Sent Events (SSE)** | Streaming protocol | One-way real-time stream from server to browser. Simpler than WebSockets for our use case (we only stream server responses). |
| **localStorage** | Browser storage | Persists chat history and feedback state across page refreshes. No cookies, no server-side sessions. |

### Infrastructure Stack

| Component | What It Is | Why We Use It |
|-----------|-----------|---------------|
| **[Nginx](https://nginx.org/)** | Reverse proxy & web server | Serves static frontend files, proxies API requests to FastAPI, handles SSL/TLS, rate limiting (20 req/s per IP), gzip compression, and SSE streaming. |
| **[systemd](https://systemd.io/)** | Process manager | Auto-starts the chatbot on boot, restarts on crash, enforces memory limits (8GB max), CPU quotas (60%), and security hardening (no new privileges, read-only filesystem). |
| **[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)** | Secure tunnel | Exposes the local server to the internet without opening firewall ports. Handles SSL termination and DDoS protection. |

---

## Project Structure

```
pt-chatbot/
|
|-- backend/
|   |-- app.py                # The main application. All API endpoints, streaming,
|   |                         # voice, caching, rate limiting, admin dashboard.
|   |-- providers.py          # The AI brain. Retrieval, reranking, generation,
|   |                         # web search, self-evaluation, follow-ups.
|   |-- agency_config.py      # Your chatbot's identity. Company info, system prompt,
|                             # keywords, paths. THE file to edit when rebranding.
|
|-- frontend/
|   |-- index.html            # Chat interface. Single-page app with markdown rendering,
|   |                         # voice input, source references, follow-up chips.
|   |-- architecture.html     # System documentation page. Pipeline flow, component details.
|   |-- admin.html            # Audit dashboard. Conversation logs, feedback, stats.
|   |-- pt.jpg                # Company logo (displayed in chat UI).
|
|-- configs/
|   |-- backend.env.template  # Environment variables template. Copy to backend.env
|   |                         # and fill in your API keys.
|   |-- pt-chatbot.service    # systemd unit file. Controls auto-start, memory limits,
|   |                         # security hardening.
|   |-- pt-chatbot.conf       # Nginx config. Domain, SSL, rate limiting, SSE proxy.
|
|-- knowledge/                # DROP YOUR DOCUMENTS HERE. PDF, DOCX, TXT, MD.
|                             # The ingest script will chunk and index them automatically.
|
|-- scripts/
|   |-- setup.sh              # One-command setup. Creates dirs, installs deps, downloads
|   |                         # models, configures systemd and nginx.
|   |-- ingest.sh             # Document ingestion. Clears ChromaDB, re-embeds everything
|                             # on GPU, stores in vector database.
|
|-- requirements.txt          # Python dependencies. Pin versions for reproducibility.
```

---

## Setup

### Prerequisites

- Ubuntu 22.04+ (tested on 24.04 aarch64)
- Python 3.11+
- NVIDIA GPU with CUDA 12+ (optional — falls back to CPU, just slower)
- Redis server
- Nginx (for production)

### Quick Start

```bash
# 1. Clone
git clone https://github.com/pendakwahteknologi/pt-chatbot.git
cd pt-chatbot

# 2. Run setup (installs everything)
bash scripts/setup.sh

# 3. Add your OpenAI key
nano /opt/pt-chatbot/backend.env
# Set: OPENAI_API_KEYS=sk-your-key-here

# 4. Add your documents
cp your-documents/*.pdf /opt/pt-chatbot/knowledge/

# 5. Ingest documents into vector DB
bash scripts/ingest.sh

# 6. Start
sudo systemctl start pt-chatbot

# 7. Verify
curl http://localhost:8003/api/health
```

### Environment Variables

| Variable | Default | What It Does |
|----------|---------|-------------|
| `OPENAI_API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint. Change this to use Azure OpenAI, local models, or any OpenAI-compatible API. |
| `OPENAI_API_KEYS` | *(required)* | Comma-separated API keys. Multiple keys enable round-robin rotation for rate limit distribution. |
| `OPENAI_MODEL` | `gpt-4o` | Main model for generating answers. The heavy lifter. |
| `OPENAI_MODEL_FAST` | `gpt-4o-mini` | Fast model for query expansion, self-eval, and follow-ups. Cheaper and quicker. |
| `EMBEDDING_DEVICE` | `cuda` | Where to run the embedding model. `cuda` for GPU, `cpu` for CPU. |
| `CROSS_ENCODER_DEVICE` | `cuda` | Where to run the reranker. Same options. |
| `RETRIEVAL_TOP_K` | `15` | How many document chunks to retrieve before reranking. Higher = better recall, slower. |
| `EMBEDDING_BATCH_SIZE` | `256` | How many chunks to embed at once during ingestion. Higher = faster on GPU. |
| `CACHE_TTL_SECONDS` | `600` | How long to cache responses (in seconds). 600 = 10 minutes. |
| `TAVILY_API_KEY` | *(optional)* | Enables Tavily web search. Get a key at [tavily.com](https://tavily.com). |
| `BRAVE_API_KEY` | *(optional)* | Enables Brave web search as fallback. Get a key at [brave.com/search/api](https://brave.com/search/api). |
| `CHROMA_PERSIST_DIR` | `/opt/pt-chatbot/chroma_db` | Where ChromaDB stores its data on disk. |

---

## API Reference

### Chat

```bash
# Synchronous (waits for full response)
curl -X POST http://localhost:8003/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What services do you offer?"}]}'

# Streaming (real-time SSE)
curl -X POST http://localhost:8003/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Tell me about AI training"}]}'
```

### Voice

```bash
# Speech-to-text (send base64 audio)
curl -X POST http://localhost:8003/api/voice/transcribe/local \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "<base64-encoded-audio>"}'

# Text-to-speech (returns WAV audio)
curl -X POST http://localhost:8003/api/voice/synthesize/local \
  -H "Content-Type: application/json" \
  -d '{"text": "Selamat datang ke Pendakwah Teknologi"}'
```

### System & Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Service health + document count |
| `GET` | `/api/mode` | Current mode, features, model info |
| `GET` | `/api/cache/stats` | Redis cache statistics |
| `POST` | `/api/cache/clear` | Clear response cache |
| `POST` | `/api/feedback` | Submit rating (1-5) and comment |
| `GET` | `/api/feedback/stats` | Rating distribution and average |
| `GET` | `/api/admin/conversations` | Full conversation audit log |
| `GET` | `/api/admin/feedback` | All feedback entries |
| `GET` | `/api/admin/summary` | Aggregate stats dashboard |

---

## Deployment Architecture

```
Internet
    |
    v
[Cloudflare Tunnel] --- SSL termination, DDoS protection
    |
    v
[Nginx] --- Rate limiting (20 req/s), gzip, static files, SSE proxy
    |
    v
[Uvicorn x4 workers] --- FastAPI application, async I/O, uvloop
    |         |         |
    v         v         v
[Redis]   [ChromaDB]  [SQLite]     <-- Shared state
            |
            v
    [GPU: Embeddings + Reranker + Whisper + TTS]
            |
            v
    [OpenAI API: GPT-4o + GPT-4o-mini]
    [Tavily / Brave: Web Search]
```

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/pendakwahteknologi">
        <img src="https://github.com/pendakwahteknologi.png" width="100" style="border-radius:50%;" alt="Pendakwah Teknologi"><br>
        <sub><b>Pendakwah Teknologi</b></sub>
      </a>
      <br>
      <sub>Architecture, Development, Deployment</sub>
    </td>
  </tr>
</table>

Built with grit by the [Pendakwah Teknologi](https://pendakwah.tech) team — innovating digital experiences with expert content, training & event solutions.

Want to contribute? Fork the repo, make your changes, and open a pull request.

---

## License

Proprietary. Copyright [Pendakwah Teknologi](https://pendakwah.tech).
