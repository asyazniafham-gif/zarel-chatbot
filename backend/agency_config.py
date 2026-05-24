"""
Agency Configuration — Zarel Sevan
=========================================================
AI Chatbot for Zarel Sevan TikTok Shop e-commerce
"""

AGENCY_ID = "zs"
AGENCY_NAME = "Zarel Sevan"
AGENCY_NAME_EN = "Zarel Sevan"
AGENCY_ACRONYM = "ZS"
AGENCY_WEBSITE = "https://zarelsevan.my"

CONTACT_ADDRESS = "Zarel Sevan, Malaysia"
CONTACT_PHONE = ""
CONTACT_FAX = ""
CONTACT_EMAIL = "admin@zarelsevan.my"
CONTACT_HOURS = "Mon–Sun: 9:00 AM – 10:00 PM"
TIKTOK_SHOP_URL = "https://www.tiktok.com/@zarelsevan"

INTERNAL_KEYWORDS = [
    "zarel", "zarel sevan", "zs",
    # Products
    "produk", "product", "item", "barang",
    "harga", "price", "diskaun", "discount", "promosi", "promotion",
    "stok", "stock", "ready", "available",
    # Shopping
    "tiktok shop", "tiktok", "shop", "beli", "buy", "order", "pesanan",
    "cod", "cash on delivery", "bayar", "payment",
    "hantar", "pos", "shipping", "delivery", "penghantaran",
    "tracking", "jejak",
    # Returns / Service
    "pulang", "return", "refund", "tukar", "exchange", "warranty", "waranti",
    "rosak", "complaint", "aduan",
    # Affiliation
    "affiliate", "komisen", "commission", "reseller", "dropship",
]

EXTERNAL_KEYWORDS = [
    "cuaca", "weather", "berita", "news",
    "perbandingan", "comparison", "pesaing", "competitor",
    "terkini", "semasa", "terbaru", "current",
]

NEWS_KEYWORDS = [
    "promosi terkini", "produk terbaru", "kemaskini stok",
]

NEWS_URLS = []
NEWS_BASE_URL = "https://zarelsevan.my"
WEB_SEARCH_PREFIX = "Zarel Sevan TikTok Shop Malaysia produk"

WEBSITE_LIVE_PAGES = []
WEBSITE_KEYWORD_MAPPING = {}

CHROMA_COLLECTION_NAME = f"{AGENCY_ID}_knowledge"

INSTALL_DIR = f"/app"
FRONTEND_DIR = f"/app/frontend"
SERVICE_NAME = f"{AGENCY_ID}-chatbot"
PORT = 8010
CHROMA_DB_DIR = f"{INSTALL_DIR}/chroma_db"
KNOWLEDGE_DIR = f"{INSTALL_DIR}/knowledge"
DOCUMENTS_DIR = f"{INSTALL_DIR}/documents"
LOG_DIR = f"{INSTALL_DIR}/logs"
HF_CACHE_DIR = f"{INSTALL_DIR}/.hf_cache"

SYSTEM_PROMPT = f"""You are the official AI assistant for {AGENCY_NAME} — a Malaysian TikTok Shop selling quality products.

ABOUT {AGENCY_NAME}:
{AGENCY_NAME} is a Malaysian e-commerce store operating on TikTok Shop. We offer quality products with fast nationwide delivery and friendly customer service.
TikTok Shop: {TIKTOK_SHOP_URL}
Website: {AGENCY_WEBSITE}
Email: {CONTACT_EMAIL}
Hours: {CONTACT_HOURS}

HOW TO RESPOND:
1. Always respond in English. If the customer writes in Malay, still reply in English.
2. Be direct and concise. Do not repeat the question or use filler phrases.
3. Write like a helpful, friendly store assistant.
4. Do not use emojis.
5. For questions about price, stock, or orders — refer to the knowledge base information only.

CITING SOURCES — CRITICAL:
- If information comes from the knowledge base, state the source clearly.
- If you don't have enough information, tell the customer and direct them to contact us.
- NEVER fabricate product details, prices, or stock information not found in the context.

LIMITATIONS:
- Do not provide unverified price or stock information.
- NEVER reveal inventory quantities or unit counts. Only state whether a product is "available" or "out of stock".
- For complaints or order issues, direct customers to {CONTACT_EMAIL} or our TikTok Shop at {TIKTOK_SHOP_URL}.
- Do not give legal or financial advice."""
