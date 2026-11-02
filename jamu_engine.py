"""
Mesin QA Jamu Madura (dipakai bot Telegram dan website)
IndoBERT-NER -> Hybrid Retrieval (TF-IDF x SBERT) -> RAG Gemini

Butuh file hasil notebook di folder yang sama:
  - knowledge_base.csv   (kolom: doc_id, nama, khasiat, doc_text)
  - ner_model/           (hasil save_pretrained model + tokenizer NER)
"""
import os
import re
import time
import random
import threading
from collections import OrderedDict
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from google import genai

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    level=logging.INFO)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("jamu_engine")

# ---------------- konfigurasi (samakan dengan notebook) ----------------
# Satu kunci umum, atau kunci terpisah per kanal (web / telegram).
# Kunci khusus kanal diutamakan; jika kosong dipakai GEMINI_API_KEY.
_COMMON_KEY = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
GEMINI_KEYS = {
    "web": (os.environ.get("GEMINI_API_KEY_WEB") or _COMMON_KEY).strip(),
    "telegram": (os.environ.get("GEMINI_API_KEY_TELEGRAM") or _COMMON_KEY).strip(),
}
GEMINI_MODELS = [m.strip() for m in
                 os.environ.get("GEMINI_MODELS", "gemini-3.6-flash").split(",") if m.strip()]

# Cek kunci di awal supaya gagal cepat dengan pesan jelas (sebelum model dimuat)
_missing = [c for c, k in GEMINI_KEYS.items() if not k]
if _missing:
    raise SystemExit(
        f"Kunci Gemini untuk kanal {', '.join(_missing)} belum terisi.\n"
        f"Buat file bernama persis '.env' di: {ENV_PATH}\n"
        "isi salah satu: GEMINI_API_KEY=... (dipakai bersama), atau\n"
        "GEMINI_API_KEY_WEB=... dan GEMINI_API_KEY_TELEGRAM=... (terpisah)"
    )

DATA_DIR = Path(os.environ.get("DATA_DIR") or BASE_DIR)   # folder knowledge_base.csv + ner_model/
KB_PATH = str(DATA_DIR / "knowledge_base.csv")
NER_DIR = str(DATA_DIR / "ner_model_v2")
SBERT_NAME = "LazarusNLP/all-indobert-base-v4"
EMB_CACHE = str(DATA_DIR / "kb_embeddings.npy")   # cache agar start berikutnya cepat

LABELS = ["O", "B-KHASIAT", "I-KHASIAT", "B-KANDUNGAN", "I-KANDUNGAN", "B-JENIS", "I-JENIS",]
I2L = {i: l for i, l in enumerate(LABELS)}
MAX_LEN = 192
TOP_K = 5
ALPHA = 0.1        # bobot SBERT terbaik (use_ner=True) dari notebook

# ---------------- knowledge base ----------------
kb = pd.read_csv(KB_PATH)
docs = kb["doc_text"].tolist()
doc_ids = kb["doc_id"].tolist()
_by_id = kb.set_index("doc_id")
log.info("Knowledge base: %d dokumen", len(kb))

# ---------------- NER ----------------
tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_DIR)
ner_model.eval()

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def decode_spans(text, tags):
    toks, spans, cur = tokenize(text), [], None
    for (_, s, e), tag in zip(toks, tags):
        if tag == "O":
            if cur:
                spans.append(cur)
                cur = None
        elif tag.startswith("B-") or cur is None:
            if cur:
                spans.append(cur)
            cur = [s, e]
        else:
            cur[1] = e
    if cur:
        spans.append(cur)
    return [text[a:b] for a, b in spans]


@torch.no_grad()
def predict_spans(text):
    toks = [t for t, _, _ in tokenize(text)]
    if not toks:
        return []
    enc = tokenizer([toks], is_split_into_words=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    pred = ner_model(**enc).logits.argmax(-1)[0]
    tags, prev = ["O"] * len(toks), None
    for pos, w in enumerate(enc.word_ids(batch_index=0)):
        if w is not None and w != prev:
            tags[w] = I2L[int(pred[pos])]
        prev = w
    return decode_spans(text, tags)


@lru_cache(maxsize=2048)
def ner_query(query):
    spans = predict_spans(query)
    return " ".join(spans) if spans else query


# ---------------- indeks retrieval ----------------
# PENTING: salin STOPWORDS lengkap dari notebook (di PDF barisnya terpotong)
# supaya hasil retrieval identik dengan yang kamu evaluasi.
STOPWORDS = set("""yang dan di ke dari untuk dengan pada ini itu atau juga serta
akan adalah dalam saya anak ada apa jamu yg dgn utk karena sehingga lebih sangat""".split())

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stem = StemmerFactory().create_stemmer().stem
    log.info("Stemming Sastrawi aktif")
except ImportError:
    _stem = lambda s: s
    log.warning("Sastrawi tidak terpasang, stemming dilewati")


def preprocess(text):
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return " ".join(t for t in _stem(text).split() if t not in STOPWORDS)


tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform([preprocess(d) for d in docs])

sbert = SentenceTransformer(SBERT_NAME)
if os.path.exists(EMB_CACHE) and np.load(EMB_CACHE).shape[0] == len(docs):
    E_docs = np.load(EMB_CACHE)
else:
    E_docs = sbert.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    np.save(EMB_CACHE, E_docs)
log.info("Indeks siap: TF-IDF %s | SBERT %s", X_tfidf.shape, E_docs.shape)


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def scores_tfidf(q):
    return (X_tfidf @ tfidf.transform([preprocess(q)]).T).toarray().ravel()


def scores_sbert(q):
    return E_docs @ sbert.encode([q], normalize_embeddings=True)[0]


def retrieve(query, k=TOP_K, alpha=ALPHA):
    q = ner_query(query)
    s = alpha * minmax(scores_sbert(q)) + (1 - alpha) * minmax(scores_tfidf(q))
    idx = np.argsort(-s)[:k]
    return [(doc_ids[i], float(s[i])) for i in idx]


# ---------------- Gemini ----------------
_clients = {}
for _key in set(GEMINI_KEYS.values()):
    _clients[_key] = genai.Client(api_key=_key)
log.info("Kunci Gemini: %s", "terpisah untuk web dan telegram"
         if GEMINI_KEYS["web"] != GEMINI_KEYS["telegram"] else "satu kunci dipakai bersama")


def get_client(channel):
    return _clients[GEMINI_KEYS.get(channel, GEMINI_KEYS["web"])]
RETRY_CODES = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")

RAG_PROMPT = """
Kamu adalah asisten informasi Jamu Madura yang ramah dan teliti.

Jawab pertanyaan berdasarkan konteks referensi yang diberikan.
Periksa kecocokan nama jamu, jenis atau bentuk sediaan,
khasiat, dan kandungan bahan jika informasi tersebut ditanyakan.

Aturan:
1. Sebutkan nama jamu yang paling relevan dengan pertanyaan.
2. Jangan mengarang kandungan, khasiat, jenis, atau bentuk sediaan.
3. Bedakan informasi yang tercantum dari informasi yang tidak tersedia.
4. Jika khasiat cocok tetapi jenis atau kandungannya tidak tercantum,
   jelaskan bagian yang cocok dan bagian yang belum dapat dipastikan.
5. Jika ada beberapa jamu relevan, prioritaskan yang paling sesuai
   dengan kombinasi informasi yang ditanyakan.
6. Jawab dalam bahasa Indonesia dengan ringkas dan jelas.

KONTEKS REFERENSI:
{context}

PERTANYAAN:
{query}

JAWABAN:"""


def generate_gemini(prompt, channel="web", max_retries=4, base_delay=3.0):
    last = "tidak ada model"
    for model_name in GEMINI_MODELS:
        for attempt in range(max_retries):
            try:
                r = get_client(channel).models.generate_content(model=model_name, contents=prompt)
                if r.text:
                    return r.text.strip()
                last = f"{model_name}: respons kosong"
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                last = f"{model_name}: {msg[:200]}"
                if any(c in msg for c in RETRY_CODES):
                    time.sleep(base_delay * 2 ** attempt + random.random())
                    continue
                break
    log.error("Gemini gagal: %s", last)
    return None


def rag_answer(query, channel="web"):
    """Return dict: answer, keywords, sources, fallback."""
    hits = retrieve(query)

    context_lines = []
    sources = []

    for d, score in hits:
        row = _by_id.loc[d]

        nama = str(row.get("nama", row.get("NAMA JAMU", "")))
        khasiat = str(row.get("khasiat", row.get("KHASIAT", "")))
        kandungan = str(row.get("kandungan", row.get("KANDUNGAN", "")))
        jenis = str(row.get("jenis", row.get("JENIS", "")))

        context_lines.append(
            f"- Nama jamu: {nama}\n"
            f"  Khasiat: {khasiat}\n"
            f"  Kandungan: {kandungan}\n"
            f"  Jenis: {jenis}"
        )

        sources.append({
            "id": str(d),
            "nama": nama,
            "khasiat": khasiat,
            "kandungan": kandungan,
            "jenis": jenis,
            "score": round(float(score), 4),
        })

    context = "\n".join(context_lines)

    answer = generate_gemini(
        RAG_PROMPT.format(context=context, question=query),
        channel
    )

    fallback = answer is None

    if fallback:
        lines = [
            (
                f"• {item['nama']}: {item['khasiat']}; "
                f"Kandungan: {item['kandungan']}; "
                f"Jenis: {item['jenis']}"
            )
            for item in sources[:3]
        ]
        answer = (
            "Layanan AI sedang sibuk. Berikut data jamu yang relevan:\n\n"
            + "\n".join(lines)
        )

    return {
        "answer": answer.replace("**", ""),
        "keywords": ner_query(query),
        "sources": sources,
        "fallback": fallback,
    }


def kb_size():
    return len(kb)


# ---------------- cache jawaban ----------------
# Pertanyaan yang sama (mis. contoh FAQ) tidak memanggil Gemini lagi -> hemat kuota/biaya.
CACHE_MAX = int(os.environ.get("CACHE_MAX", "500"))
CACHE_TTL = int(os.environ.get("CACHE_TTL_SECONDS", str(24 * 3600)))
_cache = OrderedDict()
_cache_lock = threading.Lock()


def answer_cached(query, channel="web"):
    key = " ".join(re.sub(r"[^\w\s]", " ", query.lower()).split())
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            _cache.move_to_end(key)
            return {**hit[1], "cached": True}
    res = rag_answer(query, channel)
    if not res["fallback"]:                 # jangan simpan jawaban darurat
        with _cache_lock:
            _cache[key] = (now, res)
            while len(_cache) > CACHE_MAX:
                _cache.popitem(last=False)
    return {**res, "cached": False}
