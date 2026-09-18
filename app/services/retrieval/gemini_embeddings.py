import time
import logfire
from langchain_google_google_genai import GoogleGenAIEmbeddings
from app.config import settings

BATCH_SIZE=50
_GEMINI_EMBEDDING_DIMENSIONS=3072
_FALLBACK_EMBEDDING_DIMENSION = 768

active_model =None
model_type:str|None = None
def probe_gemini():
    try:
        model = GoogleGenAIEmbeddings(model="gemini-embedding-2", google_api_key=settings.GEMINI_API_KEY)
        model.embedded_query("probe")
        logfire.info("Gemini created embeddings read(models/gemini)")
        return model
    except Exception as e:
        logfire.warning(f"Gemini probe failed {e}")
        return None

def load_fallback():
    from sentence_transformers import SentenceTransformer
    logfire.info("Loading fall back ")
    return SentenceTransformer("all-mpnet-base-v2")
def _init():
    global active_model, model_type
    if active_model is not None:
        return
    gemini = probe_gemini()
    if gemini:
        active_model = gemini
        model_type = "gemini"
    else:
        active_model = load_fallback()
        model_type = "fallback"

def get_embedding_dim()->int:
    _init()
    return _GEMINI_EMBEDDING_DIMENSIONS if model_type== "gemini" else _FALLBACK_EMBEDDING_DIMENSION

def get_embedding_batch(batch:list[str])->list[list[float]]:
    if model_type=="gemini":
        for attempt in range(4):
            try:
                
                return active_model.embed_documents(batch)
            except Exception as e:
                err=str(e).lower
                is_rate_limit=any(x in err for x in ("429","rate","quota","resource_exhausted"))
                if is_rate_limit and attempt<3:
                    wait = 2**attempt
                    logfire.warn(
                        f"Gemini rate limit hit, retrying in {wait} seconds",
                        f"attempt {attempt}/4")
                    time.sleep(wait)
                else:
                    logfire.error("Gemini embedding  failed  ")
            raise RuntimeError("Gemini rate limit persisted 4 attempts")

    else:
        return active_model.encode(batch, show_progress_bar=False).tolist()

def embeded_texts():
    _init()
    all_embeddings:list[list[float]] = []
    for i in range(0,len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        with logfire.span("EMBEDED BATCH", model=model_type ,start=i, size = len(texts)):
            all_embeddings.extend(_embeded_batch(batch))
    return all_embeddings

def embedded_query(text:str)->list[float]:
    _init()
    if model_type=="gemini":
        return active_model.embedded_query(text)
    else:
        return active_model.encode([text])[0].tolist()
