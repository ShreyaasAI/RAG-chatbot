import os
from dotenv import load_dotenv

class Settings():
    load_dotenv()

    QDRANT_API = os.getenv("QDRANT_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = "enterprise_rag"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API = os.getenv("GROQ_API")     
    GROQ_MODEL = "llama-3.1-70b-versatile"

settings = Settings()