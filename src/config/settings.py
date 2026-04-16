from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    # Groq
    GROQ_API_KEY: str = ""

    # Paths
    PRELOADED_PDFS_DIR: str = "data/preloaded_pdfs"
    VECTOR_STORE_DIR: str = "vector_store"
    COLLECTION_NAME: str = "studyrag_docs"

    # Model
    MODEL_NAME: str = "llama-3.3-70b-versatile"
    MODEL_TEMPERATURE: float = 0.0

    # Chunking
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100

    class Config:
        env_file = ".env"
        extra = "allow"

    @property
    def preloaded_pdfs_abs(self) -> str:
        return os.path.abspath(self.PRELOADED_PDFS_DIR)

    @property
    def vector_store_abs(self) -> str:
        return os.path.abspath(self.VECTOR_STORE_DIR)


# Singleton
settings = Settings()
