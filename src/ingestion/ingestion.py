"""
ingestion.py
------------
Handles:
  1. Preloaded PDFs  →  embedded once, persisted to disk (ChromaDB PersistentClient)
  2. Session uploads →  added incrementally to the same persistent store
  3. Index loading   →  fast reload from disk (no re-embedding if already indexed)
"""

import os
import logging
import hashlib
from typing import Optional

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ── Embed model (module-level singleton, downloaded once) ─────────────────────
_embed_model: Optional[HuggingFaceEmbedding] = None


def get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading HuggingFace embedding model…")
        _embed_model = HuggingFaceEmbedding()
    return _embed_model


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def get_persistent_chroma_client() -> chromadb.PersistentClient:
    """Returns a PersistentClient pointed at VECTOR_STORE_DIR."""
    os.makedirs(settings.vector_store_abs, exist_ok=True)
    return chromadb.PersistentClient(path=settings.vector_store_abs)


def get_or_create_index() -> VectorStoreIndex:
    """
    Load the persistent vector index from disk.
    If it doesn't exist yet, returns an empty index.
    """
    embed_model = get_embed_model()
    _configure_llama_settings(embed_model)

    db = get_persistent_chroma_client()
    collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )
    logger.info(f"Index loaded — collection '{settings.COLLECTION_NAME}' "
                f"has {collection.count()} vectors")
    return index


def _configure_llama_settings(embed_model):
    Settings.embed_model = embed_model
    Settings.llm = Groq(
        model=settings.MODEL_NAME,
        temperature=settings.MODEL_TEMPERATURE,
        api_key=settings.GROQ_API_KEY,
    )


def _file_hash(path: str) -> str:
    """SHA-256 of file content — used to skip already-indexed files."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_indexed(collection: chromadb.Collection, file_hash: str) -> bool:
    """Check metadata for this hash."""
    results = collection.get(where={"file_hash": file_hash}, limit=1)
    return len(results["ids"]) > 0


def ingest_files(file_paths: list[str]) -> dict:
    """
    Embed and store a list of PDF/document file paths into the persistent index.
    Skips files that are already indexed (by content hash).

    Returns:
        {"indexed": [list of new files], "skipped": [list of already-indexed files]}
    """
    embed_model = get_embed_model()
    _configure_llama_settings(embed_model)

    db = get_persistent_chroma_client()
    collection = db.get_or_create_collection(settings.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )

    indexed, skipped = [], []

    for path in file_paths:
        if not os.path.isfile(path):
            logger.warning(f"File not found, skipping: {path}")
            continue

        fhash = _file_hash(path)
        fname = os.path.basename(path)

        if _already_indexed(collection, fhash):
            logger.info(f"Already indexed, skipping: {fname}")
            skipped.append(fname)
            continue

        try:
            reader = SimpleDirectoryReader(input_files=[path])
            documents = reader.load_data()

            # Tag each document with metadata
            for doc in documents:
                doc.metadata["file_name"] = fname
                doc.metadata["file_hash"] = fhash
                doc.metadata["file_path"] = path

            parser = SimpleNodeParser.from_defaults(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )
            nodes = parser.get_nodes_from_documents(documents)

            # Add metadata to nodes too
            for node in nodes:
                node.metadata["file_name"] = fname
                node.metadata["file_hash"] = fhash

            index.insert_nodes(nodes)
            indexed.append(fname)
            logger.info(f"Indexed: {fname}  ({len(nodes)} chunks)")

        except Exception as e:
            logger.error(f"Failed to index {fname}: {e}")

    return {"indexed": indexed, "skipped": skipped}


def ingest_preloaded_pdfs() -> dict:
    """
    Scan PRELOADED_PDFS_DIR and ingest any new PDFs.
    Safe to call on every app startup — skips already-indexed files.
    """
    pdfs_dir = settings.preloaded_pdfs_abs
    os.makedirs(pdfs_dir, exist_ok=True)

    pdf_files = [
        os.path.join(pdfs_dir, f)
        for f in os.listdir(pdfs_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        logger.info(f"No PDFs found in {pdfs_dir}")
        return {"indexed": [], "skipped": []}

    logger.info(f"Found {len(pdf_files)} PDFs in preloaded dir, checking…")
    return ingest_files(pdf_files)


def ingest_uploaded_files(uploaded_files, tmp_dir: str) -> dict:
    """
    Accept Streamlit UploadedFile objects, save to tmp_dir, ingest.
    Returns ingestion summary.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    saved_paths = []

    for uf in uploaded_files:
        dest = os.path.join(tmp_dir, uf.name)
        uf.seek(0)
        with open(dest, "wb") as f:
            f.write(uf.read())
        saved_paths.append(dest)
        logger.info(f"Saved upload → {dest}")

    return ingest_files(saved_paths)


def get_collection_stats() -> dict:
    """Return counts and file names currently in the vector store."""
    try:
        db = get_persistent_chroma_client()
        collection = db.get_or_create_collection(settings.COLLECTION_NAME)
        total = collection.count()

        # Get unique file names from metadata
        if total > 0:
            results = collection.get(include=["metadatas"])
            file_names = sorted({
                m.get("file_name", "unknown")
                for m in results["metadatas"]
                if m.get("file_name")
            })
        else:
            file_names = []

        return {"total_vectors": total, "file_names": file_names, "total_files": len(file_names)}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"total_vectors": 0, "file_names": [], "total_files": 0}
