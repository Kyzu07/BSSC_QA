"""Vector store management for BSSC_QA framework."""
from pathlib import Path
from typing import List, Optional
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages ChromaDB vector store operations."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: Optional[int] = None,
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = self._sanitize_collection_name(collection_name)
        self.embedding_dimension = embedding_dimension

        # Initialize embeddings based on configuration
        if embedding_model == "offline-hash":
            from core.embeddings import OfflineHashEmbeddings

            dimension = embedding_dimension or 384
            self.embeddings = OfflineHashEmbeddings(dimension=dimension)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model
            )

        # Initialize or load vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store."""
        return self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents."""
        return self.vector_store.similarity_search(query, k=k)

    def get_collection_count(self) -> int:
        """Get total number of documents in collection."""
        return self.vector_store._collection.count()

    def reset_collection(self):
        """Clear all documents from collection and recreate it."""
        client = self.vector_store._client
        try:
            client.delete_collection(name=self.collection_name)
        except Exception:
            # Collection might not exist yet; ignore and re-create
            pass

        # Recreate the collection so subsequent operations work without re-instantiating the manager
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

    @staticmethod
    def _sanitize_collection_name(raw_name: str) -> str:
        """Ensure the collection name meets Chroma's validation requirements."""
        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", raw_name.strip())
        sanitized = sanitized.strip("._-")
        if not sanitized:
            raise ValueError("Collection name must contain at least one alphanumeric character.")
        if len(sanitized) < 3:
            sanitized = sanitized.ljust(3, "0")
        return sanitized[:512]
