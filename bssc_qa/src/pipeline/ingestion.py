"""Document ingestion pipeline for BSSC_QA framework."""
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from tqdm import tqdm

class IngestionPipeline:
    """Manages document ingestion process."""
    
    def __init__(self, vector_store_manager, chunk_size: int = 512, 
                 chunk_overlap: int = 50):
        self.vs_manager = vector_store_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def ingest_document(self, source: str) -> List[str]:
        """Ingest single document."""
        from pipeline.document_loaders import load_document
        from pipeline.chunking import chunk_text
        from utils.text_processing import normalize_text, clean_text
        
        # Load document
        doc_data = load_document(source)
        
        # Clean and normalize
        content = clean_text(doc_data['content'])
        content = normalize_text(content)
        
        # Chunk text
        chunks = chunk_text(
            content,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            metadata=doc_data['metadata']
        )
        
        # Convert to LangChain documents
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    'chunk_id': chunk.chunk_id,
                    'tokens': chunk.tokens,
                    'position': chunk.position
                }
            )
            for chunk in chunks
        ]
        
        # Add to vector store
        doc_ids = self.vs_manager.add_documents(documents)
        
        return doc_ids
    
    def ingest_directory(self, directory: Path, pattern: str = '*.txt',
                        max_files: int = None) -> Dict[str, Any]:
        """Ingest all files in directory."""
        files = list(directory.glob(pattern))
        
        if max_files:
            files = files[:max_files]
        
        results = {
            'total_files': len(files),
            'processed': 0,
            'failed': 0,
            'total_chunks': 0,
            'errors': []
        }
        
        for file_path in tqdm(files, desc="Ingesting documents"):
            try:
                doc_ids = self.ingest_document(str(file_path))
                results['processed'] += 1
                results['total_chunks'] += len(doc_ids)
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        return results
