"""Retrieval tool for agent access to ChromaDB."""
from langchain_core.tools import StructuredTool

class RetrievalTool:
    """Tool for retrieving relevant chunks from vector store."""
    
    def __init__(self, vector_store_manager):
        self.vs_manager = vector_store_manager
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant text chunks for a given query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            Formatted string with retrieved chunks
        """
        results = self.vs_manager.similarity_search(query, k=k)
        
        output = []
        for i, doc in enumerate(results, 1):
            output.append(f"[Chunk {i}]")
            output.append(f"Source: {doc.metadata.get('filename', 'Unknown')}")
            output.append(f"Position: {doc.metadata.get('position', 'N/A')}")
            output.append(f"Content: {doc.page_content}")
            output.append("")
        
        return "\n".join(output)
    
    def get_tool(self):
        """Return the retrieval method as a LangChain StructuredTool."""
        return StructuredTool.from_function(
            func=self.retrieve_context,
            name="retrieve_context",
            description="Retrieve relevant text chunks for a given query.",
        )
