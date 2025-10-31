"""Answer Synthesis Agent for BSSC_QA."""
from typing import Dict, Any, List
from langgraph.prebuilt import create_react_agent
import uuid

class SynthesisAgent:
    """Agent for synthesizing answers from evidence."""
    
    def __init__(self, llm, vector_store_manager, max_evidence_spans: int = 3):
        self.llm = llm
        self.vs_manager = vector_store_manager
        self.max_evidence_spans = max_evidence_spans
        
        # Create agent with no additional tools (uses retrieval internally)
        self.agent = create_react_agent(
            self.llm,
            [],
            prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Return system prompt for synthesis agent."""
        return """You are an expert answer synthesizer. Your task is to create accurate, 
comprehensive answers based on provided evidence.

Guidelines:
1. Base answers strictly on the evidence provided
2. Be clear, concise, and well-structured
3. Include relevant details and context
4. Maintain factual accuracy
5. Match answer complexity to question complexity

Your answer should:
- Directly address the question
- Use evidence to support claims
- Be complete but not unnecessarily verbose"""
    
    def synthesize_answer(self, question: str, question_type: str = "factual") -> Dict[str, Any]:
        """Generate answer for a question using retrieved evidence.
        
        Args:
            question: The question to answer
            question_type: Type of question (factual, conceptual, analytical)
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve relevant evidence
        evidence_docs = self.vs_manager.similarity_search(question, k=self.max_evidence_spans)
        
        # Format evidence
        evidence_text = self._format_evidence(evidence_docs)
        print(f"🧾 Retrieved these evidences: \n{evidence_text}")
        
        # Generate answer
        prompt = f"""Question: {question}
Question Type: {question_type}

Evidence:
{evidence_text}

Based on the evidence above, provide a clear and accurate answer to the question."""
        
        try:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            
            answer = result["messages"][-1].content
            
            return {
                "qa_id": str(uuid.uuid4()),
                "question": question,
                "answer": answer,
                "evidence_spans": [doc.page_content for doc in evidence_docs],
                "chunk_ids": [doc.metadata.get('chunk_id', '') for doc in evidence_docs],
                "question_type": question_type
            }
            
        except Exception as e:
            print(f"Error synthesizing answer: {e}")
            return {
                "qa_id": str(uuid.uuid4()),
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "evidence_spans": [],
                "chunk_ids": [],
                "question_type": question_type
            }
    
    def _format_evidence(self, docs: List) -> str:
        """Format evidence documents for prompt."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[Evidence {i}]")
            formatted.append(doc.page_content)
            formatted.append("")
        return "\n".join(formatted)
