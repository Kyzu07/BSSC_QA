"""Answer Synthesis Agent for BSSC_QA."""
from typing import Dict, Any, List, Optional
import logging
import uuid

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from utils.prompt_loader import PromptManager

class SynthesisAgent:
    """Agent for synthesizing answers from evidence."""
    
    def __init__(self, llm, vector_store_manager, max_evidence_spans: int = 3,
                 prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.vs_manager = vector_store_manager
        self.max_evidence_spans = max_evidence_spans
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompts = self.prompt_manager.get_agent_prompts("synthesis")
        
        self.agent = self._initialize_agent()
    
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
        # print(f"🧾 Retrieved these evidences: \n{evidence_text[:256]}")
        
        # Generate answer
        prompt = self.prompt_manager.render(
            "synthesis",
            "user_template",
            question=question,
            question_type=question_type,
            evidence=evidence_text
        )
        print(f"🧠 Synthesis prompt: \n{prompt[:512]}")
        
        try:
            answer = self._invoke_prompt(prompt)
            
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

    def _initialize_agent(self):
        """Attempt to create a ReAct agent, falling back if tools are unsupported."""
        try:
            return create_react_agent(
                self.llm,
                [],
                prompt=self.prompts["system"]
            )
        except NotImplementedError:
            logging.warning(
                "LLM %s does not support tool binding; falling back to direct prompting.",
                type(self.llm).__name__,
            )
        except Exception as exc:
            logging.warning(
                "Falling back to direct prompting for synthesis agent: %s",
                exc,
            )
        return None

    def _invoke_prompt(self, prompt: str) -> str:
        """Run the configured agent or call the LLM directly, returning answer text."""
        if self.agent is not None:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            return self._content_to_string(result["messages"][-1].content)

        messages = [
            SystemMessage(content=self.prompts["system"]),
            HumanMessage(content=prompt)
        ]
        result = self.llm.invoke(messages)
        content = getattr(result, "content", result)
        return self._content_to_string(content)

    def _content_to_string(self, content: Any) -> str:
        """Normalize response payload to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            segments = []
            for item in content:
                if isinstance(item, str):
                    segments.append(item)
                elif isinstance(item, dict):
                    segments.append(item.get("text") or item.get("content") or "")
                else:
                    segments.append(str(item))
            return "\n".join(filter(None, segments))
        return str(content)
