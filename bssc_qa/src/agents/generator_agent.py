"""Question Generator Agent for BSSC_QA."""
from typing import List, Dict, Any, Optional
import logging
import uuid

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from utils.prompt_loader import PromptManager

class GeneratorAgent:
    """Agent for generating questions from text chunks."""
    
    def __init__(self, llm, retrieval_tool, chunk_analysis_tool,
                 prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.retrieval_tool = retrieval_tool
        self.chunk_analyzer = chunk_analysis_tool
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompts = self.prompt_manager.get_agent_prompts("generator")
        
        # Define tools for agent
        @tool
        def analyze_content(chunk_text: str) -> str:
            """Analyze chunk to understand content and suggest question types."""
            analysis = self.chunk_analyzer.analyze_chunk(chunk_text)
            suggested_types = self.chunk_analyzer.suggest_question_types(analysis)
            
            return f"""Content Analysis:
                    - Sentences: {analysis['sentence_count']}
                    - Key entities: {', '.join(analysis['entities'][:5])}
                    - Contains numbers: {analysis['has_numbers']}
                    - Suggested question types: {', '.join(suggested_types)}
                    """
        
        self.tools = [analyze_content]
        if self.retrieval_tool is not None:
            # Accept both helper class with get_tool() and raw LangChain tool instances
            retrieve_fn = getattr(self.retrieval_tool, "get_tool", None)
            if callable(retrieve_fn):
                retrieval_tool_instance = retrieve_fn()
            else:
                retrieval_tool_instance = self.retrieval_tool
            if retrieval_tool_instance is not None:
                self.tools.append(retrieval_tool_instance)
        
        self.agent = self._initialize_agent()
    
    def generate_questions(self, chunk_text: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate questions from a text chunk.
        
        Args:
            chunk_text: Source text content
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        prompt = self.prompt_manager.render(
            "generator",
            "user_template",
            chunk_text=chunk_text,
            count=count
        )
        
        try:
            response_text = self._invoke_prompt(prompt)
            
            # Simple parsing for now (can be enhanced)
            questions = self._parse_questions(response_text, chunk_text, count)
            
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def _content_to_string(self, content: Any) -> str:
        """Normalize agent message content into a plain string."""
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
    
    def _parse_questions(self, response: str, chunk_text: str, count: int) -> List[Dict[str, Any]]:
        """Parse LLM response into structured questions."""
        import json
        import re
        
        questions = []
        
        # Try to extract JSON objects
        json_pattern = r'\{[^{}]*"question"[^{}]*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches[:count]:
            try:
                q_dict = json.loads(match)
                questions.append({
                    "question_id": str(uuid.uuid4()),
                    "question": q_dict.get("question", ""),
                    "question_type": q_dict.get("type", "factual"),
                    "rationale": q_dict.get("rationale", ""),
                    "chunk_text": chunk_text[:200]  # Store snippet
                })
            except:
                continue
        
        # Fallback: extract questions ending with ?
        if len(questions) < count:
            question_pattern = r'([A-Z][^.!?]*\?)'
            found = re.findall(question_pattern, response)
            for q in found[:count - len(questions)]:
                questions.append({
                    "question_id": str(uuid.uuid4()),
                    "question": q.strip(),
                    "question_type": "factual",
                    "rationale": "Generated question",
                    "chunk_text": chunk_text[:200]
                })
        
        return questions[:count]

    def _initialize_agent(self):
        """Attempt to create a ReAct agent, fallback to direct prompting on failure."""
        try:
            return create_react_agent(
                self.llm,
                self.tools,
                prompt=self.prompts["system"]
            )
        except NotImplementedError:
            logging.warning(
                "LLM %s does not support tool binding; falling back to direct prompting.",
                type(self.llm).__name__,
            )
        except Exception as exc:
            logging.warning(
                "Falling back to direct prompting because create_react_agent failed: %s",
                exc,
            )
        return None

    def _invoke_prompt(self, prompt: str) -> str:
        """Run either the ReAct agent or a direct LLM call and return the response text."""
        if self.agent is not None:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            return self._content_to_string(result["messages"][-1].content)

        # Direct call fallback
        messages = [
            SystemMessage(content=self.prompts["system"]),
            HumanMessage(content=prompt)
        ]
        result = self.llm.invoke(messages)
        content = getattr(result, "content", result)
        return self._content_to_string(content)
