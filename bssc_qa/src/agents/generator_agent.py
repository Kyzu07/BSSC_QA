"""Question Generator Agent for BSSC_QA."""
from typing import List, Dict, Any
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import uuid

class GeneratorAgent:
    """Agent for generating questions from text chunks."""
    
    def __init__(self, llm, retrieval_tool, chunk_analysis_tool):
        self.llm = llm
        self.retrieval_tool = retrieval_tool
        self.chunk_analyzer = chunk_analysis_tool
        
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
- Suggested question types: {', '.join(suggested_types)}"""
        
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
        
        # Create agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Return system prompt for generator agent."""
        return """You are an expert question generator. Your task is to create high-quality, 
diverse questions from given text content.

Guidelines:
1. Questions should be clear, specific, and answerable from the content
2. Vary question types: factual, conceptual, analytical
3. Ask about key concepts, entities, and relationships
4. Ensure questions test understanding, not just recall
5. Each question should be complete and grammatically correct

Output format for each question:
{
  "question": "Your question here?",
  "type": "factual|conceptual|analytical",
  "rationale": "Why this question is valuable"
}"""
    
    def generate_questions(self, chunk_text: str, count: int = 3) -> List[Dict[str, Any]]:
        """Generate questions from a text chunk.
        
        Args:
            chunk_text: Source text content
            count: Number of questions to generate
            
        Returns:
            List of question dictionaries
        """
        prompt = f"""Generate {count} diverse, high-quality questions from this content:

{chunk_text}

Provide exactly {count} questions in the specified JSON format."""
        
        try:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            
            # Parse response
            response_text = result["messages"][-1].content
            
            # Simple parsing for now (can be enhanced)
            questions = self._parse_questions(response_text, chunk_text, count)
            
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
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
