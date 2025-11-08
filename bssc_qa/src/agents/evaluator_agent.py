"""Quality Evaluator Agent for BSSC_QA."""
from typing import Dict, Any, Optional
import logging

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from utils.prompt_loader import PromptManager

class EvaluatorAgent:
    """Agent for evaluating QA pair quality."""
    
    def __init__(self, llm, validation_tool, quality_threshold: float = 0.75,
                 prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.validator = validation_tool
        self.quality_threshold = quality_threshold
        self.prompt_manager = prompt_manager or PromptManager()
        self.prompts = self.prompt_manager.get_agent_prompts("evaluator")
        
        self.agent = self._initialize_agent()
    
    def evaluate_qa(self, qa: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a QA pair and return detailed scores.
        
        Args:
            qa: QA dictionary with question, answer, and evidence
            
        Returns:
            Evaluation results with scores and flags
        """
        # First do rule-based validation
        basic_validation = self.validator.validate_qa(
            qa['question'],
            qa['answer']
        )
        
        # If basic validation fails badly, return early
        if basic_validation.overall_score < 0.3:
            return {
                **qa,
                'scores': basic_validation.scores,
                'overall_score': basic_validation.overall_score,
                'flags': basic_validation.flags,
                'passed': False
            }
        
        # LLM-based evaluation for deeper analysis
        evidence_text = "\n\n".join(qa.get('evidence_spans', []))
        
        prompt = self.prompt_manager.render(
            "evaluator",
            "user_template",
            question=qa['question'],
            answer=qa['answer'],
            evidence=evidence_text
        )
        
        try:
            response_text = self._invoke_prompt(prompt)
            
            llm_scores = self._parse_scores(response_text)
            
            # Combine rule-based and LLM scores
            combined_scores = {
                **basic_validation.scores,
                **llm_scores
            }
            
            overall_score = sum(combined_scores.values()) / len(combined_scores)
            passed = overall_score >= self.quality_threshold
            
            return {
                **qa,
                'scores': combined_scores,
                'overall_score': overall_score,
                'flags': basic_validation.flags,
                'passed': passed
            }
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            # Fallback to basic validation
            return {
                **qa,
                'scores': basic_validation.scores,
                'overall_score': basic_validation.overall_score,
                'flags': basic_validation.flags + ['llm_evaluation_failed'],
                'passed': basic_validation.passed
            }
    
    def _parse_scores(self, response: str) -> Dict[str, float]:
        """Parse LLM response to extract scores."""
        import re
        
        scores = {}
        metrics = ['relevance', 'clarity', 'completeness', 'factuality']
        
        for metric in metrics:
            pattern = rf"{metric}\s*:\s*([0-9.]+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    scores[metric] = float(match.group(1))
                except:
                    scores[metric] = 0.00001  # Default
            else:
                scores[metric] = 0.00001
        
        return scores

    def _initialize_agent(self):
        """Attempt to create ReAct agent, with fallback to direct LLM prompting."""
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
                "Falling back to direct prompting for evaluator agent: %s",
                exc,
            )
        return None

    def _invoke_prompt(self, prompt: str) -> str:
        """Run the configured agent or call the LLM directly."""
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
        """Normalize an LLM/agent response payload into text."""
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
