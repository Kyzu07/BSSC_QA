"""Quality Evaluator Agent for BSSC_QA."""
from typing import Dict, Any
from langgraph.prebuilt import create_react_agent

class EvaluatorAgent:
    """Agent for evaluating QA pair quality."""
    
    def __init__(self, llm, validation_tool, quality_threshold: float = 0.75):
        self.llm = llm
        self.validator = validation_tool
        self.quality_threshold = quality_threshold
        
        # Create agent
        self.agent = create_react_agent(
            self.llm,
            [],
            prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Return system prompt for evaluator agent."""
        return """You are a quality evaluation expert. Your task is to assess QA pairs 
across multiple dimensions and provide detailed scores.

Evaluation Criteria:
1. Relevance (0-1): Does the answer address the question?
2. Clarity (0-1): Are both Q&A clear and unambiguous?
3. Completeness (0-1): Is the answer comprehensive?
4. Factuality (0-1): Is the answer accurate based on evidence?

Provide scores for each criterion and identify any issues."""
    
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
        
        prompt = f"""Evaluate this QA pair:

Question: {qa['question']}
Answer: {qa['answer']}

Evidence:
{evidence_text[:500]}

Provide scores (0.0 to 1.0) for:
- Relevance
- Clarity
- Completeness
- Factuality

Format: 
relevance: X.X
clarity: X.X
completeness: X.X
factuality: X.X"""
        
        try:
            result = self.agent.invoke({
                "messages": [{"role": "user", "content": prompt}]
            })
            
            llm_scores = self._parse_scores(result["messages"][-1].content)
            
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
            pattern = f"{metric}\s*:\s*([0-9.]+)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    scores[metric] = float(match.group(1))
                except:
                    scores[metric] = 0.00001  # Default
            else:
                scores[metric] = 0.00001
        
        return scores
