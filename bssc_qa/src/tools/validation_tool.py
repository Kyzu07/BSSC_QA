"""Validation tool for QA pair quality assessment."""
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of QA validation."""
    scores: Dict[str, float]
    overall_score: float
    flags: List[str]
    passed: bool

class ValidationTool:
    """Tool for validating QA pairs."""
    
    def __init__(self, quality_threshold: float = 0.75):
        self.quality_threshold = quality_threshold
    
    def validate_qa(self, question: str, answer: str, 
                   evidence: str = None) -> ValidationResult:
        """Validate a QA pair across multiple metrics.
        
        Args:
            question: The question text
            answer: The answer text
            evidence: Optional evidence text for factuality check
            
        Returns:
            ValidationResult with scores and flags
        """
        scores = {}
        flags = []
        
        # 1. Length checks
        if len(question.strip()) < 10:
            flags.append("question_too_short")
            scores['length'] = 0.3
        elif len(question.strip()) > 300:
            flags.append("question_too_long")
            scores['length'] = 0.7
        else:
            scores['length'] = 1.0
        
        if len(answer.strip()) < 20:
            flags.append("answer_too_short")
            scores['answer_length'] = 0.3
        else:
            scores['answer_length'] = 1.0
        
        # 2. Basic quality checks
        if not question.strip().endswith('?'):
            flags.append("missing_question_mark")
            scores['format'] = 0.8
        else:
            scores['format'] = 1.0
        
        # 3. Relevance (simple keyword overlap check)
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words) / len(q_words) if q_words else 0
        scores['relevance'] = min(overlap * 2, 1.0)  # Scale to 0-1
        
        # 4. Completeness (answer should be substantial)
        word_count = len(answer.split())
        if word_count < 5:
            scores['completeness'] = 0.3
            flags.append("answer_too_brief")
        elif word_count < 15:
            scores['completeness'] = 0.6
        else:
            scores['completeness'] = 1.0
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        passed = overall_score >= self.quality_threshold
        
        return ValidationResult(
            scores=scores,
            overall_score=overall_score,
            flags=flags,
            passed=passed
        )
