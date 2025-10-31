"""Data structures for BSSC_QA framework."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Plan:
    """Question generation plan for a chunk."""
    chunk_id: str
    topics: List[str]
    question_types: List[str]
    count: int
    bloom_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class QCandidate:
    """Question candidate before answer generation."""
    question_id: str
    question: str
    chunk_ids: List[str]
    bloom_level: Optional[str] = None
    question_type: str = "factual"
    rationale: str = ""

@dataclass
class QA:
    """Question-Answer pair with evidence."""
    qa_id: str
    question: str
    answer: str
    evidence_spans: List[str]
    chunk_ids: List[str]
    bloom_level: Optional[str] = None

@dataclass
class QA_Validated:
    """Validated QA pair with quality scores."""
    qa_id: str
    question: str
    answer: str
    evidence_spans: List[str]
    chunk_ids: List[str]
    bloom_level: Optional[str]
    scores: Dict[str, float]
    overall_score: float
    flags: List[str]
    passed: bool
