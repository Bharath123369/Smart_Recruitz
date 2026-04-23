from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
import operator
from app.models.enums import ReadinessLevel, L1Status, TalentPoolAction

class InterviewValidationState(BaseModel):
    # Input Group
    interview_id: str
    candidate_id: str
    position: str
    interview_type: str
    transcript: str
    questions_with_rubric: List[Dict[str, Any]]
    
    # Processing Group
    pii_stripped_transcript: Optional[str] = None
    answer_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    dimension_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Output Group
    overall_score: float = 0.0
    readiness_level: Optional[ReadinessLevel] = None
    l1_status: L1Status = L1Status.PENDING
    recommendation: str = ""
    talent_pool_action: Optional[TalentPoolAction] = None
    
    # Accumulating Fields
    errors: Annotated[List[str], operator.add] = Field(default_factory=list)
    audit_trail: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    
    # Metadata
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    processing_time_ms: int = 0
