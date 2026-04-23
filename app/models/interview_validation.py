from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from .enums import ReadinessLevel, L1Status, TalentPoolAction, AnswerQuality

class AnswerEvaluation(BaseModel):
    question_id: str
    question_text: str
    answer_text: str
    score: float = Field(ge=0, le=10, description="Score from 0 to 10")
    feedback: str = Field(description="Detailed feedback on the answer")
    quality: AnswerQuality

class DimensionScores(BaseModel):
    technical_depth: float = Field(default=0.0, description="Score for technical knowledge")
    experience_relevance: float = Field(default=0.0, description="Score for relevant work experience")
    communication_skills: float = Field(default=0.0, description="Score for clarity and articulation")

class InterviewValidationResponse(BaseModel):
    interview_id: str
    candidate_id: str
    overall_score: float
    readiness_level: ReadinessLevel
    l1_status: L1Status
    talent_pool_action: TalentPoolAction
    recommendation: str
    dimension_scores: DimensionScores
    evaluations: List[AnswerEvaluation]
    completed_at: datetime
