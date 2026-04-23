import logging
from typing import Dict, Any
from ..state import InterviewValidationState
from app.models.enums import ReadinessLevel

logger = logging.getLogger(__name__)

# Weights as per exec_plan.txt: Tech×0.40 + Exp×0.35 + Comm×0.25
WEIGHTS = {
    "technical_depth": 0.40,
    "experience_relevance": 0.35,
    "communication_skills": 0.25
}

def compute_score_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Computes the weighted overall score and assigns a readiness level.
    """
    logger.info(f"Computing final score for interview_id: {state.interview_id}")
    
    dim_scores = state.dimension_scores
    overall_score = 0.0
    
    for dim, weight in WEIGHTS.items():
        # dimension scores from AI are expected as 0.0 - 1.0, 
        # convert to 0-100 scale for final reporting
        val = dim_scores.get(dim, 0.0) * 100
        overall_score += val * weight
        
    # Assign Readiness Level
    level = ReadinessLevel.NOT_READY
    if overall_score >= 80:
        level = ReadinessLevel.HIGH_READY
    elif overall_score >= 60:
        level = ReadinessLevel.INTERVIEW_READY
    elif overall_score >= 55:
        level = ReadinessLevel.BORDERLINE
        
    return {
        "overall_score": round(overall_score, 2),
        "readiness_level": level,
        "audit_trail": [{"node": "compute_score", "status": "SUCCESS", "score": overall_score}]
    }
