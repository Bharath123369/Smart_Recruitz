import logging
from typing import Dict, Any
from ..state import InterviewValidationState
from app.models.enums import TalentPoolAction, L1Status

logger = logging.getLogger(__name__)

# Thresholds: ≥60 INSERT · 55–59 HOLD · <55 REJECT
def talent_pool_gate_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Determines the talent pool action and L1 status based on the overall score.
    """
    logger.info(f"Applying talent pool gate for interview_id: {state.interview_id}")
    
    score = state.overall_score
    action = TalentPoolAction.REJECT
    status = L1Status.FAILED
    recommendation = "Candidate does not meet the minimum requirements at this time."
    
    if score >= 60:
        action = TalentPoolAction.INSERT
        status = L1Status.PASSED
        recommendation = "Candidate is highly recommended for the talent pool."
    elif score >= 55:
        action = TalentPoolAction.HOLD
        status = L1Status.NEEDS_REVIEW
        recommendation = "Candidate is borderline. Suggest manual review by HR."
        
    return {
        "talent_pool_action": action,
        "l1_status": status,
        "recommendation": recommendation,
        "audit_trail": [{"node": "talent_pool_gate", "status": "SUCCESS", "action": action}]
    }
