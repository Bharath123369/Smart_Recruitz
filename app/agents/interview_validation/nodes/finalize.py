import logging
from datetime import datetime
from typing import Dict, Any
from ..state import InterviewValidationState

logger = logging.getLogger(__name__)

def finalize_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Finalizes the state, calculates processing time, and sets completion timestamp.
    """
    logger.info(f"Finalizing validation for interview_id: {state.interview_id}")
    
    completed_at = datetime.now()
    processing_time = int((completed_at - state.started_at).total_seconds() * 1000)
    
    return {
        "completed_at": completed_at,
        "processing_time_ms": processing_time,
        "audit_trail": [{"node": "finalize", "status": "SUCCESS"}]
    }
