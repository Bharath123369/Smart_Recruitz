import logging
from typing import Dict, Any
from ..state import InterviewValidationState

logger = logging.getLogger(__name__)

def validate_input_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Validates the input transcript and metadata.
    Rejects if transcript is too short or lacks Q&A pairs.
    """
    logger.info(f"Validating input for interview_id: {state.interview_id}")
    
    transcript = state.transcript.strip()
    word_count = len(transcript.split())
    
    errors = []
    
    # Validation Rules
    if word_count < 200:
        errors.append(f"Transcript too short: {word_count} words. Minimum 200 required.")
    
    # Check for Q&A patterns (more inclusive)
    patterns = ["Q:", "Question:", "Interviewer:", "Interviewer :"]
    ans_patterns = ["A:", "Answer:", "Candidate:", "Candidate :"]
    
    has_q = any(p in transcript for p in patterns)
    has_a = any(p in transcript for p in ans_patterns)
    
    if not (has_q and has_a):
        errors.append("Transcript does not follow standard Q&A format.")
    
    # count Q&A pairs (based on interviewer tags)
    qa_count = 0
    for p in patterns:
        qa_count += transcript.count(p)
        
    if qa_count < 5:
        errors.append(f"Insufficient Q&A pairs: {qa_count}. Minimum 5 required.")

    if errors:
        return {
            "errors": errors,
            "audit_trail": [{"node": "validate_input", "status": "FAILED", "errors": errors}]
        }

    return {
        "audit_trail": [{"node": "validate_input", "status": "SUCCESS", "word_count": word_count, "qa_count": qa_count}]
    }
