import re
import logging
from typing import Dict, Any
from ..state import InterviewValidationState

logger = logging.getLogger(__name__)

# Regex patterns for PII
PII_PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b',
    "aadhaar": r'\b\d{4}\s\d{4}\s\d{4}\b',
    "pan": r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
}

def strip_pii_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Strips PII (emails, phones, Aadhaar, PAN) from the transcript.
    """
    logger.info(f"Stripping PII for interview_id: {state.interview_id}")
    
    text = state.transcript
    stripped_types = []
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            stripped_types.append(pii_type)
            text = re.sub(pattern, f"[STRIPPED_{pii_type.upper()}]", text)
            
    return {
        "pii_stripped_transcript": text,
        "audit_trail": [{"node": "strip_pii", "status": "SUCCESS", "stripped_types": stripped_types}]
    }
