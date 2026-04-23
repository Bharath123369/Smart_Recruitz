import logging
import json
import asyncio
from typing import Dict, Any
from ..state import InterviewValidationState

logger = logging.getLogger(__name__)

async def mock_score_answers_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Mock node that simulates Claude 3.5 Sonnet's response for testing.
    """
    logger.info(f"[MOCK] Scoring answers for interview_id: {state.interview_id}")
    
    # Simulate API latency
    await asyncio.sleep(1)
    
    # Mock result based on transcript length as a simple heuristic
    word_count = len(state.transcript.split())
    base_score = min(0.9, word_count / 1000) 
    
    mock_result = {
        "evaluations": [
            {
                "question_id": "Q1",
                "question_text": "Tell me about your experience with FastAPI",
                "answer_text": "I have built several microservices...",
                "score": 8.5,
                "feedback": "Strong technical explanation.",
                "quality": "EXCELLENT"
            }
        ],
        "dimension_scores": {
            "technical_depth": base_score,
            "experience_relevance": min(1.0, base_score + 0.1),
            "communication_skills": 0.85
        }
    }
    
    return {
        "answer_evaluations": mock_result["evaluations"],
        "dimension_scores": mock_result["dimension_scores"],
        "audit_trail": [{"node": "score_answers", "status": "SUCCESS", "mode": "MOCK"}]
    }
