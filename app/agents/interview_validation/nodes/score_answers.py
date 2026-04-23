import logging
import json
from typing import Dict, Any, List
from app.agents.interview_validation.state import InterviewValidationState
from app.agents.interview_validation.prompts import INTERVIEW_VALIDATION_SYSTEM_PROMPT, INTERVIEW_VALIDATION_USER_PROMPT
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

def score_answers_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Calls Claude 3.5 Sonnet to score the transcript.
    """
    logger.info(f"Scoring answers for interview_id: {state.interview_id}")
    
    # In a real scenario, we'd use environment variables for API keys
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    
    prompt = INTERVIEW_VALIDATION_USER_PROMPT.format(
        transcript=state.pii_stripped_transcript,
        questions=state.questions_with_rubric
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=INTERVIEW_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        # Heuristic: Find JSON in response
        result = json.loads(response.content)
        
        return {
            "answer_evaluations": result.get("evaluations", []),
            "dimension_scores": result.get("dimension_scores", {}),
            "audit_trail": [{"node": "score_answers", "status": "SUCCESS"}]
        }
    except Exception as e:
        logger.error(f"Error calling Claude: {str(e)}")
        return {
            "errors": [f"AI Scoring failed: {str(e)}"],
            "audit_trail": [{"node": "score_answers", "status": "FAILED", "error": str(e)}]
        }
