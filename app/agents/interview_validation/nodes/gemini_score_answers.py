import logging
import json
from typing import Dict, Any
from app.agents.interview_validation.state import InterviewValidationState
from app.agents.interview_validation.prompts import INTERVIEW_VALIDATION_SYSTEM_PROMPT, INTERVIEW_VALIDATION_USER_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

def gemini_score_answers_node(state: InterviewValidationState) -> Dict[str, Any]:
    """
    Calls Google Gemini (Free Tier) to score the transcript for testing purposes.
    """
    logger.info(f"Scoring answers for interview_id: {state.interview_id} using Gemini")
    
    # Using Gemini Pro (2.5 Flash as standard)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = INTERVIEW_VALIDATION_USER_PROMPT.format(
        transcript=state.pii_stripped_transcript,
        questions=state.questions_with_rubric
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=INTERVIEW_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ])
        
        # Clean markdown framing if Gemini returns it
        content = response.content
        if content.startswith("```json"):
            content = content[7:-3]
            
        result = json.loads(content)
        
        # Normalize keys for compute_score node (e.g., "Technical Depth" -> "technical_depth")
        raw_scores = result.get("dimension_scores", {})
        normalized_scores = {}
        for k, v in raw_scores.items():
            norm_key = k.lower().replace(" ", "_").strip()
            normalized_scores[norm_key] = v

        return {
            "answer_evaluations": result.get("evaluations", []),
            "dimension_scores": normalized_scores,
            "audit_trail": [{"node": "score_answers", "status": "SUCCESS", "model": "gemini"}]
        }
    except Exception as e:
        logger.error(f"Error calling Gemini: {str(e)}")
        return {
            "errors": [f"AI Scoring failed: {str(e)}"],
            "audit_trail": [{"node": "score_answers", "status": "FAILED", "error": str(e)}]
        }
