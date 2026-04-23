import logging
from typing import Dict, Any
from .agents.interview_validation.workflow import InterviewValidationAgent
from .models.interview_validation import InterviewValidationResponse

logger = logging.getLogger(__name__)

class InterviewValidationService:
    """
    Service layer for orchestrating interview validation.
    """
    
    async def validate_interview(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the interview validation agent workflow.
        """
        logger.info(f"Starting validation for interview_id: {input_data.get('interview_id')}")
        
        try:
            # Execute LangGraph
            final_state = await InterviewValidationAgent.ainvoke(input_data)
            
            if final_state.get("errors"):
                return {"status": "error", "errors": final_state["errors"]}
            
            return {
                "status": "success",
                "data": final_state
            }
        except Exception as e:
            logger.error(f"Error in validation service: {str(e)}")
            return {"status": "error", "errors": [str(e)]}

interview_validation_service = InterviewValidationService()
