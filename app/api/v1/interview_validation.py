from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.interview_validation_service import interview_validation_service

router = APIRouter(prefix="/interview-validation", tags=["Interview Validation"])

@router.post("/validate")
async def validate_interview(payload: Dict[str, Any]):
    """
    Endpoint to trigger interview validation.
    """
    result = await interview_validation_service.validate_interview(payload)
    
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["errors"])
        
    return result["data"]

@router.patch("/override")
async def override_validation(interview_id: str, reason: str, override_status: str):
    """
    Endpoint for HR to override an automated validation result.
    """
    # Placeholder for HR override logic
    return {"message": f"Override successful for {interview_id}", "new_status": override_status}
