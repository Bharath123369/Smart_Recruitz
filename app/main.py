import os
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
import io
from PyPDF2 import PdfReader
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env vars
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyA417kDUZM5yic1ZJTnq_OSzr23q2GuJHA"

import csv
from app.agents.interview_validation.workflow import InterviewValidationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SmartRecruitz Interview Validation API",
    description="API for validating and scoring interview transcripts using AI.",
    version="1.0.0"
)

# Directory for storing stripped transcripts via API
STRIPPED_STORAGE = os.path.join(os.path.dirname(__file__), "api_stripped_transcripts")
os.makedirs(STRIPPED_STORAGE, exist_ok=True)

# Path for progressive results logging
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "api_live_results.csv")

def log_result_to_csv(result_data: Dict[str, Any]):
    """Appends a single validation result to the CSV file progressively."""
    file_exists = os.path.isfile(RESULTS_CSV)
    fieldnames = ["timestamp", "interview_id", "candidate_id", "overall_score", "readiness_level", "talent_pool_action", "recommendation"]
    
    with open(RESULTS_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "interview_id": result_data.get("interview_id"),
            "candidate_id": result_data.get("candidate_id"),
            "overall_score": result_data.get("overall_score"),
            "readiness_level": result_data.get("readiness_level"),
            "talent_pool_action": result_data.get("talent_pool_action"),
            "recommendation": result_data.get("recommendation")
        })

class QuestionRubric(BaseModel):
    question: str
    rubric: str

class ValidationRequest(BaseModel):
    interview_id: str
    candidate_id: str
    position: str
    interview_type: str = "L1_SCREENING"
    transcript: str
    questions_with_rubric: List[QuestionRubric]



@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/validate")
async def validate_transcript(request: ValidationRequest):
    """
    Triggers the interview validation workflow for a given transcript.
    """
    logger.info(f"Received validation request for interview_id: {request.interview_id}")
    
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    # Prepare input for LangGraph
    input_data = {
        "interview_id": request.interview_id,
        "candidate_id": request.candidate_id,
        "position": request.position,
        "interview_type": request.interview_type,
        "transcript": request.transcript,
        "questions_with_rubric": [q.model_dump() for q in request.questions_with_rubric]
    }

    try:
        # Invoke the agent
        result = await InterviewValidationAgent.ainvoke(input_data)
        
        # Save stripped transcript to disk
        pii_stripped = result.get("pii_stripped_transcript", "")
        if pii_stripped:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_stripped_{request.interview_id}_{timestamp}.txt"
            filepath = os.path.join(STRIPPED_STORAGE, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pii_stripped)
            logger.info(f"Stripped transcript saved to {filepath}")

        # Log to CSV progressively
        log_result_to_csv(result)
        logger.info(f"Result for {request.interview_id} logged to CSV.")

        return result

    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/validate-pdf")
async def validate_pdf(
    interview_id: str = Body(...),
    candidate_id: str = Body(...),
    position: str = Body(...),
    interview_type: str = Body("L1_SCREENING"),
    questions_with_rubric: str = Body(...), # Will parse JSON from string
    file: UploadFile = File(...)
):
    """
    Extracts text from an uploaded PDF and triggers the validation workflow.
    """
    logger.info(f"Received PDF validation request for interview_id: {interview_id}")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Extract text from PDF
        pdf_bytes = await file.read()
        reader = PdfReader(io.BytesIO(pdf_bytes))
        transcript_text = ""
        for page in reader.pages:
            transcript_text += page.extract_text() + "\n"
        
        if not transcript_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # Parse questions_with_rubric JSON string
        try:
            # Clean possible outer quotes or whitespace from Swagger
            clean_rubric = questions_with_rubric.strip()
            if clean_rubric.startswith('"') and clean_rubric.endswith('"'):
                # Handle cases where Swagger might double-wrap the string
                try:
                    clean_rubric = json.loads(clean_rubric)
                except:
                    pass
            
            # Final parsing
            if isinstance(clean_rubric, str):
                questions_data = json.loads(clean_rubric)
            else:
                questions_data = clean_rubric
                
            if not isinstance(questions_data, list):
                raise ValueError("Parsed JSON is not a list")
                
        except Exception as e:
            logger.error(f"JSON Parse Error. Data received: {repr(questions_with_rubric)}")
            raise HTTPException(
                status_code=400, 
                detail=f"questions_with_rubric must be a valid JSON list. Parse error: {str(e)}"
            )

        # Prepare input for LangGraph
        input_data = {
            "interview_id": interview_id,
            "candidate_id": candidate_id,
            "position": position,
            "interview_type": interview_type,
            "transcript": transcript_text,
            "questions_with_rubric": questions_data
        }

        # Invoke the agent
        result = await InterviewValidationAgent.ainvoke(input_data)
        
        # Save stripped transcript to disk
        pii_stripped = result.get("pii_stripped_transcript", "")
        if pii_stripped:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"api_stripped_pdf_{interview_id}_{timestamp}.txt"
            filepath = os.path.join(STRIPPED_STORAGE, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pii_stripped)
            logger.info(f"Stripped transcript (from PDF) saved to {filepath}")

        # Log to CSV progressively
        log_result_to_csv(result)

        return result

    except Exception as e:
        logger.error(f"Error during PDF validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
