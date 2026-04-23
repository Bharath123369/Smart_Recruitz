import os
import asyncio
import csv
from PyPDF2 import PdfReader

# Ensure necessary environment variables are set before importing LangGraph components
from dotenv import load_dotenv
load_dotenv()
# Always override environment with the new working key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA417kDUZM5yic1ZJTnq_OSzr23q2GuJHA"

from app.agents.interview_validation.workflow import InterviewValidationAgent
from app.agents.interview_validation.nodes.gemini_score_answers import gemini_score_answers_node
from langgraph.graph import StateGraph, END
from app.agents.interview_validation.state import InterviewValidationState
from app.agents.interview_validation.nodes.validate_input import validate_input_node
from app.agents.interview_validation.nodes.strip_pii import strip_pii_node
from app.agents.interview_validation.nodes.compute_score import compute_score_node
from app.agents.interview_validation.nodes.talent_pool_gate import talent_pool_gate_node
from app.agents.interview_validation.nodes.finalize import finalize_node


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a given PDF file path."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def build_agent():
    """Builds and compiles the InterviewValidationAgent state graph."""
    test_workflow = StateGraph(InterviewValidationState)
    test_workflow.add_node("validate_input", validate_input_node)
    test_workflow.add_node("strip_pii", strip_pii_node)
    
    # Use the real Gemini scoring node for the pipeline
    test_workflow.add_node("score_answers", gemini_score_answers_node) 
    test_workflow.add_node("compute_score", compute_score_node)
    test_workflow.add_node("talent_pool_gate", talent_pool_gate_node)
    test_workflow.add_node("finalize", finalize_node)
    
    test_workflow.set_entry_point("validate_input")
    test_workflow.add_edge("validate_input", "strip_pii")
    test_workflow.add_edge("strip_pii", "score_answers")
    test_workflow.add_edge("score_answers", "compute_score")
    test_workflow.add_edge("compute_score", "talent_pool_gate")
    test_workflow.add_edge("talent_pool_gate", "finalize")
    test_workflow.add_edge("finalize", END)
    
    return test_workflow.compile()

async def process_pdf_batch(pdf_dir: str, output_csv: str):
    """Iterates through PDFs in a directory and scores them using the pipeline."""
    agent = build_agent()
    
    # Common questions & rubric for testing the pipeline
    test_questions = [
        {"question": "Experience with Python frameworks like FastAPI or Django", "rubric": "Score 1-10 based on depth of framework knowledge"},
        {"question": "System design and databases", "rubric": "Score 1-10 based on understanding of scaling and database choices"}
    ]
    
    results = []
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_files = pdf_files[:1]  # Limit to 1 file for Demo/Rate limiting
    print(f"Demo Mode: Processing {len(pdf_files)} PDF(s) to avoid 429 rate limits.")

    # Directory for stripped transcripts
    stripped_dir = os.path.join(os.path.dirname(__file__), "stripped_transcripts")
    os.makedirs(stripped_dir, exist_ok=True)
    
    for filename in pdf_files:
        print(f"\nProcessing {filename}...")
        pdf_path = os.path.join(pdf_dir, filename)
        
        try:
            transcript_text = extract_text_from_pdf(pdf_path)
            
            # Prepare standard input data payload
            input_data = {
                "interview_id": f"batch-{filename}",
                "candidate_id": "cand-test",
                "position": "Senior Backend Engineer",
                "interview_type": "L1_SCREENING",
                "transcript": transcript_text,
                "questions_with_rubric": test_questions
            }
            
            # Invoke the pipeline
            result = await agent.ainvoke(input_data)
            
            score = result.get('overall_score', 0)
            readiness = result.get('readiness_level', 'UNKNOWN')
            action = result.get('talent_pool_action', 'UNKNOWN')
            reco = result.get('recommendation', 'No recommendation provided.')
            errors = " | ".join(result.get('errors', []))
            
            # Check PII Stripping Success (matches strip_pii.py: [STRIPPED_TYPE])
            pii_stripped = result.get("pii_stripped_transcript", "")
            pii_success = "[STRIPPED_EMAIL]" in pii_stripped or "[STRIPPED_PHONE]" in pii_stripped
            
            # Save the stripped transcript for manual verification
            stripped_filename = f"stripped_{filename.replace('.pdf', '.txt')}"
            with open(os.path.join(stripped_dir, stripped_filename), "w", encoding="utf-8") as f:
                f.write(pii_stripped)
            print(f" -> Stripped transcript saved to: {stripped_filename}")

            print(f" -> Score: {score}, Readiness: {readiness}, Action: {action}")
            print(f" -> AI Recommendation: {reco}")
            
            results.append({
                "Filename": filename,
                "Status": "SUCCESS" if not errors else "PARTIAL",
                "Overall Score": score,
                "Readiness Level": readiness,
                "Action": action,
                "Recommendation": reco,
                "Errors": errors,
                "PII Stripped Successfully": pii_success
            })
            
        except Exception as e:
            print(f" -> FAILED: {e}")
            results.append({
                "Filename": filename,
                "Status": "FAILED",
                "Overall Score": 0,
                "Readiness Level": "ERROR",
                "Action": "ERROR",
                "Errors": str(e),
                "PII Stripped Successfully": False
            })
            
    # Write output to CSV
    if results:
        keys = results[0].keys()
        with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nBatch processing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(__file__), "test_transcripts")
    output_report = os.path.join(os.path.dirname(__file__), "batch_test_results.csv")
    
    if os.path.exists(test_dir):
        asyncio.run(process_pdf_batch(test_dir, output_report))
    else:
        print(f"Error: Directory {test_dir} does not exist. Please place a transcript PDF in it.")
