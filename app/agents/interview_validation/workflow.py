import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import InterviewValidationState
from .nodes.validate_input import validate_input_node
from .nodes.strip_pii import strip_pii_node
from .nodes.gemini_score_answers import gemini_score_answers_node
from .nodes.compute_score import compute_score_node
from .nodes.talent_pool_gate import talent_pool_gate_node
from .nodes.finalize import finalize_node

logger = logging.getLogger(__name__)

def create_interview_validation_workflow():
    """
    Creates the LangGraph workflow for interview validation.
    """
    workflow = StateGraph(InterviewValidationState)
    
    # Add Nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("strip_pii", strip_pii_node)
    workflow.add_node("score_answers", gemini_score_answers_node)
    workflow.add_node("compute_score", compute_score_node)
    workflow.add_node("talent_pool_gate", talent_pool_gate_node)
    workflow.add_node("finalize", finalize_node)
    
    # Define Edges
    workflow.set_entry_point("validate_input")
    workflow.add_edge("validate_input", "strip_pii")
    workflow.add_edge("strip_pii", "score_answers")
    workflow.add_edge("score_answers", "compute_score")
    workflow.add_edge("compute_score", "talent_pool_gate")
    workflow.add_edge("talent_pool_gate", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compilation
    return workflow.compile()

InterviewValidationAgent = create_interview_validation_workflow()
