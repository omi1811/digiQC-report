import re
from typing import Tuple, List
from .schemas import ContextState

REQUIRED_SLOTS = {
    "member_type": ["slab", "beam", "column", "footing", "wall", "pavement", "foundation", "raft", "pile", "staircase", "lintel", "chajja"],
    "grade_of_concrete": ["m10", "m15", "m20", "m25", "m30", "m35", "m40", "m45", "m50", "m60", "m80"],
    "exposure_condition": ["mild", "moderate", "severe", "very severe", "extreme"],
    "cement_type": ["opc", "ppc", "psc", "src", "rhc", "opc 43", "opc 53"],
}

def analyze_slots(question: str, current_context: ContextState) -> Tuple[ContextState, List[str]]:
    """
    Analyze user question to fill slots in the context.
    Returns updated context and list of missing fields.
    
    IMPORTANT: We are now very lenient - we almost NEVER block the user.
    Only ask for additional info if absolutely required for a specific check.
    """
    text = question.lower()
    
    # 1. Update Context from Text (passively collect info)
    if not current_context.member_type:
        for val in REQUIRED_SLOTS["member_type"]:
            if val in text:
                current_context.member_type = val
                break
    
    if not current_context.grade_of_concrete:
        # Regex for Grade (e.g., M25, M-25, M 30)
        match = re.search(r'\b(m\s?-?\s?\d{2})\b', text)
        if match:
            grade = match.group(1).upper().replace(" ", "").replace("-", "")
            current_context.grade_of_concrete = grade
    
    if not current_context.exposure_condition:
        for val in REQUIRED_SLOTS["exposure_condition"]:
            if val in text:
                current_context.exposure_condition = val
                break
                
    if not current_context.cement_type:
        for val in REQUIRED_SLOTS["cement_type"]:
            if val in text:
                current_context.cement_type = val.upper()
                break

    # 2. NEVER block the user - return empty missing list
    # The knowledge engine can handle any query with defaults
    # This ensures the user always gets an answer
    missing = []
    
    return current_context, missing
