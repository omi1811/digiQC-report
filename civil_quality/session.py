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
    
    Note: We've made this more lenient - only asks for critical missing info
    when the question specifically requires it. General questions work without context.
    """
    text = question.lower()
    
    # 1. Update Context from Text
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

    # 2. Identify Missing Fields - ONLY for questions that specifically need them
    # Be lenient - most questions can be answered with defaults or general guidance
    
    missing = []
    
    # Only ask for specifics if user is asking about suitability/verification
    is_suitability_check = any([
        "okay" in text, "suitable" in text, "correct" in text,
        "right" in text, "acceptable" in text, "proper" in text,
        "is m" in text, "can i use" in text, "should i" in text
    ])
    
    # For suitability checks, we need grade and exposure at minimum
    if is_suitability_check:
        if not current_context.grade_of_concrete and "grade" not in text:
            missing.append("grade_of_concrete")
        if not current_context.exposure_condition:
            missing.append("exposure_condition")
    
    # Don't require all fields - be helpful with what we have
    # Cement type is almost never critical to ask about
    
    return current_context, missing
