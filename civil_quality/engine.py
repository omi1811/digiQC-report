from typing import Dict, Any
from .schemas import ContextState, ChatResponse
from .session import analyze_slots
from .search import search_tool
from .llm import llm_client, SYSTEM_PROMPT

def process_query(question: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for Civil Quality AI.
    """
    # 1. Rehydrate Context
    context = ContextState.from_dict(context_data)
    
    # 2. Analyze Slots
    context, missing_fields = analyze_slots(question, context)
    
    # 3. Check if we need more info
    if missing_fields:
        # Generate a follow-up question
        missing_str = ", ".join(missing_fields).replace("_", " ")
        reply = f"To give you a specific answer, I need a bit more detail about: {missing_str}. Could you provide that?"
        
        return ChatResponse(
            reply=reply,
            follow_up_needed=True,
            missing_fields=missing_fields,
            context_state=context.to_dict()
        ).__dict__

    # 4. Execution (Search + LLM)
    # Construct search query: Question + Key Context
    search_query = question
    if context.grade_of_concrete: search_query += f" {context.grade_of_concrete}"
    if context.member_type: search_query += f" {context.member_type}"
    if context.exposure_condition: search_query += f" {context.exposure_condition}"
    
    # Search
    web_results = search_tool.search(search_query)
    
    # Build Prompt
    user_prompt = llm_client.build_prompt(context.to_dict(), web_results, question)
    
    # Call LLM
    llm_response = llm_client.generate(SYSTEM_PROMPT, user_prompt)
    
    # Extract Sources from LLM response or just pass web results
    # The LLM is instructed to list sources, but we can also pass the raw sources in the response object
    sources = [{"title": r["title"], "url": r["url"]} for r in web_results]
    
    return ChatResponse(
        reply=llm_response,
        follow_up_needed=False,
        context_state=context.to_dict(),
        sources=sources
    ).__dict__
