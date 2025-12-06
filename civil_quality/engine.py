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
    # Augment search query with relevant IS code identifiers to bias results to standard references
    try:
        kb_matches = llm_client._is_question_in_kb(question)
        if kb_matches and kb_matches.get('is_codes'):
            search_query += ' ' + ' '.join(kb_matches.get('is_codes'))
    except Exception:
        # fallback - don't block
        pass

    # Add targeted domain constraints for technical queries for better quality
    technical_keywords = ['is ', 'is code', 'standard', 'waterproof', 'membrane', 'epoxy', 'tiling', 'plumbing', 'electrical', 'fire', 'safety', 'earthing']
    if any(k in search_query.lower() for k in technical_keywords):
        search_query += ' site:bis.gov.in OR site:gov.in filetype:pdf'

    web_results = search_tool.search(search_query)

    # Prioritize web results from authoritative sources (BIS, gov) and PDF/edu
    def _score_result(r):
        url = (r.get('url') or '').lower()
        score = 0
        if 'bis.gov.in' in url or '.gov.in' in url or '.nic.in' in url:
            score += 10
        if url.endswith('.pdf') or 'pdf' in url:
            score += 5
        if 'researchgate' in url or '.edu' in url or '.org' in url:
            score += 3
        # Demote forums and gaming pages
        for bad in ['dofus', 'reddit', 'forum', 'games']:
            if bad in url:
                score -= 10
        return score

    web_results = sorted(web_results, key=_score_result, reverse=True)
    
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
