import logging
import requests
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class WebSearch:
    def __init__(self):
        self.enabled = True
        self.use_lib = False
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.use_lib = True
        except ImportError:
            logger.warning("duckduckgo_search not installed. Using fallback HTTP search.")
            self.use_lib = False

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search the web for the query.
        Returns list of {title, snippet, url}.
        """
        if not self.enabled:
            return []

        if self.use_lib:
            try:
                results = self.ddgs.text(query, max_results=max_results)
                formatted = []
                for r in results:
                    formatted.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", "")
                    })
                return formatted
            except Exception as e:
                logger.error(f"DDGS Search failed: {e}")
                # Fallthrough to fallback
        
        # Fallback: Simple HTML scrape of DuckDuckGo Lite
        return self._fallback_search(query, max_results)

    def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
            
            if resp.status_code != 200:
                return []

            # Simple regex to extract results (Fragile but works for fallback)
            # Look for result links
            results = []
            # Pattern for result blocks
            # This is very rough scraping
            links = re.findall(r'<a class="result__a" href="([^"]+)">([^<]+)</a>', resp.text)
            snippets = re.findall(r'<a class="result__snippet" href="[^"]+">([^<]+)</a>', resp.text)
            
            for i in range(min(len(links), max_results)):
                href, title = links[i]
                snippet = snippets[i] if i < len(snippets) else ""
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": href
                })
            
            return results
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

# Singleton instance
search_tool = WebSearch()
