import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

def search_news_evidence(query: str):
    """
    Search for evidence related to a news claim using Tavily.
    """
    if not TAVILY_API_KEY:
        print("Warning: TAVILY_API_KEY not set")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    
    try:
        # Search for context and evidence
        search_result = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=False,
            include_images=False
        )
        
        results = []
        for res in search_result.get('results', []):
            results.append({
                "title": res.get("title"),
                "url": res.get("url"),
                "content": res.get("content"),
                "score": res.get("score")
            })
            
        return results
    except Exception as e:
        print(f"Error searching Tavily: {str(e)}")
        return []
