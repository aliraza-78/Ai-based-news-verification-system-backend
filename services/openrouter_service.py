import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def analyze_news_with_genai(text: str, search_results: list):
    """
    Analyze news authenticity using OpenRouter Gen AI.
    """
    if not OPENROUTER_API_KEY:
        return {
            "error": "OPENROUTER_API_KEY not set",
            "prediction": "Unknown",
            "confidence": 0,
            "explanation": "AI service configuration missing."
        }

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )

    # Prepare context from search results
    context = ""
    for i, res in enumerate(search_results):
        context += f"\nSource {i+1} ({res['url']}):\n{res['content']}\n"

    prompt = f"""
    Analyze the following news article for authenticity. 
    Use the provided search context as evidence.

    News Article Text:
    {text[:2000]}

    Search Evidence:
    {context}

    Your task is to:
    1. Determine if the news is 'Real', 'Fake', or 'Satire/Opinion'.
    2. Provide a confidence score (0-100).
    3. Provide a detailed explanation of why it is flagged as such.
    4. Highlight suspicious phrases if any.
    5. Summarize the findings.

    Respond STRICTLY in JSON format with the following keys:
    {{
        "prediction": "Real/Fake/Satire",
        "confidence": 85,
        "explanation": "Detailed explanation...",
        "suspicious_phrases": ["phrase 1", "phrase 2"],
        "reason_summary": "Short summary...",
        "credibility_score": 75
    }}
    """

    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001", # High quality and fast
            messages=[
                {"role": "system", "content": "You are a professional fact-checker assisting in fake news detection."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        end_time = time.time()
        
        duration = end_time - start_time
        usage = response.usage
        
        result_content = response.choices[0].message.content
        result_json = json.loads(result_content)
        
        return {
            "data": result_json,
            "meta": {
                "model": response.model,
                "tokens_prompt": usage.prompt_tokens,
                "tokens_completion": usage.completion_tokens,
                "tokens_total": usage.total_tokens,
                "time_seconds": round(duration, 3)
            }
        }
    except Exception as e:
        print(f"Error in Gen AI analysis: {str(e)}")
        return {
            "error": str(e),
            "prediction": "Error",
            "confidence": 0,
            "explanation": "Failed to connect to AI service."
        }
