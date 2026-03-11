import os
import uvicorn
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import timedelta

# Import local modules
import database
from auth import router as auth_router, get_current_user_id
from services.url_parser import extract_article_from_url, validate_url
from services.tavily_service import search_news_evidence
from services.openrouter_service import analyze_news_with_genai
from services.history_service import (
    save_full_verification,
    get_user_verifications,
    get_verification_by_id,
    delete_user_verification,
    get_user_statistics,
    format_verification_for_response
)
from models import ml_model

# Initialize FastAPI app (Trigger reload)
app = FastAPI(title="AI Fake News Analysis API")

# Custom error handler for FastAPI's default validation errors to match frontend format
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
        )
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)

# Initialize database
database.init_db()

# ============ Pydantic Models ============
class PredictRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    category: Optional[str] = None

# ============== Prediction Endpoints ==============

@app.post("/api/predict")
async def predict(request: PredictRequest, current_user_id: str = Depends(get_current_user_id)):
    """Analyze news text or URL for fake news detection."""
    text = (request.text or "").strip()
    url = (request.url or "").strip() if request.url else None
    
    # Extracted info from URL
    extracted_title = None
    extracted_authors = []
    
    # Handle URL input
    if url and not text:
        validation = validate_url(url)
        if not validation['is_valid']:
            return JSONResponse(status_code=400, content={"error": validation['error']})
        
        extraction = extract_article_from_url(validation['normalized_url'])
        if not extraction['success']:
            return JSONResponse(status_code=400, content={"error": extraction['error']})
        
        text = extraction['text']
        extracted_title = extraction['title']
        extracted_authors = extraction['authors']
        url = validation['normalized_url']
    
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text or URL provided"})
    

    try:
        # 1. Get Evidence via Tavily
        search_query = extracted_title if extracted_title else text[:100]
        evidence = search_news_evidence(search_query)
        
        # 2. Try Gen AI Analysis via OpenRouter first
        ai_result = analyze_news_with_genai(text, evidence)

        if ai_result.get("data") and ai_result.get("meta"):
            # Preferred path: structured GenAI response
            analysis_data = ai_result["data"]
            meta_info = ai_result["meta"]
        else:
            # Fallback: use local ML model so we never hard-fail with 500
            model_result = ml_model.predict(text)
            label = model_result.get("label", "Unknown")
            confidence = float(model_result.get("confidence", 0))
            credibility_score = confidence if label == "Real" else 100.0 - confidence

            analysis_data = {
                "prediction": label,
                "confidence": confidence,
                "credibility_score": credibility_score,
                "reason_summary": model_result.get("note", "Heuristic ML analysis result."),
                "explanation": model_result.get("note", "Heuristic ML analysis result."),
                "suspicious_phrases": [],
            }
            meta_info = {
                "model": model_result.get("model", "ml_model"),
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
                "time_seconds": 0.0,
            }

        # Save to history
        verification_id = save_full_verification(
            user_id=current_user_id,
            text_content=text,
            url=url,
            title=extracted_title,
            prediction_label=analysis_data.get('prediction', 'Unknown'),
            confidence_score=analysis_data.get('confidence', 0),
            credibility_score=analysis_data.get('credibility_score', 0),
            analysis_result={
                'analysis': {
                    'reason_summary': analysis_data.get('reason_summary', ''),
                    'explanation': analysis_data.get('explanation', ''),
                    'suspicious_phrases': analysis_data.get('suspicious_phrases', [])
                },
                'source_analysis': {'domain': url},
                'reason_summary': analysis_data.get('reason_summary', '')
            }
        )
        
        return {
            'verification_id': verification_id,
            'label': analysis_data.get('prediction', 'Unknown'),
            'confidence': analysis_data.get('confidence', 0),
            'credibility_score': analysis_data.get('credibility_score', 0),
            'explanation': analysis_data.get('explanation', ''),
            'reason_summary': analysis_data.get('reason_summary', ''),
            'suspicious_phrases': analysis_data.get('suspicious_phrases', []),
            'ai_meta': meta_info,
            'extracted_info': {
                'title': extracted_title,
                'authors': extracted_authors,
                'url': url
            } if extracted_title else None
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict_public(request: PredictRequest):
    """Public prediction endpoint for backward compatibility."""
    text = (request.text or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})

    try:
        search_query = text[:100]
        evidence = search_news_evidence(search_query)
        ai_result = analyze_news_with_genai(text, evidence)

        if ai_result.get("data"):
            analysis_data = ai_result["data"]
        else:
            # Fallback to local ML model instead of returning 500
            model_result = ml_model.predict(text)
            label = model_result.get("label", "Unknown")
            confidence = float(model_result.get("confidence", 0))
            credibility_score = confidence if label == "Real" else 100.0 - confidence

            analysis_data = {
                "prediction": label,
                "confidence": confidence,
                "credibility_score": credibility_score,
                "suspicious_phrases": [],
                "reason_summary": model_result.get("note", "Heuristic ML analysis result."),
            }

        return {
            "label": analysis_data.get("prediction", "Unknown"),
            "confidence": analysis_data.get("confidence", 0),
            "credibility_score": analysis_data.get("credibility_score", 0),
            "keywords": analysis_data.get("suspicious_phrases", [])[:5],
            "reason_summary": analysis_data.get("reason_summary", ""),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ============== History Endpoints ==============

@app.get("/api/history")
async def get_history(
    limit: int = Query(20, gt=0, le=100),
    offset: int = Query(0, ge=0),
    current_user_id: str = Depends(get_current_user_id)
):
    """Get user's verification history."""
    return get_user_verifications(current_user_id, limit, offset)

@app.get("/api/history/{verification_id}")
async def get_history_detail(verification_id: str, current_user_id: str = Depends(get_current_user_id)):
    """Get detailed verification by ID."""
    verification = get_verification_by_id(verification_id, current_user_id)
    if not verification:
        return JSONResponse(status_code=404, content={"error": "Verification not found"})
    return format_verification_for_response(verification)

@app.delete("/api/history/{verification_id}")
async def delete_history(verification_id: str, current_user_id: str = Depends(get_current_user_id)):
    """Delete a verification record."""
    if not delete_user_verification(verification_id, current_user_id):
        return JSONResponse(status_code=404, content={"error": "Verification not found"})
    return {"message": "Verification deleted successfully"}

@app.get("/api/history/stats")
async def get_stats(current_user_id: str = Depends(get_current_user_id)):
    """Get user statistics."""
    return get_user_statistics(current_user_id)

# ============== Utility Endpoints ==============

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": "Gen AI (OpenRouter)"}

@app.get("/api/model/info")
async def model_info():
    """Get model info."""
    return {
        "name": "Gen AI - Gemini 2.0 Flash",
        "type": "Generative AI",
        "integration": "OpenRouter & Tavily Search"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
