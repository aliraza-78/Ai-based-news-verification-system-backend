import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pymongo import MongoClient, DESCENDING
from bson.objectid import ObjectId
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/fake_news_db")
DB_NAME = "fake_news_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def init_db():
    """Initialize MongoDB collections and indexes."""
    # Users collection
    db.users.create_index("username", unique=True)
    db.users.create_index("email", unique=True, sparse=True)
    
    # Verification history collection
    db.verification_history.create_index([("user_id", 1), ("created_at", -1)])
    
    # OTP collection
    db.otps.create_index("email", unique=True)
    db.otps.create_index("expires_at", expireAfterSeconds=0)
    
    # Analysis details (can be embedded or separate, sticking to separate for now to match old schema)
    db.analysis_details.create_index("verification_id", unique=True)
    
    # Source analysis
    db.source_analysis.create_index("verification_id", unique=True)
    
    # Trusted sources
    db.trusted_sources.create_index("domain", unique=True)

def save_verification(user_id: str, text_content: str, url: str, title: str,
                      prediction_label: str, confidence_score: float, 
                      credibility_score: float) -> str:
    """Save a verification record and return its ID."""
    verification = {
        "user_id": user_id,
        "text_content": text_content,
        "url": url,
        "title": title,
        "prediction_label": prediction_label,
        "confidence_score": confidence_score,
        "credibility_score": credibility_score,
        "created_at": datetime.utcnow()
    }
    result = db.verification_history.insert_one(verification)
    return str(result.inserted_id)

def save_analysis_details(verification_id: str, suspicious_phrases: list,
                          emotional_tone: float, factual_tone: float,
                          sensational: float, exaggeration: float,
                          neutrality: float, clickbait: float,
                          reason_summary: str, highlighted_text: list):
    """Save analysis details."""
    details = {
        "verification_id": verification_id,
        "suspicious_phrases": suspicious_phrases,
        "emotional_tone_score": emotional_tone,
        "factual_tone_score": factual_tone,
        "sensational_score": sensational,
        "exaggeration_score": exaggeration,
        "neutrality_score": neutrality,
        "clickbait_score": clickbait,
        "reason_summary": reason_summary,
        "highlighted_text": highlighted_text
    }
    db.analysis_details.insert_one(details)

def save_source_analysis(verification_id: str, domain: str, reliability: float,
                         trusted_similarity: float, claim_consistency: float):
    """Save source analysis data."""
    source_data = {
        "verification_id": verification_id,
        "source_domain": domain,
        "source_reliability_score": reliability,
        "trusted_source_similarity": trusted_similarity,
        "claim_consistency_score": claim_consistency
    }
    db.source_analysis.insert_one(source_data)

def get_user_history(user_id: str, limit: int = 20, offset: int = 0) -> list:
    """Get verification history for a user."""
    cursor = db.verification_history.find({"user_id": user_id}) \
                                   .sort("created_at", DESCENDING) \
                                   .skip(offset) \
                                   .limit(limit)
    history = []
    for doc in cursor:
        doc['id'] = str(doc.pop('_id'))
        history.append(doc)
    return history

def get_user_history_count(user_id: str) -> int:
    """Get total count of verifications for a user."""
    return db.verification_history.count_documents({"user_id": user_id})

def get_verification_detail(verification_id: str, user_id: str) -> dict:
    """Get full verification details including analysis."""
    try:
        verification = db.verification_history.find_one({
            "_id": ObjectId(verification_id),
            "user_id": user_id
        })
    except:
        return None
        
    if not verification:
        return None
    
    verification['id'] = str(verification.pop('_id'))
    
    # Get analysis details
    analysis = db.analysis_details.find_one({"verification_id": verification_id})
    if analysis:
        analysis.pop('_id')
        verification['analysis'] = analysis
    
    # Get source analysis
    source = db.source_analysis.find_one({"verification_id": verification_id})
    if source:
        source.pop('_id')
        verification['source_analysis'] = source
        
    return verification

def delete_verification(verification_id: str, user_id: str) -> bool:
    """Delete a verification record."""
    try:
        result = db.verification_history.delete_one({
            "_id": ObjectId(verification_id),
            "user_id": user_id
        })
        if result.deleted_count > 0:
            # Also delete associated details
            db.analysis_details.delete_one({"verification_id": verification_id})
            db.source_analysis.delete_one({"verification_id": verification_id})
            return True
    except:
        pass
    return False

def get_user_stats(user_id: str) -> dict:
    """Get user statistics."""
    total = db.verification_history.count_documents({"user_id": user_id})
    fake_count = db.verification_history.count_documents({
        "user_id": user_id,
        "prediction_label": "Fake"
    })
    real_count = total - fake_count
    
    # Average credibility
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": None, "avg_credibility": {"$avg": "$credibility_score"}}}
    ]
    agg_result = list(db.verification_history.aggregate(pipeline))
    avg_credibility = agg_result[0]['avg_credibility'] if agg_result else 0
    
    # Recent verifications (last 7 days)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent = db.verification_history.count_documents({
        "user_id": user_id,
        "created_at": {"$gte": seven_days_ago}
    })
    
    return {
        'total_verifications': total,
        'fake_count': fake_count,
        'real_count': real_count,
        'avg_credibility': round(avg_credibility, 1),
        'recent_verifications': recent
    }

def get_user_by_username(username: str) -> dict:
    """Get user by username."""
    user = db.users.find_one({"username": username})
    if user:
        user['id'] = str(user.pop('_id'))
    return user

def get_user_by_id(user_id: str) -> dict:
    """Get user by ID."""
    try:
        user = db.users.find_one({"_id": ObjectId(user_id)}, {"password": 0})
        if user:
            user['id'] = str(user.pop('_id'))
        return user
    except:
        return None

def create_user(username: str, password_hash: str, email: str = None) -> str:
    """Create a new user and return their ID."""
    user = {
        "username": username,
        "password": password_hash,
        "email": email,
        "created_at": datetime.utcnow()
    }
    result = db.users.insert_one(user)
    return str(result.inserted_id)

def save_otp(email: str, otp: str, expires_at: datetime):
    """Save an OTP for an email address."""
    db.otps.update_one(
        {"email": email},
        {"$set": {"otp": otp, "expires_at": expires_at}},
        upsert=True
    )

def verify_otp(email: str, otp: str) -> bool:
    """Verify an OTP for an email address."""
    doc = db.otps.find_one({"email": email, "otp": otp, "expires_at": {"$gt": datetime.utcnow()}})
    if doc:
        db.otps.delete_one({"_id": doc["_id"]})
        return True
    return False

def get_verification_trends(user_id: str, days: int = 30) -> list:
    """Get verification trends over time."""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    pipeline = [
        {
            "$match": {
                "user_id": user_id,
                "created_at": {"$gte": start_date}
            }
        },
        {
            "$project": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "prediction_label": 1,
                "credibility_score": 1
            }
        },
        {
            "$group": {
                "_id": "$date",
                "total": {"$sum": 1},
                "fake": {"$sum": {"$cond": [{"$eq": ["$prediction_label", "Fake"]}, 1, 0]}},
                "real": {"$sum": {"$cond": [{"$eq": ["$prediction_label", "Real"]}, 1, 0]}},
                "avg_credibility": {"$avg": "$credibility_score"}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    
    results = list(db.verification_history.aggregate(pipeline))
    trends = []
    for res in results:
        trends.append({
            "date": res["_id"],
            "total": res["total"],
            "fake": res["fake"],
            "real": res["real"],
            "avg_credibility": res["avg_credibility"]
        })
    return trends
