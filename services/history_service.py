from typing import Dict, List, Optional
import database


def save_full_verification(
    user_id: str,
    text_content: str,
    url: Optional[str],
    title: Optional[str],
    prediction_label: str,
    confidence_score: float,
    credibility_score: float,
    analysis_result: Dict
) -> str:
    """
    Save a complete verification with all analysis details.
    
    Args:
        user_id: ID of the user
        text_content: The analyzed text
        url: Optional source URL
        title: Optional article title
        prediction_label: 'Real' or 'Fake'
        confidence_score: ML model confidence (0-100)
        credibility_score: Computed credibility score (0-100)
        analysis_result: Full analysis from xai_engine
    
    Returns:
        verification_id: The ID of the saved verification
    """
    # Save main verification record
    verification_id = database.save_verification(
        user_id=user_id,
        text_content=text_content,
        url=url,
        title=title,
        prediction_label=prediction_label,
        confidence_score=confidence_score,
        credibility_score=credibility_score
    )
    
    # Extract analysis components
    analysis = analysis_result.get('analysis', {})
    source = analysis_result.get('source_analysis', {})
    phrases = analysis_result.get('suspicious_phrases', {})
    highlighted = analysis_result.get('highlighted_text', [])
    reason_summary = analysis_result.get('reason_summary', '')
    
    # Save analysis details
    database.save_analysis_details(
        verification_id=verification_id,
        suspicious_phrases=phrases.get('phrases', []),
        emotional_tone=analysis.get('emotional_tone', 0),
        factual_tone=analysis.get('factual_tone', 0),
        sensational=analysis.get('sensational_score', 0),
        exaggeration=analysis.get('exaggeration_score', 0),
        neutrality=analysis.get('neutrality_score', 0),
        clickbait=analysis.get('clickbait_score', 0),
        reason_summary=reason_summary,
        highlighted_text=highlighted
    )
    
    # Save source analysis
    database.save_source_analysis(
        verification_id=verification_id,
        domain=source.get('domain'),
        reliability=source.get('reliability_score', 50),
        trusted_similarity=source.get('trusted_similarity', 50),
        claim_consistency=source.get('claim_consistency', 50)
    )
    
    return verification_id


def get_user_verifications(
    user_id: int,
    limit: int = 20,
    offset: int = 0
) -> Dict:
    """
    Get paginated verification history for a user.
    
    Returns:
        Dict with items, total, limit, offset, and has_more
    """
    items = database.get_user_history(user_id, limit, offset)
    total = database.get_user_history_count(user_id)
    
    return {
        'items': items,
        'total': total,
        'limit': limit,
        'offset': offset,
        'has_more': offset + limit < total
    }


def get_verification_by_id(verification_id: int, user_id: int) -> Optional[Dict]:
    """
    Get full verification details by ID.
    
    Returns None if not found or user doesn't own it.
    """
    return database.get_verification_detail(verification_id, user_id)


def delete_user_verification(verification_id: int, user_id: int) -> bool:
    """
    Delete a verification record.
    
    Returns True if deleted, False if not found.
    """
    return database.delete_verification(verification_id, user_id)


def get_user_statistics(user_id: int) -> Dict:
    """
    Get comprehensive user statistics.
    """
    stats = database.get_user_stats(user_id)
    trends = database.get_verification_trends(user_id, days=30)
    
    return {
        **stats,
        'trends': trends
    }


def format_verification_for_response(verification: Dict) -> Dict:
    """
    Format a verification record for API response.
    """
    if not verification:
        return None
    
    result = {
        'id': verification['id'],
        'text_content': verification.get('text_content', ''),
        'url': verification.get('url'),
        'title': verification.get('title'),
        'prediction_label': verification['prediction_label'],
        'confidence_score': verification['confidence_score'],
        'credibility_score': verification['credibility_score'],
        'created_at': verification['created_at']
    }
    
    # Add analysis if available
    if 'analysis' in verification:
        analysis = verification['analysis']
        result['analysis'] = {
            'emotional_tone': analysis.get('emotional_tone_score', 0),
            'factual_tone': analysis.get('factual_tone_score', 0),
            'neutrality_score': analysis.get('neutrality_score', 0),
            'sensational_score': analysis.get('sensational_score', 0),
            'exaggeration_score': analysis.get('exaggeration_score', 0),
            'clickbait_score': analysis.get('clickbait_score', 0),
            'reason_summary': analysis.get('reason_summary', ''),
            'suspicious_phrases': analysis.get('suspicious_phrases', []),
            'highlighted_text': analysis.get('highlighted_text', [])
        }
    
    # Add source analysis if available
    if 'source_analysis' in verification:
        source = verification['source_analysis']
        result['source_analysis'] = {
            'domain': source.get('source_domain'),
            'reliability_score': source.get('source_reliability_score', 50),
            'trusted_similarity': source.get('trusted_source_similarity', 50),
            'claim_consistency': source.get('claim_consistency_score', 50)
        }
    
    return result
