from typing import Dict


def calculate_credibility_score(
    ml_confidence: float,
    prediction_label: str,
    linguistic_analysis: Dict,
    source_analysis: Dict
) -> Dict:
    """
    Calculate composite credibility score.
    
    The credibility score represents how trustworthy the news article is.
    It combines multiple signals:
    - ML model prediction and confidence
    - Linguistic analysis (emotional tone, factual tone, neutrality)
    - Source reliability
    - Claim consistency
    
    Args:
        ml_confidence: Model confidence (0-100)
        prediction_label: 'Real' or 'Fake'
        linguistic_analysis: Dict from linguistic_analyzer
        source_analysis: Dict from source_analyzer
    
    Returns:
        Dict with credibility_score (0-100) and breakdown
    """
    
    # For "Fake" predictions, base credibility should be low
    # For "Real" predictions, base credibility should be high
    if prediction_label.lower() == 'fake':
        ml_component = 100 - ml_confidence  # Low credibility for high-confidence fake
    else:
        ml_component = ml_confidence  # High credibility for high-confidence real
    
    # Get linguistic scores
    emotional_tone = linguistic_analysis.get('emotional_tone', 50)
    factual_tone = linguistic_analysis.get('factual_tone', 50)
    neutrality = linguistic_analysis.get('neutrality_score', 50)
    sensational = linguistic_analysis.get('sensational_score', 50)
    clickbait = linguistic_analysis.get('clickbait_score', 50)
    
    # Get source scores
    source_reliability = source_analysis.get('reliability_score', 50)
    claim_consistency = source_analysis.get('claim_consistency', 50)
    trusted_similarity = source_analysis.get('trusted_similarity', 50)
    
    # Calculate linguistic quality score
    # Higher factual tone and neutrality = better
    # Lower emotional, sensational, clickbait = better
    linguistic_quality = (
        factual_tone * 0.3 +
        neutrality * 0.3 +
        (100 - emotional_tone) * 0.15 +
        (100 - sensational) * 0.15 +
        (100 - clickbait) * 0.10
    )
    
    # Calculate source quality score
    source_quality = (
        source_reliability * 0.5 +
        claim_consistency * 0.3 +
        trusted_similarity * 0.2
    )
    
    # Composite credibility score with weighted components
    credibility_score = (
        ml_component * 0.40 +           # ML model: 40%
        source_quality * 0.25 +          # Source: 25%
        linguistic_quality * 0.20 +      # Linguistic: 20%
        claim_consistency * 0.15         # Consistency: 15%
    )
    
    # Ensure score is in valid range
    credibility_score = max(0, min(100, credibility_score))
    
    # Determine credibility level
    if credibility_score >= 75:
        level = 'high'
        description = 'This article appears to be credible and trustworthy.'
    elif credibility_score >= 50:
        level = 'medium'
        description = 'This article has mixed credibility signals. Verify with additional sources.'
    elif credibility_score >= 25:
        level = 'low'
        description = 'This article shows several concerning patterns. Approach with skepticism.'
    else:
        level = 'very_low'
        description = 'This article appears unreliable. Do not trust without verification.'
    
    return {
        'credibility_score': round(credibility_score, 1),
        'level': level,
        'description': description,
        'breakdown': {
            'ml_component': round(ml_component, 1),
            'linguistic_quality': round(linguistic_quality, 1),
            'source_quality': round(source_quality, 1),
            'claim_consistency': round(claim_consistency, 1)
        },
        'weights': {
            'ml_model': 0.40,
            'source': 0.25,
            'linguistic': 0.20,
            'consistency': 0.15
        }
    }


def get_credibility_explanation(credibility_result: Dict, linguistic_analysis: Dict, source_analysis: Dict) -> str:
    """
    Generate human-readable explanation of credibility score.
    """
    score = credibility_result['credibility_score']
    breakdown = credibility_result['breakdown']
    
    explanations = []
    
    # ML component explanation
    ml_score = breakdown['ml_component']
    if ml_score >= 70:
        explanations.append("Our AI model indicates high likelihood of authenticity.")
    elif ml_score <= 30:
        explanations.append("Our AI model detected patterns commonly found in misleading content.")
    
    # Linguistic explanation
    ling_score = breakdown['linguistic_quality']
    if ling_score >= 70:
        explanations.append("The writing style is factual and balanced.")
    elif ling_score <= 40:
        reasons = []
        if linguistic_analysis.get('sensational_score', 0) > 60:
            reasons.append("sensational language")
        if linguistic_analysis.get('emotional_tone', 0) > 60:
            reasons.append("emotional tone")
        if linguistic_analysis.get('clickbait_score', 0) > 60:
            reasons.append("clickbait patterns")
        if reasons:
            explanations.append(f"Concerning linguistic patterns detected: {', '.join(reasons)}.")
    
    # Source explanation
    source_score = breakdown['source_quality']
    domain = source_analysis.get('domain')
    if source_analysis.get('is_trusted'):
        explanations.append(f"The source ({domain}) is recognized as reliable.")
    elif domain and source_score <= 40:
        explanations.append(f"The source ({domain}) is not in our trusted sources database.")
    
    # Consistency explanation
    if breakdown['claim_consistency'] <= 50:
        explanations.append("Some inconsistencies were detected in the article's claims.")
    
    if not explanations:
        explanations.append(credibility_result['description'])
    
    return ' '.join(explanations)
