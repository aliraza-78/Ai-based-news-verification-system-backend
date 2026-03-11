from typing import Dict, List, Optional
from . import phrase_detector
from . import linguistic_analyzer
from . import source_analyzer
from . import credibility_scorer


def generate_prediction_reasons(
    prediction_label: str,
    confidence: float,
    linguistic: Dict,
    source: Dict,
    phrases: Dict
) -> str:
    """
    Generate human-readable reasons for the prediction.
    """
    reasons = []
    
    # Start with prediction statement
    if prediction_label.lower() == 'fake':
        reasons.append(f"This article is classified as potentially misleading with {confidence:.0f}% confidence.")
    else:
        reasons.append(f"This article appears to be authentic with {confidence:.0f}% confidence.")
    
    # Add linguistic reasons
    if linguistic.get('sensational_score', 0) > 60:
        reasons.append("It contains sensational language that may exaggerate the facts.")
    
    if linguistic.get('clickbait_score', 0) > 60:
        reasons.append("Clickbait patterns were detected in the writing style.")
    
    if linguistic.get('emotional_tone', 0) > 70:
        reasons.append("The content uses highly emotional language rather than neutral reporting.")
    
    if linguistic.get('factual_tone', 0) < 30:
        reasons.append("The article lacks factual indicators such as dates, statistics, or cited sources.")
    
    if linguistic.get('exaggeration_score', 0) > 60:
        reasons.append("Exaggerated claims using absolute language were found.")
    
    # Add source reasons
    if source.get('is_trusted'):
        reasons.append(f"The source ({source.get('domain')}) is recognized as reliable.")
    elif source.get('domain') and source.get('reliability_score', 50) < 40:
        reasons.append(f"The source ({source.get('domain')}) is not verified and shows concerning patterns.")
    
    if source.get('claim_consistency', 100) < 60:
        reasons.append("Internal inconsistencies were detected in the article's claims.")
    
    # Add phrase-based reasons
    phrase_count = phrases.get('total_count', 0)
    if phrase_count > 5:
        categories = phrases.get('category_counts', {})
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_categories:
            cat_names = [cat[0].replace('_', ' ') for cat in top_categories if cat[1] > 0]
            if cat_names:
                reasons.append(f"Multiple suspicious phrases were detected, particularly {', '.join(cat_names)}.")
    
    return ' '.join(reasons)


def analyze_text(
    text: str,
    url: Optional[str] = None,
    ml_prediction: str = None,
    ml_confidence: float = None
) -> Dict:
    """
    Perform comprehensive XAI analysis on text.
    
    This is the main entry point for the XAI engine.
    
    Args:
        text: The news article text to analyze
        url: Optional URL of the article source
        ml_prediction: Prediction from ML model ('Real' or 'Fake')
        ml_confidence: Confidence score from ML model (0-100)
    
    Returns:
        Complete analysis result with all XAI features
    """
    
    # Default ML values if not provided
    if ml_prediction is None:
        ml_prediction = 'Unknown'
    if ml_confidence is None:
        ml_confidence = 50.0
    
    # Run all analyzers
    linguistic = linguistic_analyzer.get_full_linguistic_analysis(text)
    source = source_analyzer.get_full_source_analysis(text, url)
    phrases = phrase_detector.detect_suspicious_phrases(text)
    highlighted = phrase_detector.get_highlighted_phrases(text)
    
    # Calculate credibility score
    credibility = credibility_scorer.calculate_credibility_score(
        ml_confidence=ml_confidence,
        prediction_label=ml_prediction,
        linguistic_analysis=linguistic,
        source_analysis=source
    )
    
    # Generate reason summary
    reason_summary = generate_prediction_reasons(
        prediction_label=ml_prediction,
        confidence=ml_confidence,
        linguistic=linguistic,
        source=source,
        phrases=phrases
    )
    
    # Generate credibility explanation
    credibility_explanation = credibility_scorer.get_credibility_explanation(
        credibility, linguistic, source
    )
    
    return {
        'prediction': {
            'label': ml_prediction,
            'confidence': ml_confidence
        },
        'credibility': credibility,
        'analysis': {
            'emotional_tone': linguistic['emotional_tone'],
            'factual_tone': linguistic['factual_tone'],
            'neutrality_score': linguistic['neutrality_score'],
            'sensational_score': linguistic['sensational_score'],
            'exaggeration_score': linguistic['exaggeration_score'],
            'clickbait_score': linguistic['clickbait_score'],
            'sentiment': linguistic['sentiment']
        },
        'source_analysis': {
            'domain': source['domain'],
            'reliability_score': source['reliability_score'],
            'is_trusted': source['is_trusted'],
            'trusted_similarity': source['trusted_similarity'],
            'claim_consistency': source['claim_consistency'],
            'category': source['category']
        },
        'suspicious_phrases': {
            'phrases': phrases['phrases'],
            'category_counts': phrases['category_counts'],
            'total_count': phrases['total_count']
        },
        'highlighted_text': highlighted,
        'reason_summary': reason_summary,
        'credibility_explanation': credibility_explanation,
        'details': {
            'linguistic': linguistic.get('details', {}),
            'source_issues': source.get('consistency_issues', []),
            'sources_cited': source.get('sources_cited', 0)
        }
    }


def get_analysis_summary(analysis_result: Dict) -> Dict:
    """
    Get a simplified summary of the analysis for quick display.
    """
    return {
        'label': analysis_result['prediction']['label'],
        'confidence': analysis_result['prediction']['confidence'],
        'credibility_score': analysis_result['credibility']['credibility_score'],
        'credibility_level': analysis_result['credibility']['level'],
        'reason_summary': analysis_result['reason_summary'],
        'suspicious_phrase_count': analysis_result['suspicious_phrases']['total_count'],
        'source_trusted': analysis_result['source_analysis']['is_trusted'],
        'source_domain': analysis_result['source_analysis']['domain']
    }
