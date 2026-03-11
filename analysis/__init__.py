# Analysis module for XAI features
from .xai_engine import analyze_text, get_analysis_summary
from .linguistic_analyzer import get_full_linguistic_analysis
from .source_analyzer import get_full_source_analysis
from .phrase_detector import detect_suspicious_phrases, get_highlighted_phrases
from .credibility_scorer import calculate_credibility_score

__all__ = [
    'analyze_text',
    'get_analysis_summary',
    'get_full_linguistic_analysis',
    'get_full_source_analysis',
    'detect_suspicious_phrases',
    'get_highlighted_phrases',
    'calculate_credibility_score'
]
