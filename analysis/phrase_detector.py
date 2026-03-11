import re
import json
import os
from typing import List, Dict, Tuple

# Load suspicious keywords
KEYWORDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'suspicious_keywords.json')

def load_keywords() -> Dict[str, List[str]]:
    """Load suspicious keywords from JSON file."""
    try:
        with open(KEYWORDS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'sensational': ['shocking', 'unbelievable', 'breaking'],
            'clickbait': ["you won't believe", 'secret revealed'],
            'exaggeration': ['always', 'never', 'everyone', 'nobody'],
            'emotional': ['outrageous', 'disgusting', 'horrifying'],
            'uncertainty_hedges': ['allegedly', 'reportedly'],
            'bias_indicators': ['fake news', 'mainstream media'],
            'scientific_misuse': ['studies show', 'miracle cure']
        }


SUSPICIOUS_KEYWORDS = load_keywords()


def find_phrase_positions(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Find all positions of a phrase in text (case-insensitive)."""
    positions = []
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    for match in pattern.finditer(text):
        positions.append((match.start(), match.end()))
    return positions


def detect_suspicious_phrases(text: str) -> Dict:
    """
    Detect suspicious phrases in text.
    
    Returns:
        Dict with:
        - phrases: List of detected phrases with category and positions
        - category_counts: Count of phrases per category
        - total_count: Total suspicious phrases found
    """
    text_lower = text.lower()
    detected = []
    category_counts = {}
    
    for category, phrases in SUSPICIOUS_KEYWORDS.items():
        category_counts[category] = 0
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in text_lower:
                positions = find_phrase_positions(text, phrase)
                for start, end in positions:
                    detected.append({
                        'text': text[start:end],
                        'start': start,
                        'end': end,
                        'category': category,
                        'reason': get_category_reason(category)
                    })
                    category_counts[category] += 1
    
    # Remove duplicates (overlapping matches)
    detected = remove_overlapping(detected)
    
    return {
        'phrases': detected,
        'category_counts': category_counts,
        'total_count': len(detected)
    }


def remove_overlapping(phrases: List[Dict]) -> List[Dict]:
    """Remove overlapping phrase detections, keeping the longer match."""
    if not phrases:
        return []
    
    # Sort by start position, then by length (longer first)
    sorted_phrases = sorted(phrases, key=lambda x: (x['start'], -(x['end'] - x['start'])))
    
    result = []
    last_end = -1
    
    for phrase in sorted_phrases:
        if phrase['start'] >= last_end:
            result.append(phrase)
            last_end = phrase['end']
    
    return result


def get_category_reason(category: str) -> str:
    """Get human-readable reason for a category."""
    reasons = {
        'sensational': 'Sensational language that may exaggerate importance',
        'clickbait': 'Clickbait phrase designed to attract attention',
        'exaggeration': 'Absolute or exaggerated language lacking nuance',
        'emotional': 'Emotionally charged language that may bias interpretation',
        'uncertainty_hedges': 'Unverified or unconfirmed claim',
        'bias_indicators': 'Politically biased or inflammatory language',
        'scientific_misuse': 'Potentially misleading scientific claim'
    }
    return reasons.get(category, 'Suspicious phrase detected')


def calculate_suspicion_score(detection_result: Dict, text_length: int) -> float:
    """
    Calculate overall suspicion score based on detected phrases.
    
    Score ranges from 0 (no suspicion) to 100 (highly suspicious).
    """
    if text_length == 0:
        return 0
    
    # Weights for different categories
    weights = {
        'sensational': 8,
        'clickbait': 10,
        'exaggeration': 5,
        'emotional': 6,
        'uncertainty_hedges': 4,
        'bias_indicators': 9,
        'scientific_misuse': 8
    }
    
    total_weight = 0
    category_counts = detection_result['category_counts']
    
    for category, count in category_counts.items():
        weight = weights.get(category, 5)
        total_weight += count * weight
    
    # Normalize by text length (per 100 words approximately)
    word_count = len(text.split()) if 'text' in dir() else text_length // 5
    word_count = max(word_count, 1)
    
    # Calculate normalized score
    normalized = (total_weight / word_count) * 20
    
    # Cap at 100
    return min(normalized * 10, 100)


def get_highlighted_phrases(text: str) -> List[Dict]:
    """
    Get phrases to highlight with their severity levels.
    
    Returns list of dicts with: text, start, end, reason, severity
    """
    result = detect_suspicious_phrases(text)
    
    severity_map = {
        'clickbait': 'high',
        'bias_indicators': 'high',
        'sensational': 'medium',
        'scientific_misuse': 'medium',
        'emotional': 'medium',
        'exaggeration': 'low',
        'uncertainty_hedges': 'low'
    }
    
    highlighted = []
    for phrase in result['phrases']:
        highlighted.append({
            'text': phrase['text'],
            'start': phrase['start'],
            'end': phrase['end'],
            'reason': phrase['reason'],
            'category': phrase['category'],
            'severity': severity_map.get(phrase['category'], 'low')
        })
    
    return highlighted
