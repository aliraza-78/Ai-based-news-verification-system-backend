import re
from typing import Dict, List, Tuple
from collections import Counter

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


# Download required NLTK data
def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    if NLTK_AVAILABLE:
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('corpora/stopwords', 'stopwords'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        ]
        
        for find_path, download_name in resources:
            try:
                nltk.data.find(find_path)
            except LookupError:
                try:
                    nltk.download(download_name, quiet=True)
                except Exception as e:
                    print(f"Warning: Could not download NLTK resource {download_name}: {e}")
            except Exception as e:
                # Handle potential OSErrors or other find issues
                try:
                    nltk.download(download_name, quiet=True)
                except:
                    pass


ensure_nltk_data()


def analyze_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of text using TextBlob.
    
    Returns:
        Dict with polarity (-1 to 1) and subjectivity (0 to 1)
    """
    if not TEXTBLOB_AVAILABLE:
        return {'polarity': 0, 'subjectivity': 0.5}
    
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except Exception:
        return {'polarity': 0, 'subjectivity': 0.5}


def calculate_emotional_tone(text: str) -> float:
    """
    Calculate emotional tone score (0-100).
    Higher score = more emotional language.
    """
    sentiment = analyze_sentiment(text)
    
    # Emotional tone is based on:
    # 1. Absolute polarity (strong positive or negative)
    # 2. Subjectivity (opinion vs fact)
    
    polarity_factor = abs(sentiment['polarity']) * 50
    subjectivity_factor = sentiment['subjectivity'] * 50
    
    # Check for exclamation marks and caps
    exclamation_count = text.count('!')
    caps_words = len(re.findall(r'\b[A-Z]{2,}\b', text))
    
    word_count = len(text.split())
    word_count = max(word_count, 1)
    
    punctuation_factor = min((exclamation_count / word_count) * 100, 20)
    caps_factor = min((caps_words / word_count) * 100, 10)
    
    emotional_score = polarity_factor + subjectivity_factor + punctuation_factor + caps_factor
    
    return min(emotional_score, 100)


def calculate_factual_tone(text: str) -> float:
    """
    Calculate factual tone score (0-100).
    Higher score = more factual/objective language.
    """
    sentiment = analyze_sentiment(text)
    
    # Factual content tends to be:
    # 1. Low subjectivity
    # 2. Neutral polarity
    # 3. Contains numbers, dates, proper nouns
    
    objectivity = (1 - sentiment['subjectivity']) * 40
    neutrality = (1 - abs(sentiment['polarity'])) * 30
    
    # Check for factual indicators
    numbers = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
    dates = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', text, re.IGNORECASE))
    quotes = len(re.findall(r'"[^"]+"|\'[^\']+\'', text))
    
    word_count = len(text.split())
    word_count = max(word_count, 1)
    
    factual_indicators = min((numbers + dates * 2 + quotes * 2) / word_count * 100, 30)
    
    factual_score = objectivity + neutrality + factual_indicators
    
    return min(factual_score, 100)


def calculate_neutrality_score(text: str) -> float:
    """
    Calculate neutrality score (0-100).
    Higher score = more neutral/balanced language.
    """
    sentiment = analyze_sentiment(text)
    
    # Neutrality is based on:
    # 1. Low absolute polarity
    # 2. Low subjectivity
    # 3. Balanced perspective indicators
    
    polarity_neutrality = (1 - abs(sentiment['polarity'])) * 50
    subjectivity_neutrality = (1 - sentiment['subjectivity']) * 30
    
    # Check for balanced language
    balanced_phrases = [
        'on the other hand', 'however', 'conversely', 'alternatively',
        'some argue', 'others believe', 'critics say', 'supporters claim',
        'according to', 'research suggests', 'studies indicate'
    ]
    
    text_lower = text.lower()
    balance_count = sum(1 for phrase in balanced_phrases if phrase in text_lower)
    balance_factor = min(balance_count * 5, 20)
    
    return min(polarity_neutrality + subjectivity_neutrality + balance_factor, 100)


def detect_exaggeration(text: str) -> Dict:
    """
    Detect exaggeration patterns in text.
    
    Returns:
        Dict with score (0-100) and examples found
    """
    exaggeration_patterns = [
        (r'\b(always|never|everyone|nobody|all|none)\b', 8),
        (r'\b(absolutely|completely|totally|entirely|perfectly)\b', 5),
        (r'\b(best|worst|greatest|most|least)\s+ever\b', 10),
        (r'\b(unprecedented|unbelievable|incredible|amazing)\b', 7),
        (r'\b(100%|guaranteed|proven|definitely|certainly)\b', 6),
        (r'\b(revolutionary|game-changing|world-changing)\b', 8),
        (r'!!+', 5),  # Multiple exclamation marks
    ]
    
    examples = []
    total_score = 0
    
    for pattern, weight in exaggeration_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = match[0]
            examples.append(match)
            total_score += weight
    
    word_count = len(text.split())
    word_count = max(word_count, 1)
    
    # Normalize by word count
    normalized_score = (total_score / word_count) * 50
    
    return {
        'score': min(normalized_score, 100),
        'examples': examples[:10]  # Limit examples
    }


def detect_sensational_language(text: str) -> Dict:
    """
    Detect sensational language patterns.
    
    Returns:
        Dict with score (0-100) and patterns found
    """
    sensational_patterns = [
        (r'\b(shocking|explosive|bombshell|devastating)\b', 10),
        (r'\b(breaking|urgent|alert|emergency|crisis)\b', 8),
        (r'\b(terrifying|horrifying|outrageous|insane)\b', 9),
        (r'\b(secret|hidden|exposed|revealed|uncovered)\b', 6),
        (r'\b(scandal|conspiracy|cover-up|corruption)\b', 8),
        (r'[A-Z]{3,}', 3),  # All caps words
        (r'!{2,}', 5),  # Multiple exclamation marks
    ]
    
    patterns_found = []
    total_score = 0
    
    for pattern, weight in sensational_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            patterns_found.append(match)
            total_score += weight
    
    word_count = len(text.split())
    word_count = max(word_count, 1)
    
    normalized_score = (total_score / word_count) * 50
    
    return {
        'score': min(normalized_score, 100),
        'patterns': patterns_found[:10]
    }


def detect_clickbait(text: str) -> Dict:
    """
    Detect clickbait patterns in text.
    
    Returns:
        Dict with score (0-100) and patterns found
    """
    clickbait_patterns = [
        (r"you won't believe", 15),
        (r"what happens next", 12),
        (r"this is why", 8),
        (r"here's why", 8),
        (r"the truth about", 10),
        (r"secret.{0,20}revealed", 12),
        (r"doctors hate", 15),
        (r"one weird trick", 15),
        (r"number \d+ will", 12),
        (r"wait until you see", 10),
        (r"will shock you", 12),
        (r"you need to (see|know|read)", 10),
        (r"this changes everything", 12),
        (r"\?$", 3),  # Ends with question (headlines)
    ]
    
    patterns_found = []
    total_score = 0
    text_lower = text.lower()
    
    for pattern, weight in clickbait_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            patterns_found.extend(matches if isinstance(matches[0], str) else [pattern])
            total_score += weight * len(matches)
    
    # Check title case (common in clickbait headlines)
    words = text.split()
    if len(words) > 3:
        title_case_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        if title_case_ratio > 0.8:
            total_score += 10
    
    word_count = max(len(words), 1)
    normalized_score = (total_score / word_count) * 30
    
    return {
        'score': min(normalized_score, 100),
        'patterns': patterns_found[:10]
    }


def get_full_linguistic_analysis(text: str) -> Dict:
    """
    Perform comprehensive linguistic analysis.
    
    Returns complete analysis with all scores and details.
    """
    emotional = calculate_emotional_tone(text)
    factual = calculate_factual_tone(text)
    neutrality = calculate_neutrality_score(text)
    exaggeration = detect_exaggeration(text)
    sensational = detect_sensational_language(text)
    clickbait = detect_clickbait(text)
    sentiment = analyze_sentiment(text)
    
    return {
        'emotional_tone': round(emotional, 1),
        'factual_tone': round(factual, 1),
        'neutrality_score': round(neutrality, 1),
        'exaggeration_score': round(exaggeration['score'], 1),
        'sensational_score': round(sensational['score'], 1),
        'clickbait_score': round(clickbait['score'], 1),
        'sentiment': {
            'polarity': round(sentiment['polarity'], 2),
            'subjectivity': round(sentiment['subjectivity'], 2)
        },
        'details': {
            'exaggeration_examples': exaggeration['examples'],
            'sensational_patterns': sensational['patterns'],
            'clickbait_patterns': clickbait['patterns']
        }
    }
