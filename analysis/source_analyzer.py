import json
import os
import re
from urllib.parse import urlparse
from typing import Dict, List, Optional

# Load trusted sources
SOURCES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'trusted_sources.json')


def load_trusted_sources() -> Dict[str, Dict]:
    """Load trusted sources from JSON file."""
    try:
        with open(SOURCES_PATH, 'r') as f:
            sources = json.load(f)
            return {s['domain']: s for s in sources}
    except FileNotFoundError:
        # Default trusted sources
        return {
            'reuters.com': {'reliability_score': 98, 'category': 'news_agency'},
            'apnews.com': {'reliability_score': 97, 'category': 'news_agency'},
            'bbc.com': {'reliability_score': 95, 'category': 'broadcast'},
            'nytimes.com': {'reliability_score': 92, 'category': 'newspaper'}
        }


TRUSTED_SOURCES = load_trusted_sources()

# Known unreliable sources (low scores)
UNRELIABLE_INDICATORS = [
    'clickbait', 'viral', 'buzz', 'trending', 'breaking',
    'news24', 'daily', 'times', 'post', 'gazette'  # Generic names that could be fake
]

# Suspicious TLDs
SUSPICIOUS_TLDS = ['.co', '.info', '.biz', '.xyz', '.click', '.top', '.win']


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    if not url:
        return None
    
    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except Exception:
        return None


def get_source_reliability(url: str) -> Dict:
    """
    Get source reliability score for a URL.
    
    Returns:
        Dict with reliability_score (0-100), category, and is_trusted flag
    """
    domain = extract_domain(url)
    
    if not domain:
        return {
            'domain': None,
            'reliability_score': 50,  # Neutral for unknown
            'category': 'unknown',
            'is_trusted': False,
            'reason': 'Could not extract domain from URL'
        }
    
    # Check if in trusted sources
    if domain in TRUSTED_SOURCES:
        source = TRUSTED_SOURCES[domain]
        return {
            'domain': domain,
            'reliability_score': source['reliability_score'],
            'category': source.get('category', 'news'),
            'is_trusted': True,
            'reason': f"Recognized trusted source ({source.get('category', 'news')})"
        }
    
    # Check for partial domain matches (subdomains of trusted sources)
    for trusted_domain, source in TRUSTED_SOURCES.items():
        if domain.endswith('.' + trusted_domain):
            return {
                'domain': domain,
                'reliability_score': source['reliability_score'] - 5,  # Slightly lower for subdomains
                'category': source.get('category', 'news'),
                'is_trusted': True,
                'reason': f"Subdomain of trusted source {trusted_domain}"
            }
    
    # Calculate score for unknown sources
    score = 50  # Start neutral
    reasons = []
    
    # Check for suspicious TLDs
    for tld in SUSPICIOUS_TLDS:
        if domain.endswith(tld):
            score -= 15
            reasons.append(f"Suspicious TLD ({tld})")
            break
    
    # Check for unreliable indicators in domain
    domain_lower = domain.lower()
    for indicator in UNRELIABLE_INDICATORS:
        if indicator in domain_lower:
            score -= 5
            reasons.append(f"Contains generic/suspicious term: {indicator}")
    
    # Check for excessive numbers in domain
    if len(re.findall(r'\d', domain)) > 2:
        score -= 10
        reasons.append("Domain contains multiple numbers")
    
    # Check for very long domains
    if len(domain) > 30:
        score -= 5
        reasons.append("Unusually long domain name")
    
    # Check for hyphens (often used in fake news sites)
    if domain.count('-') > 2:
        score -= 10
        reasons.append("Multiple hyphens in domain")
    
    # Bonus for .gov, .edu domains
    if domain.endswith('.gov') or domain.endswith('.edu'):
        score += 30
        reasons.append("Government or educational domain")
    
    # Bonus for .org (slight)
    if domain.endswith('.org'):
        score += 5
        reasons.append("Non-profit organization domain")
    
    score = max(0, min(100, score))
    
    return {
        'domain': domain,
        'reliability_score': score,
        'category': 'unknown',
        'is_trusted': score >= 70,
        'reason': '; '.join(reasons) if reasons else 'Unknown source - proceed with caution'
    }


def calculate_trusted_similarity(text: str, url: str = None) -> float:
    """
    Calculate how similar the content is to trusted news sources.
    
    This is a heuristic based on writing style and content patterns.
    Returns score from 0-100.
    """
    score = 50  # Start neutral
    
    # Check for proper attribution
    attribution_patterns = [
        r'according to [A-Z][a-z]+',
        r'said [A-Z][a-z]+ [A-Z][a-z]+',
        r'reported by',
        r'sources? (say|said|confirm)',
        r'official statement',
        r'press release'
    ]
    
    for pattern in attribution_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 5
    
    # Check for date references
    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text):
        score += 5
    
    # Check for specific numbers and statistics
    stats_count = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', text))
    score += min(stats_count * 2, 15)
    
    # Check for quotes
    quotes_count = len(re.findall(r'"[^"]{10,}"', text))
    score += min(quotes_count * 5, 15)
    
    # Penalize for all caps
    caps_ratio = len(re.findall(r'\b[A-Z]{3,}\b', text)) / max(len(text.split()), 1)
    if caps_ratio > 0.1:
        score -= 15
    
    # Penalize for excessive punctuation
    exclaim_count = text.count('!')
    if exclaim_count > 3:
        score -= min(exclaim_count * 2, 20)
    
    return max(0, min(100, score))


def analyze_claim_consistency(text: str) -> Dict:
    """
    Analyze internal consistency of claims in the article.
    
    Looks for contradictory statements or inconsistencies.
    Returns score (0-100) where higher is more consistent.
    """
    score = 80  # Assume mostly consistent by default
    issues = []
    
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # Check for contradictory patterns
    contradiction_patterns = [
        (r'\bfirst\b.*\blast\b', r'\blast\b.*\bfirst\b'),
        (r'\balways\b', r'\bnever\b'),
        (r'\beveryone\b', r'\bnobody\b'),
        (r'\ball\b', r'\bnone\b'),
        (r'\bconfirmed\b', r'\bunconfirmed\b'),
        (r'\bproven\b', r'\bunproven\b'),
    ]
    
    text_lower = text.lower()
    for pattern1, pattern2 in contradiction_patterns:
        if re.search(pattern1, text_lower) and re.search(pattern2, text_lower):
            score -= 15
            issues.append(f"Potential contradiction: contains both '{pattern1}' and '{pattern2}' type statements")
    
    # Check for hedging after definitive statements
    definitive = ['definitely', 'certainly', 'absolutely', 'proven', 'confirmed']
    hedging = ['allegedly', 'reportedly', 'possibly', 'might', 'could be']
    
    has_definitive = any(word in text_lower for word in definitive)
    has_hedging = any(word in text_lower for word in hedging)
    
    if has_definitive and has_hedging:
        score -= 10
        issues.append("Mixes definitive claims with hedging language")
    
    # Check for source consistency
    source_mentions = re.findall(r'(?:according to|said|stated by|reported by)\s+([^,\.]+)', text, re.IGNORECASE)
    if len(source_mentions) > 1:
        # Multiple sources is generally good
        score += 5
    
    return {
        'score': max(0, min(100, score)),
        'issues': issues,
        'sources_cited': len(source_mentions)
    }


def get_full_source_analysis(text: str, url: str = None) -> Dict:
    """
    Perform comprehensive source analysis.
    
    Returns complete analysis with all metrics.
    """
    reliability = get_source_reliability(url) if url else {
        'domain': None,
        'reliability_score': 50,
        'category': 'text_only',
        'is_trusted': False,
        'reason': 'No URL provided for source analysis'
    }
    
    trusted_similarity = calculate_trusted_similarity(text, url)
    claim_consistency = analyze_claim_consistency(text)
    
    return {
        'domain': reliability['domain'],
        'reliability_score': reliability['reliability_score'],
        'category': reliability['category'],
        'is_trusted': reliability['is_trusted'],
        'reliability_reason': reliability['reason'],
        'trusted_similarity': round(trusted_similarity, 1),
        'claim_consistency': round(claim_consistency['score'], 1),
        'consistency_issues': claim_consistency['issues'],
        'sources_cited': claim_consistency['sources_cited']
    }
