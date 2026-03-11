from typing import Dict, Optional
import re

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


def extract_article_from_url(url: str, timeout: int = 10) -> Dict:
    """
    Extract article content from a URL using newspaper3k.
    
    Args:
        url: The URL of the article to extract
        timeout: Request timeout in seconds
    
    Returns:
        Dict with title, text, authors, publish_date, and top_image
    """
    if not NEWSPAPER_AVAILABLE:
        return {
            'success': False,
            'error': 'newspaper3k library not installed',
            'text': None,
            'title': None,
            'authors': [],
            'publish_date': None,
            'top_image': None
        }
    
    if not url:
        return {
            'success': False,
            'error': 'No URL provided',
            'text': None,
            'title': None,
            'authors': [],
            'publish_date': None,
            'top_image': None
        }
    
    # Validate URL format
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        text = article.text
        title = article.title
        authors = article.authors
        publish_date = article.publish_date
        top_image = article.top_image
        
        # Validate extracted content
        if not text or len(text.strip()) < 50:
            return {
                'success': False,
                'error': 'Could not extract meaningful content from URL',
                'text': text,
                'title': title,
                'authors': authors,
                'publish_date': str(publish_date) if publish_date else None,
                'top_image': top_image
            }
        
        return {
            'success': True,
            'error': None,
            'text': text,
            'title': title,
            'authors': authors,
            'publish_date': str(publish_date) if publish_date else None,
            'top_image': top_image
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to extract article: {str(e)}',
            'text': None,
            'title': None,
            'authors': [],
            'publish_date': None,
            'top_image': None
        }


def validate_url(url: str) -> Dict:
    """
    Validate URL format and accessibility.
    
    Returns:
        Dict with is_valid, normalized_url, and error message
    """
    if not url:
        return {
            'is_valid': False,
            'normalized_url': None,
            'error': 'No URL provided'
        }
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Basic URL pattern validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        return {
            'is_valid': False,
            'normalized_url': None,
            'error': 'Invalid URL format'
        }
    
    return {
        'is_valid': True,
        'normalized_url': url,
        'error': None
    }


def get_url_domain(url: str) -> Optional[str]:
    """Extract domain from URL."""
    if not url:
        return None
    
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        if domain.startswith('www.'):
            domain = domain[4:]
        
        return domain
    except Exception:
        return None
