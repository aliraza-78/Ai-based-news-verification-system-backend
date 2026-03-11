import os
from typing import Dict, Tuple
import numpy as np

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Model file paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'vectorizer.pkl')

# Global model instances
_model = None
_vectorizer = None
_model_loaded = False


def load_model() -> Tuple[bool, str]:
    """
    Load the ML model and vectorizer.
    
    Returns:
        Tuple of (success, message)
    """
    global _model, _vectorizer, _model_loaded
    
    if not JOBLIB_AVAILABLE:
        return False, "joblib library not available"
    
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            _model = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            _model_loaded = True
            return True, "Model loaded successfully"
        else:
            return False, "Model files not found"
    except Exception as e:
        return False, f"Failed to load model: {str(e)}"


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _model_loaded


def predict(text: str) -> Dict:
    """
    Make a prediction on the given text.
    
    Args:
        text: The news article text to analyze
    
    Returns:
        Dict with label, confidence, and probabilities
    """
    global _model, _vectorizer, _model_loaded
    
    # Try to load model if not loaded
    if not _model_loaded:
        success, message = load_model()
        if not success:
            # Return mock prediction for demo purposes
            return _mock_predict(text)
    
    try:
        # Vectorize text
        vec = _vectorizer.transform([text])
        
        # Get prediction and probabilities
        prediction = _model.predict(vec)[0]
        probabilities = _model.predict_proba(vec)[0]
        
        # Map prediction to label
        # Assuming: 0 = Real, 1 = Fake
        if prediction == 1:
            label = "Fake"
            confidence = probabilities[1] * 100
        else:
            label = "Real"
            confidence = probabilities[0] * 100
        
        return {
            'label': label,
            'confidence': round(confidence, 1),
            'probabilities': {
                'real': round(probabilities[0] * 100, 1),
                'fake': round(probabilities[1] * 100, 1)
            },
            'model_loaded': True
        }
        
    except Exception as e:
        # Fallback to mock prediction
        result = _mock_predict(text)
        result['error'] = str(e)
        return result


def _mock_predict(text: str) -> Dict:
    """
    Generate a mock prediction when model is not available.
    Uses heuristics based on text characteristics.
    """
    text_lower = text.lower()
    
    # Count suspicious indicators
    fake_indicators = 0
    
    # Sensational words
    sensational_words = [
        'shocking', 'unbelievable', 'breaking', 'urgent', 'secret',
        'revealed', 'exposed', 'conspiracy', 'miracle', 'cure'
    ]
    for word in sensational_words:
        if word in text_lower:
            fake_indicators += 1
    
    # Excessive punctuation
    if text.count('!') > 3:
        fake_indicators += 1
    if text.count('?') > 5:
        fake_indicators += 1
    
    # All caps words
    caps_words = len([w for w in text.split() if w.isupper() and len(w) > 2])
    if caps_words > 3:
        fake_indicators += 1
    
    # Calculate mock confidence
    base_fake_prob = min(fake_indicators * 0.1, 0.5) + 0.3
    
    # Add some randomness for more realistic feel
    np.random.seed(hash(text) % 2**32)
    noise = np.random.uniform(-0.1, 0.1)
    fake_prob = min(max(base_fake_prob + noise, 0.1), 0.9)
    real_prob = 1 - fake_prob
    
    if fake_prob > 0.5:
        label = "Fake"
        confidence = fake_prob * 100
    else:
        label = "Real"
        confidence = real_prob * 100
    
    return {
        'label': label,
        'confidence': round(confidence, 1),
        'probabilities': {
            'real': round(real_prob * 100, 1),
            'fake': round(fake_prob * 100, 1)
        },
        'model_loaded': False,
        'note': 'Using heuristic prediction (model not loaded)'
    }


def get_model_info() -> Dict:
    """Get information about the loaded model."""
    global _model, _model_loaded
    
    if not _model_loaded:
        return {
            'loaded': False,
            'type': None,
            'features': None
        }
    
    return {
        'loaded': True,
        'type': type(_model).__name__,
        'features': _vectorizer.max_features if hasattr(_vectorizer, 'max_features') else None
    }


# Try to load model on module import
load_model()
