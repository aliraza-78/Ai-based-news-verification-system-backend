# Services module
from .url_parser import extract_article_from_url, validate_url, get_url_domain
from .history_service import (
    save_full_verification,
    get_user_verifications,
    get_verification_by_id,
    delete_user_verification,
    get_user_statistics
)
from .pdf_generator import generate_verification_pdf, get_pdf_filename

__all__ = [
    'extract_article_from_url',
    'validate_url',
    'get_url_domain',
    'save_full_verification',
    'get_user_verifications',
    'get_verification_by_id',
    'delete_user_verification',
    'get_user_statistics',
    'generate_verification_pdf',
    'get_pdf_filename'
]
