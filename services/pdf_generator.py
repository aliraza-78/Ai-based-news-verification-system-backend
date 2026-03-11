from typing import Dict
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart


def create_color_bar(score: float, width: float = 200, height: float = 20) -> Drawing:
    """Create a colored progress bar for score visualization."""
    d = Drawing(width, height)
    
    # Background
    d.add(Rect(0, 0, width, height, fillColor=colors.lightgrey, strokeColor=None))
    
    # Determine color based on score
    if score >= 70:
        fill_color = colors.green
    elif score >= 40:
        fill_color = colors.orange
    else:
        fill_color = colors.red
    
    # Score bar
    bar_width = (score / 100) * width
    d.add(Rect(0, 0, bar_width, height, fillColor=fill_color, strokeColor=None))
    
    return d


def generate_verification_pdf(verification: Dict) -> BytesIO:
    """
    Generate a PDF report for a verification result.
    
    Args:
        verification: Full verification data including analysis
    
    Returns:
        BytesIO buffer containing the PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=20,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkblue
    )
    
    normal_style = styles['Normal']
    
    # Build content
    story = []
    
    # Header
    story.append(Paragraph("AI Fake News Analysis Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.darkblue))
    story.append(Spacer(1, 20))
    
    # Report metadata
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
    
    if verification.get('url'):
        story.append(Paragraph(f"<b>Source URL:</b> {verification['url']}", normal_style))
    
    if verification.get('title'):
        story.append(Paragraph(f"<b>Article Title:</b> {verification['title']}", normal_style))
    
    story.append(Spacer(1, 20))
    
    # Verdict Section
    story.append(Paragraph("Verification Result", heading_style))
    
    label = verification.get('prediction_label', 'Unknown')
    confidence = verification.get('confidence_score', 0)
    credibility = verification.get('credibility_score', 0)
    
    # Color for verdict
    if label.lower() == 'real':
        verdict_color = colors.green
        verdict_text = "AUTHENTIC"
    else:
        verdict_color = colors.red
        verdict_text = "POTENTIALLY MISLEADING"
    
    verdict_style = ParagraphStyle(
        'Verdict',
        parent=styles['Normal'],
        fontSize=16,
        textColor=verdict_color,
        spaceAfter=10
    )
    
    story.append(Paragraph(f"<b>Verdict: {verdict_text}</b>", verdict_style))
    story.append(Paragraph(f"Model Confidence: {confidence:.1f}%", normal_style))
    story.append(Paragraph(f"Credibility Score: {credibility:.1f}/100", normal_style))
    
    story.append(Spacer(1, 15))
    
    # Reason Summary
    analysis = verification.get('analysis', {})
    reason = analysis.get('reason_summary', '')
    if reason:
        story.append(Paragraph("Analysis Summary", heading_style))
        story.append(Paragraph(reason, normal_style))
        story.append(Spacer(1, 15))
    
    # Linguistic Analysis Section
    story.append(Paragraph("Linguistic Analysis", heading_style))
    
    linguistic_data = [
        ['Metric', 'Score', 'Interpretation'],
        ['Emotional Tone', f"{analysis.get('emotional_tone', 0):.1f}%", 
         'High = emotional, Low = neutral'],
        ['Factual Tone', f"{analysis.get('factual_tone', 0):.1f}%", 
         'High = factual, Low = opinion-based'],
        ['Neutrality', f"{analysis.get('neutrality_score', 0):.1f}%", 
         'High = balanced, Low = biased'],
        ['Sensational Language', f"{analysis.get('sensational_score', 0):.1f}%", 
         'High = sensational, Low = measured'],
        ['Clickbait Indicators', f"{analysis.get('clickbait_score', 0):.1f}%", 
         'High = clickbait-like, Low = standard'],
        ['Exaggeration', f"{analysis.get('exaggeration_score', 0):.1f}%", 
         'High = exaggerated, Low = precise'],
    ]
    
    linguistic_table = Table(linguistic_data, colWidths=[2*inch, 1*inch, 3*inch])
    linguistic_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    story.append(linguistic_table)
    story.append(Spacer(1, 15))
    
    # Source Analysis Section
    source = verification.get('source_analysis', {})
    if source.get('domain'):
        story.append(Paragraph("Source Analysis", heading_style))
        
        source_data = [
            ['Domain', source.get('domain', 'Unknown')],
            ['Source Reliability', f"{source.get('reliability_score', 50):.1f}%"],
            ['Trusted Source Similarity', f"{source.get('trusted_similarity', 50):.1f}%"],
            ['Claim Consistency', f"{source.get('claim_consistency', 50):.1f}%"],
        ]
        
        source_table = Table(source_data, colWidths=[2.5*inch, 3.5*inch])
        source_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        story.append(source_table)
        story.append(Spacer(1, 15))
    
    # Suspicious Phrases Section
    phrases = analysis.get('suspicious_phrases', [])
    if phrases:
        story.append(Paragraph("Suspicious Phrases Detected", heading_style))
        
        phrase_items = []
        for phrase in phrases[:10]:  # Limit to 10
            if isinstance(phrase, dict):
                text = phrase.get('text', '')
                category = phrase.get('category', '').replace('_', ' ').title()
                phrase_items.append(f"<b>{text}</b> - {category}")
            else:
                phrase_items.append(str(phrase))
        
        for item in phrase_items:
            story.append(Paragraph(f"• {item}", normal_style))
        
        story.append(Spacer(1, 15))
    
    # Article Text Preview
    text_content = verification.get('text_content', '')
    if text_content:
        story.append(Paragraph("Article Text (Preview)", heading_style))
        
        # Truncate long text
        preview = text_content[:1000]
        if len(text_content) > 1000:
            preview += "... [truncated]"
        
        text_style = ParagraphStyle(
            'ArticleText',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            textColor=colors.darkgrey
        )
        story.append(Paragraph(preview, text_style))
        story.append(Spacer(1, 15))
    
    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1  # Center
    )
    
    story.append(Paragraph(
        "This report is generated by AI Fake News Analysis System. "
        "Results are based on machine learning analysis and should be used as guidance only. "
        "Always verify information from multiple trusted sources.",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


def get_pdf_filename(verification: Dict) -> str:
    """Generate a filename for the PDF report."""
    verification_id = verification.get('id', 'unknown')
    label = verification.get('prediction_label', 'analysis').lower()
    date_str = datetime.now().strftime("%Y%m%d")
    
    return f"news_verification_{verification_id}_{label}_{date_str}.pdf"
