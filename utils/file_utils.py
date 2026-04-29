"""
File Utilities
==============
Load and extract text from various document formats.
"""

import re
import io
from typing import Tuple, Optional
from pathlib import Path


def extract_text_from_file(file_obj, filename: str) -> Tuple[str, str]:
    """
    Extract text from uploaded file object.

    Returns:
        (text, source_name) tuple
    """
    ext = Path(filename).suffix.lower()
    source = Path(filename).stem

    if ext == ".txt":
        text = file_obj.read().decode("utf-8", errors="replace")
        return _clean_text(text), source

    elif ext == ".pdf":
        return _extract_pdf(file_obj), source

    elif ext == ".docx":
        return _extract_docx(file_obj), source

    elif ext in (".md", ".markdown"):
        text = file_obj.read().decode("utf-8", errors="replace")
        # Strip markdown formatting
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'`{1,3}.*?`{1,3}', '', text, flags=re.DOTALL)
        return _clean_text(text), source

    else:
        # Try to read as plain text
        try:
            text = file_obj.read().decode("utf-8", errors="replace")
            return _clean_text(text), source
        except Exception:
            raise ValueError(f"Unsupported file format: {ext}")


def _clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Normalize unicode
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', ' ', text)
    # Collapse multiple spaces (but not newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _extract_pdf(file_obj) -> str:
    """Extract text from PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_obj)
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        return _clean_text('\n\n'.join(pages))
    except ImportError:
        raise ImportError("pypdf required for PDF support: pip install pypdf")
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {e}")


def _extract_docx(file_obj) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(file_obj)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text('\n\n'.join(paragraphs))
    except ImportError:
        raise ImportError("python-docx required for DOCX support: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {e}")


def estimate_chunk_count(text: str, chunk_size: int = 512, overlap: int = 64) -> int:
    """Estimate number of chunks that will be generated."""
    if not text:
        return 0
    effective_chunk = chunk_size - overlap
    return max(1, len(text) // effective_chunk)
