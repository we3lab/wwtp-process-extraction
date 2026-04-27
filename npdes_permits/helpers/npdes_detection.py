import os
import re
import unicodedata
from datetime import datetime
from itertools import zip_longest
import PyPDF2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def normalize_text(s: str) -> str:
    """Normalize text by removing special whitespace characters and lowercasing."""
    if not s:
        return ''
    # Normalize Unicode form
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'[\u00AD\u200B\u200C\u200D\uFEFF]', '', s)
    s = re.sub(r'[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]', '', s)
    #s = re.sub(r'\s+', '', s)
    
    # Lowercase everything
    return s.lower()

def extract_pdf_text(pdf_path: str, max_pages=5) -> str:
    page_texts = []
    try:
        with open(pdf_path, 'rb') as f:
            try:
                reader = PyPDF2.PdfReader(f)
            except Exception:
                # try pdfminer fallback
                try:
                    from pdfminer.high_level import extract_text
                    txt = extract_text(pdf_path, maxpages=max_pages)
                    return normalize_text(txt) if txt else ''
                except Exception:
                    return ''
            num_pages = min(len(reader.pages), max_pages)
            for i in range(num_pages):
                try:
                    page = reader.pages[i]
                    text = page.extract_text() or ''
                except Exception:
                    text = ''
                if text:
                    page_texts.append(text)
    except Exception:
        return ''
    return normalize_text(' '.join(page_texts))

def detect_text_from_pdf(pdf_path : str, text_searched : str, max_pages=5):
    """Detect if 'text_searched' is in the first 'max_pages' of the PDF at 'pdf_path'."""
    # Use the combined normalized text from the first pages for more robust matching
    combined = extract_pdf_text(pdf_path, max_pages)
    if not combined:
        return False
    text_searched_normalized = normalize_text(text_searched)
    if text_searched_normalized in combined:
        return True
    # try spaceless fallback
    combined_nospace = re.sub(r"\s+", "", combined)
    k_nospace = re.sub(r"\s+", "", text_searched_normalized)
    if k_nospace and k_nospace in combined_nospace:
        return True
    return False

def detect_npdes_pattern(pdf_path: str, max_pages=5) -> bool:
    """Detect flexible NPDES-like sentences in a PDF.
    Matches patterns like:
      "the following <...> subject to <...> set forth in this <...> order"
    """
    txt = extract_pdf_text(pdf_path, max_pages)
    if not txt:
        return False

    # tolerant regex: allow spaces, lines changes and some special chars between letters
    inner_sep = r"(?:[\s\u00AD\u200B\-])*"

    def fuzzy(word: str) -> str:
        """Return a regex that matches `word` even if the extractor inserted
        whitespace, soft-hyphens, zero-width spaces or hyphens between letters.
        """
        parts = []
        for ch in word:
            # escape regex metacharacters
            parts.append(re.escape(ch) + inner_sep)
        return ''.join(parts)

    # fuzzy tokens for the fixed keywords
    f_the = fuzzy('the')
    f_following = fuzzy('following')
    f_subject = fuzzy('subject')
    f_to = fuzzy('to')
    f_set = fuzzy('set')
    f_forth = fuzzy('forth')
    f_in = fuzzy('in')
    f_this = fuzzy('this')
    f_order = fuzzy('order')

    # allow up to 600 chars in captures (non-greedy), DOTALL so dot matches newlines
    pattern = re.compile(
        rf"{f_the}{f_following}(.{{1,600}}?){f_subject}{f_to}(.{{1,600}}?){f_set}{f_forth}{f_in}{f_this}(.{{1,600}}?){f_order}",
        flags=re.I | re.DOTALL,
    )

    if pattern.search(txt):
        return True
    return False

def length_of_pdf(pdf_path: str) -> int:
    """Return the number of pages in the PDF at 'pdf_path'."""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return len(reader.pages)

_CAG_PERMIT_RE = re.compile(r'\bca\s*g\d+', re.IGNORECASE)

def detect_npdes(pdf_file: str, max_pages=5, min_length=10) -> bool:
    """Detect NPDES-like patterns in the PDF at 'pdf_file'."""
    if length_of_pdf(pdf_file) < min_length:
        return False
    early_text = extract_pdf_text(pdf_file, max_pages)
    has_noa = bool(re.search(r'notice\s+of\s+applicability', early_text))
    # General order documents contain a CAG permit number but are not NOAs — they cover
    # all enrollees and are not facility-specific, so reject them.
    if _CAG_PERMIT_RE.search(early_text) and not has_noa:
        return False
    # detect patterns in pdfs :
    # 1. "Table 1. Discharger Information" (for some pdfs without the full sentence)
    # 2. flexible NPDES-like sentence pattern ("the following <...> subject to <...> set forth in this <...> order")
    has_pattern = detect_text_from_pdf(pdf_file, "Table 1. Discharger Information", max_pages)
    has_pattern2 = detect_npdes_pattern(pdf_file, max_pages)
    return has_pattern or has_pattern2 or has_noa



