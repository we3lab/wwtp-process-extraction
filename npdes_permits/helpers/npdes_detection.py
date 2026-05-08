import re
import unicodedata
import PyPDF2
from pdfminer.high_level import extract_text


RULES = {
    "NPDES": {
        "patterns": ["Table 1. Discharger Information"],
        "detect_npdes_pattern": True,
        "max_pages": 5,
    },
    "NOA": {
        "patterns": ["notice of applicability"],
        "patterns_case_sensitive": ["NOA"],
        "max_pages": 5,
    },
    "WDR": {
        "patterns": ["waste discharge requirements"],
        "detect_npdes_pattern": True,
        "max_pages": 2,
    },
}


def normalize_text(s: str) -> str:
    """Normalize text by removing special whitespace characters and lowercasing."""
    if not s:
        return ""
    # Normalize Unicode form
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[­​‌‍﻿]", "", s)
    s = re.sub(r"[  ᠎ -   　]", "", s)
    # Lowercase everything
    return s.lower()


def extract_pdf_text(pdf_path: str, max_pages=5, lowercase=True) -> str:
    raw = ""
    with open(pdf_path, "rb") as f:
        try:
            reader = PyPDF2.PdfReader(f)
        except Exception:
            raw = extract_text(pdf_path, maxpages=max_pages) or ""
        else:
            parts = []
            for i in range(min(len(reader.pages), max_pages)):
                try:
                    t = reader.pages[i].extract_text()
                except Exception:
                    t = None
                if t:
                    parts.append(t)
            raw = " ".join(parts)

    raw = unicodedata.normalize("NFKC", raw)
    raw = re.sub(r"[­​‌‍﻿]", "", raw)
    raw = re.sub(r"[  ᠎ -   　]", "", raw)
    return raw.lower() if lowercase else raw


def detect_text_from_pdf(pdf_path: str, text_searched: str, max_pages=5):
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
    inner_sep = r"(?:[\s­​\-])*"

    def fuzzy(word: str) -> str:
        """Return a regex that matches `word` even if the extractor inserted
        whitespace, soft-hyphens, zero-width spaces or hyphens between letters.
        """
        parts = []
        for ch in word:
            # escape regex metacharacters
            parts.append(re.escape(ch) + inner_sep)
        return "".join(parts)
    # fuzzy tokens for the fixed keywords
    f_the = fuzzy("the")
    f_following = fuzzy("following")
    f_subject = fuzzy("subject")
    f_to = fuzzy("to")
    f_set = fuzzy("set")
    f_forth = fuzzy("forth")
    f_in = fuzzy("in")
    f_this = fuzzy("this")
    f_order = fuzzy("order")

    # allow up to 600 chars in captures (non-greedy), DOTALL so dot matches newlines
    pattern = re.compile(
        rf"{f_the}{f_following}(.{{1,600}}?){f_subject}{f_to}(.{{1,600}}?){f_set}{f_forth}{f_in}{f_this}(.{{1,600}}?){f_order}",
        flags=re.I | re.DOTALL,
    )
    return bool(pattern.search(txt))


def length_of_pdf(pdf_path: str) -> int:
    try:
        with open(pdf_path, "rb") as f:
            return len(PyPDF2.PdfReader(f).pages)
    except Exception:
        return 0


_CAG_PERMIT_RE = re.compile(r"\bca\s*g\d+", re.IGNORECASE)


def _rule_matches(pdf_file, rule, max_pages_default):
    mp = rule.get("max_pages", max_pages_default)
    text = extract_pdf_text(pdf_file, mp)
    text_nospace = re.sub(r"\s+", "", text)
    pattern_hit = any(
        (normalize_text(p) in text) or (re.sub(r"\s+", "", normalize_text(p)) in text_nospace)
        for p in rule.get("patterns", [])
    )
    if not pattern_hit and rule.get("patterns_case_sensitive"):
        raw = extract_pdf_text(pdf_file, mp, lowercase=False)
        pattern_hit = any(p in raw for p in rule["patterns_case_sensitive"])
    fuzzy_hit = bool(rule.get("detect_npdes_pattern")) and detect_npdes_pattern(pdf_file, mp)
    return pattern_hit or fuzzy_hit


def detect_npdes(pdf_file: str, max_pages=5, min_length=10) -> str | None:
    """Return matched doc type ("NPDES", "NOA", "WDR") or None by applying RULES."""
    if length_of_pdf(pdf_file) < min_length:
        return None

    # Check NOA first — needed to gate the CAG short-circuit
    noa_text = extract_pdf_text(pdf_file, RULES["NOA"].get("max_pages", max_pages))
    has_noa = _rule_matches(pdf_file, RULES["NOA"], max_pages)
    has_cag = bool(_CAG_PERMIT_RE.search(noa_text))

    # Generic CAG order (no NOA): not facility-specific, skip
    if has_cag and not has_noa:
        return None

    if has_noa:
        return "NOA"

    for doc_type in ("NPDES", "WDR"):
        if _rule_matches(pdf_file, RULES[doc_type], max_pages):
            return doc_type

    return None
