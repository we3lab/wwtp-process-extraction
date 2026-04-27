from PyPDF2 import PdfReader
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
import re
import os


def load_documents(pdf_file):
    """Load PDFs from a directory and return a list of Documents.

    For each PDF in `pdf_directory`, attempt to extract the relevant
    permit sections (Facility Description and Planned Changes) using
    `extract_permit_sections`. If extraction succeeds, create a
    `Document` with the extracted text (preferring `txt_section` and
    `txt_changes`) and metadata including the source path and the
    Attachment F start page.
    """
    document = []
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_file}")
        return document

    res = extract_permit_sections(str(pdf_path))
    if not res:
        # fallback: try to load full PDF text
        reader = PdfReader(str(pdf_path))
        full_text = ""
        for p in reader.pages:
            try:
                full_text += (p.extract_text() or "") + "\n"
            except Exception:
                continue
        page_content = normalize_text(full_text)
        metadata = {"source": str(pdf_path), "page": 0}
    else:
        # prefer the focused section + planned changes
        page_content = (res.get("txt_section", "") + "\n\n" + res.get("txt_changes", "")).strip()
        if not page_content:
            page_content = res.get("full_text", "")
        metadata = {
            "source": str(pdf_path),
            "page": res.get("metadata", {}).get("attachment_f_page", 0),
        }

    if page_content:
        document.append(Document(page_content=page_content, metadata=metadata))
    return document


def normalize_text(text):
    """Normalize text by replacing line breaks and multiple spaces"""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    return text.strip()


def _truncate_after_historic_ef(text):
    """If 'Historic Ef' (possibly broken by whitespace) appears, truncate text before it."""
    if not text:
        return text
    m = re.search(r"Historic\s*Ef", text, re.IGNORECASE)
    if m:
        return text[: m.start()].strip()
    return text


def find_attachment_f(pdf_path, start_page=10):
    """
    Find the page containing 'ATTACHMENT F - FACT SHEET' after start_page.
    Handles different hyphen types (-, –, —, ‐).

    Returns: (page_number, position_in_full_text) or (None, None)
    """
    reader = PdfReader(pdf_path)

    # Pattern that matches various hyphen types
    pattern = r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET"

    # TOC pages (either main document or Attachment F internal TOC) are full of dot-leader
    # sequences ("............"). Content pages have prose with few or no dot leaders.
    dot_leaders = re.compile(r"\.{5,}")

    for pg in range(start_page, len(reader.pages)):
        try:
            text = reader.pages[pg].extract_text() or ""
            if not re.search(pattern, text, re.IGNORECASE):
                continue
            if len(dot_leaders.findall(text)) >= 3:
                continue  # looks like a TOC page, skip
            return pg, None
        except Exception as e:
            print(f"Error reading page {pg}: {e}")
            continue

    return None, None


def find_nth_occurrence(text, search_text, n=1, start_pos=0):
    pattern = re.escape(search_text)
    matches = list(re.finditer(pattern, text[start_pos:], re.IGNORECASE))

    if len(matches) >= n:
        return start_pos + matches[n - 1].start()

    return -1


_CHANGES_SEP = "\n\n===PLANNED CHANGES===\n\n"

# Flexible patterns: allow optional spaces within "Description" and match
# both "Facility Description" and "Facilities Description" (some PDFs use plural).
_FAC_DESC_RE = re.compile(
    r"F\s*a\s*c\s*i\s*l\s*i\s*t\s*(?:y|i\s*e\s*s)\s*"
    r"D\s*e\s*s\s*c\s*r\s*i\s*p\s*t\s*i\s*o\s*n",
    re.IGNORECASE,
)

# Centralized phrase priority lists used throughout the extractor
DESCRIPTION_PRIORITIES = [
    "facility description",
    "facilities description",
    "facility information",
    "project description",
]

_NOA_DESC_RE = re.compile(
    "|".join(re.escape(p) for p in DESCRIPTION_PRIORITIES), re.IGNORECASE
)

APPLICABLE_PLANS_ALTS = [
    "applicable plans, policies and regulations",
    "applicable plans and policies",
    "applicable plans",
    "plans, policies and regulations",
    "plans and regulations",
]

PLANNED_CHANGES_ALTS = ["Planned Changes", "Planned upgrades"]


def _txt_cache_path(pdf_path):
    p = Path(pdf_path)
    return p.parent / "text" / (p.stem + ".txt")


def _find_nth_re(text, pattern, n=1, start_pos=0):
    """Find the nth match of a compiled regex in text[start_pos:]."""
    matches = list(pattern.finditer(text[start_pos:]))
    if len(matches) >= n:
        return start_pos + matches[n - 1].start()
    return -1


def _find_nth_raw(text, phrase, n=1, start_pos=0):
    """Find nth occurrence of phrase in raw text, matching any whitespace between words.

    Handles PDF extraction artifacts like double spaces or newlines inside phrases.
    """
    pattern = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    matches = list(pattern.finditer(text[start_pos:]))
    if len(matches) >= n:
        return start_pos + matches[n - 1].start()
    return -1


def _find_section_start(full_text):
    """Return (pos, pattern, n) for the section-start anchor, or (-1, None, None)."""
    second_fac = _find_nth_re(full_text, _FAC_DESC_RE, n=2)
    if second_fac != -1:
        return second_fac, _FAC_DESC_RE, 2
    first_fac = _find_nth_re(full_text, _FAC_DESC_RE, n=1)
    if first_fac != -1:
        return first_fac, _FAC_DESC_RE, 1
    return -1, None, None


def _strip_attachment_toc(raw_text, full_text):
    """Trim TOC-like leading content from Attachment F text."""
    lower = raw_text.lower()
    toc_idx = lower.find("table of contents")
    leader_re = re.compile(r"\.{5,}")

    cut_at = -1
    if toc_idx != -1:
        # Prefer the first Attachment F heading that looks like content (F-3+).
        m = re.search(r"attachment f.{0,200}?f-3", lower[toc_idx:])
        if m:
            cut_at = toc_idx + m.start()
        else:
            next_att = lower.find("attachment f", toc_idx + 1)
            if next_att != -1:
                cut_at = next_att
    else:
        head = lower[:2000]
        matches = list(leader_re.finditer(head))
        if len(matches) >= 2:
            cut_at = matches[-1].end()

    if cut_at != -1:
        return raw_text[cut_at:], normalize_text(raw_text[cut_at:])
    return raw_text, full_text


def _find_noa_anchor(full_text):
    """Return (start, phrase, occurrence_n) for NOA description anchor or (None, None, None)."""
    occurrences = {p: [] for p in DESCRIPTION_PRIORITIES}
    for m in _NOA_DESC_RE.finditer(full_text):
        phrase = m.group(0).lower()
        if phrase in occurrences and len(occurrences[phrase]) < 2:
            occurrences[phrase].append(m.start())

    candidates = []
    for phrase, positions in occurrences.items():
        if len(positions) >= 2:
            candidates.append((positions[1], phrase, 2))
        elif len(positions) == 1:
            candidates.append((positions[0], phrase, 1))

    if not candidates:
        return None, None, None
    start, phrase, occ_n = max(candidates, key=lambda x: x[0])
    return start, phrase, occ_n


def _extract_from_pdf(pdf_path):
    """Parse permit PDF and return sections dict, or None on failure."""
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return None

    try:
        reader = PdfReader(pdf_path)
        # Step 1: Find ATTACHMENT F - FACT SHEET after page 10
        attachment_page, _ = find_attachment_f(pdf_path, start_page=10)

        if attachment_page is None:
            # NOA fallback: "Facility Description" or "Facility Information" → "Receiving Water"
            raw_full_text = "\n".join(
                reader.pages[i].extract_text() or "" for i in range(len(reader.pages))
            )
            full_text = normalize_text(raw_full_text)

            start, noa_phrase, fac_n = _find_noa_anchor(full_text)
            if start is None:
                return None

            end = find_nth_occurrence(full_text, "receiving water", n=1, start_pos=start + 100)
            end = end if end != -1 else start + 10000

            raw_start = _find_nth_raw(raw_full_text, noa_phrase, n=fac_n)
            if raw_start == -1:
                raw_start = 0
            raw_end = _find_nth_raw(
                raw_full_text, "receiving water", n=1, start_pos=raw_start + 100
            )
            if raw_end == -1:
                raw_end = raw_start + 10000

            txt_section = full_text[start:end].strip()
            txt_section = _truncate_after_historic_ef(txt_section)
            return {
                "txt_section": txt_section,
                "txt_changes": "",
                "full_text": full_text,
                "raw_txt_section": raw_full_text[raw_start:raw_end].strip(),
                "raw_txt_changes": "",
                "metadata": {"attachment_f_page": None},
            }

        # Step 2: Extract text from Attachment F onwards (include a small lookback
        # to catch headings that appear on the page before the Attachment F body).
        raw_start_page = max(0, attachment_page - 2)

        def _read_pages(end_page):
            raw = []
            for i in range(raw_start_page, end_page):
                try:
                    raw.append(reader.pages[i].extract_text() or "")
                except Exception:
                    raw.append("")
            raw_text = "\n".join(raw)
            return raw_text, normalize_text(raw_text)

        raw_full_text, full_text = _read_pages(len(reader.pages))
        if not full_text:
            return None

        raw_full_text, full_text = _strip_attachment_toc(raw_full_text, full_text)

        # Step 3: Find section start (pattern + occurrence tracked for raw lookup)
        fac_desc_start, fac_pattern, fac_n = _find_section_start(full_text)
        if fac_desc_start == -1:
            # If TOC was stripped and no anchor found, expand the window forward.
            end_page = min(len(reader.pages), attachment_page + 30)
            raw_full_text, full_text = _read_pages(end_page)
            raw_full_text, full_text = _strip_attachment_toc(raw_full_text, full_text)
            fac_desc_start, fac_pattern, fac_n = _find_section_start(full_text)
        if fac_desc_start == -1:
            return None

        raw_fac_start = _find_nth_re(raw_full_text, fac_pattern, n=fac_n)
        if raw_fac_start == -1:
            raw_fac_start = 0

        # If the match is in a TOC-like area (dot leaders), skip to the next occurrence.
        toc_leaders = re.compile(r"\.{5,}")
        while True:
            toc_window = raw_full_text[raw_fac_start : raw_fac_start + 800].lower()
            is_toc = "table of contents" in toc_window or len(toc_leaders.findall(toc_window)) >= 2
            if not is_toc:
                break
            fac_n += 1
            next_pos = _find_nth_re(full_text, fac_pattern, n=fac_n)
            if next_pos == -1:
                break
            fac_desc_start = next_pos
            raw_fac_start = _find_nth_re(raw_full_text, fac_pattern, n=fac_n)
            if raw_fac_start == -1:
                raw_fac_start = 0
                break

        search_start = fac_desc_start + 800
        raw_search_start = max(0, raw_fac_start + 800)

        # Step 4: Find applicable plans boundary (track winning phrase for raw lookup)
        applicable_plans_pos = -1
        applicable_phrase = None
        for alt in APPLICABLE_PLANS_ALTS:
            pos = find_nth_occurrence(full_text, alt, n=1, start_pos=search_start)
            if pos != -1:
                applicable_plans_pos = pos
                applicable_phrase = alt
                break

        raw_applicable = -1
        if applicable_phrase:
            raw_applicable = _find_nth_raw(
                raw_full_text, applicable_phrase, n=1, start_pos=raw_search_start
            )

        # Step 5: Find Planned Changes boundary (track winning phrase for raw lookup)
        planned_changes_pos = -1
        planned_phrase = None

        if applicable_plans_pos != -1:
            # look for planned-changes variants that occur before applicable_plans_pos
            for alt in PLANNED_CHANGES_ALTS:
                pos_alt = find_nth_occurrence(full_text, alt, n=1, start_pos=search_start)
                if pos_alt != -1 and pos_alt <= applicable_plans_pos:
                    planned_changes_pos = pos_alt
                    planned_phrase = alt
                    break
        else:
            for alt in PLANNED_CHANGES_ALTS:
                pos_alt = find_nth_occurrence(full_text, alt, n=1, start_pos=search_start)
                if pos_alt != -1:
                    planned_changes_pos = pos_alt
                    planned_phrase = alt
                    break

        raw_planned = -1
        if planned_phrase:
            raw_planned = _find_nth_raw(
                raw_full_text, planned_phrase, n=1, start_pos=raw_search_start
            )

        # Step 6: Build outputs
        if planned_changes_pos == -1:
            if applicable_plans_pos != -1:
                txt_section = full_text[fac_desc_start:applicable_plans_pos].strip()
                raw_end = raw_applicable if raw_applicable != -1 else len(raw_full_text)
            else:
                txt_section = full_text[fac_desc_start : fac_desc_start + 10000].strip()
                raw_end = raw_fac_start + 10000
            txt_section = _truncate_after_historic_ef(txt_section)
            raw_txt_section = raw_full_text[raw_fac_start:raw_end].strip()
            return {
                "txt_section": txt_section,
                "txt_changes": "",
                "full_text": full_text,
                "raw_txt_section": raw_txt_section,
                "raw_txt_changes": "",
                "metadata": {
                    "attachment_f_page": attachment_page + 1,
                    "facility_desc_pos": fac_desc_start,
                    "planned_changes_pos": None,
                    "applicable_plans_pos": applicable_plans_pos,
                },
            }
        else:
            if applicable_plans_pos == -1:
                applicable_plans_pos = min(planned_changes_pos + 5000, len(full_text))
                raw_applicable = (
                    min(raw_planned + 5000, len(raw_full_text))
                    if raw_planned != -1
                    else raw_fac_start + 10000
                )

            txt_section = full_text[fac_desc_start:planned_changes_pos].strip()
            txt_section = _truncate_after_historic_ef(txt_section)
            txt_changes = full_text[planned_changes_pos:applicable_plans_pos].strip()
            raw_txt_section = raw_full_text[
                raw_fac_start : raw_planned if raw_planned != -1 else len(raw_full_text)
            ].strip()
            raw_txt_changes = (
                raw_full_text[raw_planned:raw_applicable].strip() if raw_planned != -1 else ""
            )
            return {
                "txt_section": txt_section,
                "txt_changes": txt_changes,
                "full_text": full_text,
                "raw_txt_section": raw_txt_section,
                "raw_txt_changes": raw_txt_changes,
                "metadata": {
                    "attachment_f_page": attachment_page + 1,
                    "facility_desc_pos": fac_desc_start,
                    "planned_changes_pos": planned_changes_pos,
                    "applicable_plans_pos": applicable_plans_pos,
                    "txt_section_length": len(txt_section),
                    "txt_changes_length": len(txt_changes),
                },
            }

    except Exception:
        return None


def extract_permit_sections(pdf_path, regenerate_text_excerpts=False):
    """
    Extract Facility Description and Planned Changes sections from a permit PDF.

    When regenerate_text_excerpts=True, parses the PDF and saves raw (PDF line
    breaks preserved) text to {pdf_dir}/text/{stem}.txt for inspection and re-use.
    When regenerate_text_excerpts=False (default), reads from that .txt cache if
    it exists; falls back to PDF parsing if it doesn't.

    Returns: dict with keys txt_section, txt_changes, full_text, metadata — or None.
    All text values are normalized (line breaks removed) for searching.
    """
    cache = _txt_cache_path(pdf_path)

    if not regenerate_text_excerpts and cache.exists():
        content = cache.read_text(encoding="utf-8")
        if _CHANGES_SEP in content:
            txt_section, txt_changes = content.split(_CHANGES_SEP, 1)
        else:
            txt_section, txt_changes = content, ""
        norm_section = normalize_text(txt_section)
        norm_section = _truncate_after_historic_ef(norm_section)
        return {
            "txt_section": norm_section,
            "txt_changes": normalize_text(txt_changes),
            "full_text": normalize_text(content),
            "metadata": {},
        }

    result = _extract_from_pdf(pdf_path)

    if regenerate_text_excerpts and result is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        content = result.get("raw_txt_section") or result.get("txt_section", "")
        raw_changes = result.get("raw_txt_changes") or result.get("txt_changes", "")
        if raw_changes:
            content += _CHANGES_SEP + raw_changes
        cache.write_text(content, encoding="utf-8")

    return result
