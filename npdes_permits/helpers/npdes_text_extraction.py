from PyPDF2 import PdfReader
from pathlib import Path
from .utils import unitprocess_keywords, extract_leaves
import re
import os
import csv

SEP = "\n\n===PLANNED CHANGES===\n\n"

def clean_excerpt(text):
    text = re.compile(r"===PAGE \d+===\n?").sub("", text)
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse horizontal whitespace, preserve newlines
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_F_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
BACK_REFERENCE_RE = re.compile(
    r"^\s*[\s,.()\-–—]*\s*(above|below|in section|as noted|as described|on official letterhead)\b",
    re.IGNORECASE,
)
ATTACHMENT_LIST_RE = re.compile(r"\(Attachment\s+\d|\bAttachment\s+\d\s*:", re.IGNORECASE)
WASTEWATER_VOCAB_RE = re.compile(
    "|".join(
        re.escape(term)
        for _, details, _ in extract_leaves(unitprocess_keywords, ignore_disposal=False)
        for term in details.get("alt_names", [])
        if term.strip()
    ),
    re.IGNORECASE,
)
VOCAB_WINDOW = 1500
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100

NPDES_PHRASES = ["facility description", "facilities description"]
NOA_WDR_PHRASES = NPDES_PHRASES + [
    "facility information",
    "project description",
    "existing facility",
    "discharge description",
    "description of discharge",
    "wastewater system description",
    "treatment technology and capacity",
    "wastewater treatment and disposal",
    "wastewater reclamation plant",
    "wastewater system operation summary",
    "wastewater treatment facility and discharge",
]
BACKUP_PHRASES = ["consists of"]
HISTORIC_EFF = ["Historic Eff", "Historical Eff", "Effluent Lim"]
APPLICABLE_PLANS = [
    "applicable plans, policies and regulations",
    "applicable plans and policies",
    "applicable plans",
    "plans, policies and regulations",
    "plans and regulations",
]
PLANNED_HEADER = [". Planned Changes", ". Planned upgrades"]
PLANNED_TEXT = ["planned changes", "planned upgrade"]
OTHER_PLANNED_END = ["receiving water", "Attachment G", "hydrogeology"]

NOA_WDR_SPEC = {
    "context": "full",
    "strip_toc": False,
    "desc_start": NOA_WDR_PHRASES,
    "desc_end": OTHER_PLANNED_END + HISTORIC_EFF + ["following table"],
    "changes_start": PLANNED_TEXT,
    "changes_end": APPLICABLE_PLANS + OTHER_PLANNED_END,
}

SPEC = {
    "NPDES": {
        "context": "attachment",
        "strip_toc": True,
        "desc_start": NPDES_PHRASES,
        "desc_end": APPLICABLE_PLANS + OTHER_PLANNED_END + PLANNED_HEADER + ["Table F-2"] + HISTORIC_EFF,
        "changes_start": PLANNED_TEXT,
        "changes_end": APPLICABLE_PLANS + OTHER_PLANNED_END,
    },
    "NOA": NOA_WDR_SPEC,
    "WDR": NOA_WDR_SPEC,
}


def phrase_pattern(phrases):
    return re.compile(
        "|".join(
            r"\s+".join(
                r"\s*".join(re.escape(ch) for ch in word if not ch.isspace())
                for word in phrase.split()
            )
            for phrase in phrases
            if phrase.strip()
        ),
        re.IGNORECASE,
    )


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def find_phrase_in_raw(raw, phrase, n=1, start=0):
    pattern = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    matches = list(pattern.finditer(raw[start:]))
    return start + matches[n - 1].start() if len(matches) >= n else -1


def first_match_after(text, phrases, start=0):
    candidates = [(text.lower().find(p.lower(), start), p) for p in phrases]
    candidates = [(pos, p) for pos, p in candidates if pos != -1]
    return min(candidates, key=lambda x: x[0]) if candidates else (-1, None)


def find_attachment_f_page(raw):
    """Find Attachment F page in delimited full text. Returns (lookback_char_pos, page_num+1) or (None, None)."""
    page_re = re.compile(r"===PAGE (\d+)===\n")
    pages = list(page_re.finditer(raw))
    for i, page_match in enumerate(pages):
        page_num = int(page_match.group(1))
        if page_num < 10:
            continue
        next_start = pages[i + 1].start() if i + 1 < len(pages) else len(raw)
        page_text = raw[page_match.end():next_start]
        if ATTACHMENT_F_RE.search(page_text) and len(DOT_RE.findall(page_text)) < 3:
            lookback_idx = max(0, i - LOOKBACK_PAGES)
            return pages[lookback_idx].start(), page_num + 1
    return None, None


def find_best_desc_match(text, pattern, start=0):
    best_pos, best_score = -1, 0
    for match in pattern.finditer(text, start):
        following = text[match.end():match.end() + 100]
        if BACK_REFERENCE_RE.search(following) or DOT_RE.search(following) or ATTACHMENT_LIST_RE.search(following):
            continue
        score = len(WASTEWATER_VOCAB_RE.findall(text[match.start():match.start() + VOCAB_WINDOW]))
        if score > best_score:
            best_score, best_pos = score, match.start()
    return best_pos


def extract_section(raw, text, raw_start, start, attachment_page, spec, mode):
    """Extract description and planned changes sections given start positions. Returns result dict or None."""
    changes_text, raw_changes_text, planned_pos = "", "", None

    # description end: first phrase found in desc_end list (or end of text)
    end_pos, end_phrase = first_match_after(text, spec["desc_end"], start + LOOKBACK_CHARS)
    end = end_pos if end_pos != -1 else len(text)
    raw_end = (find_phrase_in_raw(raw, end_phrase, start=raw_start + LOOKBACK_CHARS)
               if end_phrase else len(raw))
    if raw_end == -1:
        return None

    # planned changes: independent of desc boundary
    changes_pos, changes_phrase = first_match_after(text, spec["changes_start"], start + LOOKBACK_CHARS)
    if changes_pos != -1:
        raw_changes_start = find_phrase_in_raw(raw, changes_phrase, start=raw_start + LOOKBACK_CHARS)
        if raw_changes_start != -1:
            planned_pos = changes_pos
            changes_end_pos, changes_end_phrase = first_match_after(text, spec["changes_end"], changes_pos + LOOKBACK_CHARS)
            changes_end = changes_end_pos if changes_end_pos != -1 else len(text)
            raw_changes_end = (find_phrase_in_raw(raw, changes_end_phrase, start=raw_changes_start + LOOKBACK_CHARS)
                               if changes_end_phrase else len(raw))
            changes_text = text[changes_pos:changes_end].strip()
            raw_changes_text = raw[raw_changes_start:raw_changes_end if raw_changes_end != -1 else len(raw)].strip()
            if changes_pos < end:
                end, raw_end = changes_pos, raw_changes_start

    description = text[start:end].strip()
    raw_description = raw[raw_start:raw_end].strip()
    return {
        "txt_section": description,
        "txt_changes": changes_text,
        "full_text": text,
        "raw_txt_section": raw_description,
        "raw_txt_changes": raw_changes_text,
        "metadata": {
            "mode": mode,
            "attachment_f_page": attachment_page,
            "start_pos": start,
            "planned_changes_pos": planned_pos,
            "txt_section_length": len(description),
            **({"txt_changes_length": len(changes_text)} if changes_text else {}),
        },
    }


def extract_from_pdf(pdf_path, mode):
    if not os.path.exists(pdf_path):
        return None
    spec = SPEC[mode]
    desc_re = phrase_pattern(spec["desc_start"])

    reader = PdfReader(pdf_path)
    page_parts = []
    for i, page in enumerate(reader.pages):
        page_parts.append(f"===PAGE {i}===")
        page_parts.append(page.extract_text() or "")
    raw = "\n".join(page_parts)

    contexts = []
    if spec["context"] == "attachment":
        att_pos, attachment_page = find_attachment_f_page(raw)
        if att_pos is not None:
            raw_att = raw[att_pos:]
            if spec["strip_toc"]:
                # find end of dotted TOC lines (may span many pages)
                dot_hits = list(DOT_RE.finditer(raw_att.lower()[:20000]))
                stripped = None
                if len(dot_hits) >= 2:
                    after_toc = raw_att[dot_hits[-1].end():]
                    # sanity check: if we land on a page that still looks like tables, extend further
                    if len(DOT_RE.findall(after_toc[:500])) == 0:
                        stripped = after_toc
                if stripped is None:
                    # backup: find first narrative page after Attachment F header (no dots, has prose)
                    j = raw_att.lower().find("attachment f", 1000)
                    stripped = raw_att[j:] if j != -1 else raw_att
                raw_att = stripped
            contexts.append((raw_att, normalize_text(raw_att), attachment_page))
    if spec["context"] == "full":
        contexts.append((raw, normalize_text(raw), None))

    for raw_ctx, text, attachment_page in contexts:
        start = find_best_desc_match(text, desc_re)
        if start == -1:
            continue
        raw_start = find_best_desc_match(raw_ctx, desc_re)
        if raw_start == -1:
            continue
        result = extract_section(raw_ctx, text, raw_start, start, attachment_page, spec, mode)
        if result:
            return result

    # backup pass: "consists of" and similar, start 3 raw lines before the match
    backup_re = phrase_pattern(BACKUP_PHRASES)
    for raw_ctx, text, attachment_page in contexts:
        start = find_best_desc_match(text, backup_re)
        if start == -1:
            continue
        raw_match = find_best_desc_match(raw_ctx, backup_re)
        if raw_match == -1:
            continue
        # go back 3 newlines to include the section heading context
        i, count = raw_match - 1, 0
        while i >= 0 and count < 3:
            if raw_ctx[i] == "\n":
                count += 1
            i -= 1
        raw_start = i + 2
        result = extract_section(raw_ctx, text, raw_start, start, attachment_page, spec, mode)
        if result:
            return result

    full_text = contexts[-1][1] if contexts else ""
    return {"txt_section": "", "txt_changes": "", "full_text": full_text, "metadata": {"mode": mode}}


def extract_permit_sections(pdf_path, regenerate_text_excerpts=False):
    pdf_path = Path(pdf_path)
    site_data = next(
        (p / "site_data.csv" for p in [pdf_path.parent] + list(pdf_path.parents) if (p / "site_data.csv").exists()),
        pdf_path.parent.parent / "site_data.csv",
    )
    mode_map = {
        "NPDES PERMIT": "NPDES",
        "CO-PERMITTEE": "NPDES",
        "ENROLLEE - NPDES": "NOA",
        "ENROLLEE - WDR": "NOA",
        "WDR": "WDR",
        "INDIVIDUAL MONITORING REQUIREM": "WDR",
    }
    with site_data.open("r", newline="", encoding="utf-8") as f:
        mode = next(
            (mode_map.get((row.get("Reg_Measure_Type") or "").strip().upper(), "NPDES")
             for row in csv.DictReader(f)
             if (row.get("PDF_File") or "").strip() == pdf_path.name),
            "NPDES",
        )
    cache = Path(pdf_path).parent / "text" / f"{Path(pdf_path).stem}.txt"
    if not regenerate_text_excerpts and cache.exists():
        content = cache.read_text(encoding="utf-8")
        section, changes = content.split(SEP, 1)
        return {
            "txt_section": normalize_text(section),
            "txt_changes": normalize_text(changes),
            "full_text": normalize_text(content),
            "metadata": {},
        }
    out = extract_from_pdf(str(pdf_path), mode=mode)
    if regenerate_text_excerpts and out and out.get("raw_txt_section") is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            clean_excerpt(out["raw_txt_section"])
            + SEP + clean_excerpt(out["raw_txt_changes"]),
            encoding="utf-8",
        )
    return out
