from PyPDF2 import PdfReader
from pathlib import Path
from langchain_core.documents import Document
import re
import os
import csv

SEP = "\n\n===PLANNED CHANGES===\n\n"
_PAGE_RE = re.compile(r"===PAGE \d+===\n?")

def _clean_excerpt(s):
    s = _PAGE_RE.sub("", s)
    s = re.sub(r"[^\S\n]+", " ", s)  # collapse horizontal whitespace, preserve newlines
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()
DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100

NPDES_DESCRIPTIONS = ["facility description", "facilities description"]
OTHER_DESCRIPTIONS = [
    "facility description",
    "facilities description",
    "facility information",
    "project description",
    "existing facility",
    "facility and discharge description",
    "wastewater system description",
    "wastewater treatment and disposal",
    "wastewater system operation summary",
]
HISTORIC_EFF = ["Historic Eff", "Historical Eff", "Effluent Lim"]
APPLICABLE = [
    "applicable plans, policies and regulations",
    "applicable plans and policies",
    "applicable plans",
    "plans, policies and regulations",
    "plans and regulations",
]
PLANNED = [". Planned Changes", ". Planned upgrades"]
RECEIVING_WATER = ["receiving water"]

SPEC = {
    "NPDES": {
        "contexts": ("attachment",),
        "strip_toc": True,
        "phrases": NPDES_DESCRIPTIONS,
        "desc_end": APPLICABLE + RECEIVING_WATER + PLANNED + ["Table F-2"] + HISTORIC_EFF,
        "changes_start": PLANNED,
        "changes_end": APPLICABLE + RECEIVING_WATER,
    },
    "NOA": {
        "contexts": ("full",),
        "strip_toc": False,
        "phrases": NPDES_DESCRIPTIONS + OTHER_DESCRIPTIONS,
        "desc_end": RECEIVING_WATER + HISTORIC_EFF + ["following table"],
    },
    "WDR": {
        "contexts": ("full",),
        "strip_toc": False,
        "phrases": NPDES_DESCRIPTIONS + OTHER_DESCRIPTIONS,
        "desc_end": RECEIVING_WATER + HISTORIC_EFF + ["following table"],
    },
}

def _phrase_re(phrases):
    return re.compile(
        "|".join(
            r"\s+".join(
                r"\s*".join(re.escape(ch) for ch in w if not ch.isspace()) for w in p.split()
            )
            for p in phrases
            if p.strip()
        ),
        re.IGNORECASE,
    )


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def _find_nth_re(text, pattern, n=1, start=0):
    m = list(pattern.finditer(text[start:]))
    return start + m[n - 1].start() if len(m) >= n else -1


def _find_nth_phrase_raw(text, phrase, n=1, start=0):
    p = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    m = list(p.finditer(text[start:]))
    return start + m[n - 1].start() if len(m) >= n else -1


def _find_first_after(text, phrases, start=0):
    candidates = [(text.lower().find(p.lower(), start), p) for p in phrases]
    candidates = [(pos, p) for pos, p in candidates if pos != -1]
    return min(candidates, key=lambda x: x[0]) if candidates else (-1, None)


def _find_attachment_f(raw):
    """Find Attachment F page in delimited full text. Returns (lookback_char_pos, page_num+1) or (None, None)."""
    page_re = re.compile(r"===PAGE (\d+)===\n")
    pages = list(page_re.finditer(raw))
    for i, m in enumerate(pages):
        pg = int(m.group(1))
        if pg < 10:
            continue
        next_start = pages[i + 1].start() if i + 1 < len(pages) else len(raw)
        page_text = raw[m.end():next_start]
        if ATTACHMENT_RE.search(page_text) and len(DOT_RE.findall(page_text)) < 3:
            lookback_idx = max(0, i - LOOKBACK_PAGES)
            return pages[lookback_idx].start(), pg + 1
    return None, None


def _strip_attachment_toc(raw):
    low = raw.lower()
    # find end of dotted TOC lines (may span many pages)
    head = low[:20000]
    hits = list(DOT_RE.finditer(head))
    if len(hits) >= 2:
        after_dots = raw[hits[-1].end():]
        # sanity check: if we land on a page that still looks like tables, extend further
        if len(DOT_RE.findall(after_dots[:500])) == 0:
            return after_dots
    # backup: find first narrative page after Attachment F header (no dots, has prose)
    j = low.find("attachment f", 1000)
    return raw[j:] if j != -1 else raw


def _extract(pdf_path, mode):
    if not os.path.exists(pdf_path):
        return None
    spec = SPEC[mode]
    desc_re = _phrase_re(spec["phrases"])

    reader = PdfReader(pdf_path)
    parts = []
    for i, page in enumerate(reader.pages):
        parts.append(f"===PAGE {i}===")
        parts.append(page.extract_text() or "")
    raw = "\n".join(parts)

    contexts = []
    if "attachment" in spec["contexts"]:
        att_pos, attachment_page = _find_attachment_f(raw)
        if att_pos is not None:
            raw_att = raw[att_pos:]
            raw_att = _strip_attachment_toc(raw_att) if spec["strip_toc"] else raw_att
            contexts.append((raw_att, normalize_text(raw_att), attachment_page))
    if "full" in spec["contexts"]:
        contexts.append((raw, normalize_text(raw), None))

    for raw, text, attachment_page in contexts:
        start = _find_nth_re(text, desc_re, 2)
        n = 2 if start != -1 else 1
        start = start if start != -1 else _find_nth_re(text, desc_re, 1)
        if start == -1:
            continue
        raw_start = _find_nth_re(raw, desc_re, n)
        if raw_start == -1:
            continue

        txt_changes, raw_changes, planned_pos = "", "", None

        # description end: first phrase found in desc_end list (or end of text)
        end_pos, end_phrase = _find_first_after(text, spec["desc_end"], start + LOOKBACK_CHARS)
        end = end_pos if end_pos != -1 else len(text)
        raw_end = (_find_nth_phrase_raw(raw, end_phrase, start=raw_start + LOOKBACK_CHARS)
                   if end_phrase else len(raw))
        if raw_end == -1:
            continue

        # planned changes: independent of desc boundary
        if "changes_start" in spec:
            pl_pos, pl_phrase = _find_first_after(text, spec["changes_start"], start + LOOKBACK_CHARS)
            if pl_pos != -1:
                raw_pl = _find_nth_phrase_raw(raw, pl_phrase, start=raw_start + LOOKBACK_CHARS)
                if raw_pl != -1:
                    planned_pos = pl_pos
                    pl_end_pos, pl_end_phrase = _find_first_after(text, spec["changes_end"], pl_pos + LOOKBACK_CHARS)
                    pl_end = pl_end_pos if pl_end_pos != -1 else len(text)
                    raw_pl_end = (_find_nth_phrase_raw(raw, pl_end_phrase, start=raw_pl + LOOKBACK_CHARS)
                                  if pl_end_phrase else len(raw))
                    txt_changes = text[pl_pos:pl_end].strip()
                    raw_changes = raw[raw_pl:raw_pl_end if raw_pl_end != -1 else len(raw)].strip()
                    if pl_pos < end:
                        end, raw_end = pl_pos, raw_pl

        txt_section = text[start:end].strip()
        raw_section = raw[raw_start:raw_end].strip()
        return {
            "txt_section": txt_section,
            "txt_changes": txt_changes,
            "full_text": text,
            "raw_txt_section": raw_section,
            "raw_txt_changes": raw_changes,
            "metadata": {
                "mode": mode,
                "attachment_f_page": attachment_page,
                "start_pos": start,
                "planned_changes_pos": planned_pos,
                "txt_section_length": len(txt_section),
                **({"txt_changes_length": len(txt_changes)} if txt_changes else {}),
            },
        }
    return None


def extract_permit_sections(pdf_path, regenerate_text_excerpts=False):
    pdf_path = Path(pdf_path)
    site_data = next((p / "site_data.csv" for p in [pdf_path.parent] + list(pdf_path.parents) if (p / "site_data.csv").exists()), pdf_path.parent.parent / "site_data.csv")
    _MODE_MAP = {
        "NPDES PERMIT": "NPDES",
        "CO-PERMITTEE": "NPDES",
        "ENROLLEE - NPDES": "NOA",
        "ENROLLEE - WDR": "NOA",
        "WDR": "WDR",
        "INDIVIDUAL MONITORING REQUIREM": "WDR",
    }
    with site_data.open("r", newline="", encoding="utf-8") as f:
        mode = next(
            (_MODE_MAP.get((row.get("Reg_Measure_Type") or "").strip().upper(), "NPDES")
             for row in csv.DictReader(f)
             if (row.get("PDF_File") or "").strip() == pdf_path.name),
            "NPDES",
        )
    cache = Path(pdf_path).parent / "text" / f"{Path(pdf_path).stem}.txt"
    if not regenerate_text_excerpts and cache.exists():
        content = cache.read_text(encoding="utf-8")
        section, changes = content.split(SEP, 1) if SEP in content else (content, "")
        return {
            "txt_section": normalize_text(section),
            "txt_changes": normalize_text(changes),
            "full_text": normalize_text(content),
            "metadata": {},
        }
    out = _extract(str(pdf_path), mode=mode)
    if regenerate_text_excerpts and out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            _clean_excerpt(out["raw_txt_section"])
            + (SEP + _clean_excerpt(out["raw_txt_changes"]) if out["raw_txt_changes"] else ""),
            encoding="utf-8",
        )
    return out
