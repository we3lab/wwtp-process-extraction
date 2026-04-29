from PyPDF2 import PdfReader
from pathlib import Path
from langchain_core.documents import Document
import re
import os
import csv

DESC = [
    "facility description",
    "facilities description",
    "facility information",
    "project description",
    "existing facility",
]
PLANNED = ["Planned Changes", "Planned upgrades"]
APPLICABLE = [
    "applicable plans, policies and regulations",
    "applicable plans and policies",
    "applicable plans",
    "plans, policies and regulations",
    "plans and regulations",
]
TRUNCATE_AFTER = [r"Historic\s*Ef", r"Effluent\s*Mon"]
SEP = "\n\n===PLANNED CHANGES===\n\n"
DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100


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


SPEC = {
    "npdes": {
        "contexts": ("attachment", "full"),
        "start": "regex",
        "end": "planned",
        "phrases": DESC,
        "strip_toc": True,
    },
    "noa": {
        "contexts": ("full",),
        "start": "priority",
        "end": "receiving_water",
        "phrases": DESC,
        "strip_toc": False,
    },
    "wdr": {
        "contexts": ("full",),
        "start": "priority",
        "end": "receiving_water",
        "phrases": DESC,
        "strip_toc": False,
    },
}


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").replace("\r", " ")).strip()


def _read_pages(reader, s, e):
    return "\n".join(
        (reader.pages[i].extract_text() or "") if i < len(reader.pages) else "" for i in range(s, e)
    )


def _find_nth_re(text, pattern, n=1, start=0):
    m = list(pattern.finditer(text[start:]))
    return start + m[n - 1].start() if len(m) >= n else -1


def _find_nth_phrase_raw(text, phrase, n=1, start=0):
    p = re.compile(r"\s+".join(re.escape(w) for w in phrase.split()), re.IGNORECASE)
    m = list(p.finditer(text[start:]))
    return start + m[n - 1].start() if len(m) >= n else -1


def _find_first_after(text, phrases, start=0, max_pos=None):
    for phrase in phrases:
        pos = text.lower().find(phrase.lower(), start)
        if pos != -1 and (max_pos is None or pos <= max_pos):
            return pos, phrase
    return -1, None


def find_attachment_f(pdf_path, start_page=10):
    reader = PdfReader(pdf_path)
    for pg in range(start_page, len(reader.pages)):
        text = reader.pages[pg].extract_text() or ""
        if ATTACHMENT_RE.search(text) and len(DOT_RE.findall(text)) < 3:
            return pg, None
    return None, None


def _strip_attachment_toc(raw):
    low = raw.lower()
    i = low.find("table of contents")
    if i != -1:
        m = re.search(r"attachment f.{0,200}?f-3", low[i:])
        if m:
            return raw[i + m.start() :]
        j = low.find("attachment f", i + 1)
        return raw[j:] if j != -1 else raw
    head = low[:2000]
    hits = list(DOT_RE.finditer(head))
    return raw[hits[-1].end() :] if len(hits) >= 2 else raw


def _priority_start(text, phrases, phrases_re):
    keys = [re.sub(r"\s+", " ", p.strip().lower()) for p in phrases]
    occ = {k: [] for k in keys}
    for m in phrases_re.finditer(text):
        k = re.sub(r"\s+", " ", m.group(0).strip().lower())
        if k in occ and len(occ[k]) < 2:
            occ[k].append(m.start())
    cands = [(v[1], k, 2) for k, v in occ.items() if len(v) >= 2] + [
        (v[0], k, 1) for k, v in occ.items() if len(v) == 1
    ]
    return max(cands, key=lambda x: x[0]) if cands else (None, None, None)


def _extract(pdf_path, mode):
    if not os.path.exists(pdf_path):
        return None
    spec = SPEC[mode]
    desc_re = _phrase_re(spec["phrases"])
    reader = PdfReader(pdf_path)

    contexts = []
    if "attachment" in spec["contexts"]:
        page, _ = find_attachment_f(pdf_path, start_page=10)
        if page is not None:
            raw = _read_pages(reader, max(0, page - LOOKBACK_PAGES), len(reader.pages))
            raw = _strip_attachment_toc(raw) if spec["strip_toc"] else raw
            contexts.append((raw, normalize_text(raw), page + 1))
    if "full" in spec["contexts"]:
        raw = _read_pages(reader, 0, len(reader.pages))
        contexts.append((raw, normalize_text(raw), None))

    for raw, text, attachment_page in contexts:
        if spec["start"] == "regex":
            start = _find_nth_re(text, desc_re, 2)
            n = 2 if start != -1 else 1
            start = start if start != -1 else _find_nth_re(text, desc_re, 1)
            if start == -1:
                continue
            raw_start = _find_nth_re(raw, desc_re, n)
        else:
            start, phrase, n = _priority_start(text, spec["phrases"], desc_re)
            if start is None:
                continue
            raw_start = _find_nth_phrase_raw(raw, phrase, n=n)
        if raw_start == -1:
            continue

        txt_changes, raw_changes, planned_pos, applicable_pos = "", "", None, None
        if spec["end"] == "receiving_water":
            end = text.lower().find("receiving water", start + LOOKBACK_CHARS)
            raw_end = _find_nth_phrase_raw(raw, "receiving water", start=raw_start + LOOKBACK_CHARS)
            if end == -1 or raw_end == -1:
                continue
        else:
            applicable_pos, applicable_phrase = _find_first_after(
                text, APPLICABLE, start + LOOKBACK_CHARS
            )
            if applicable_pos == -1:
                continue
            planned_pos, planned_phrase = _find_first_after(
                text, PLANNED, start + LOOKBACK_CHARS, max_pos=applicable_pos
            )
            raw_applicable = _find_nth_phrase_raw(
                raw, applicable_phrase, start=raw_start + LOOKBACK_CHARS
            )
            if raw_applicable == -1:
                continue
            if planned_pos == -1:
                end, raw_end = applicable_pos, raw_applicable
            else:
                raw_planned = _find_nth_phrase_raw(
                    raw, planned_phrase, start=raw_start + LOOKBACK_CHARS
                )
                if raw_planned == -1:
                    continue
                end, raw_end = planned_pos, raw_planned
                txt_changes = text[planned_pos:applicable_pos].strip()
                raw_changes = raw[raw_planned:raw_applicable].strip()

        txt_section = text[start:end].strip()
        cut = [
            m.start()
            for p in TRUNCATE_AFTER
            for m in [re.search(p, txt_section, re.IGNORECASE)]
            if m
        ]
        if cut:
            txt_section = txt_section[: min(cut)].strip()
        return {
            "txt_section": txt_section,
            "txt_changes": txt_changes,
            "full_text": text,
            "raw_txt_section": raw[raw_start:raw_end].strip(),
            "raw_txt_changes": raw_changes,
            "metadata": {
                "mode": mode,
                "attachment_f_page": attachment_page,
                "start_pos": start,
                "planned_changes_pos": planned_pos,
                "applicable_plans_pos": applicable_pos,
                "txt_section_length": len(txt_section),
                **({"txt_changes_length": len(txt_changes)} if txt_changes else {}),
            },
        }
    return None


def extract_permit_sections(pdf_path, regenerate_text_excerpts=False):
    pdf_path = Path(pdf_path)
    site_data = next((p / "site_data.csv" for p in [pdf_path.parent] + list(pdf_path.parents) if (p / "site_data.csv").exists()), pdf_path.parent.parent / "site_data.csv")
    with site_data.open("r", newline="", encoding="utf-8") as f:
        mode = next((row.get("Doc_Mode", "").strip().lower() for row in csv.DictReader(f) if (row.get("PDF_File") or "").strip() == pdf_path.name), "npdes")
    cache = Path(pdf_path).parent / "text" / f"{Path(pdf_path).stem}.{mode}.txt"
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
            (
                out["raw_txt_section"]
                + (SEP + out["raw_txt_changes"] if out["raw_txt_changes"] else "")
            ),
            encoding="utf-8",
        )
    return out


def load_documents(pdf_file):
    docs, pdf_path = [], Path(pdf_file)
    if not pdf_path.exists():
        print(f"PDF file not found: {pdf_file}")
        return docs
    res = extract_permit_sections(str(pdf_path))
    if res:
        content = (
            res.get("txt_section", "") + "\n\n" + res.get("txt_changes", "")
        ).strip() or res.get("full_text", "")
        meta = {
            "source": str(pdf_path),
            "page": res.get("metadata", {}).get("attachment_f_page", 0),
        }
    else:
        reader = PdfReader(str(pdf_path))
        content = normalize_text("\n".join((p.extract_text() or "") for p in reader.pages))
        meta = {"source": str(pdf_path), "page": 0}
    if content:
        docs.append(Document(page_content=content, metadata=meta))
    return docs
