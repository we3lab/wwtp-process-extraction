import os
import re
import csv
import pandas as pd
import pdfplumber
from collections import Counter, defaultdict
from PyPDF2 import PdfReader
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from helpers.utils import extract_leaves, SEP, unitprocess_keywords, package_sub_readers

DATE_FOLDER = "2026-5-15"


def clean_excerpt(text):
    text = re.compile(r"===PAGE \d+===\n?").sub("", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_F_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
WASTEWATER_VOCAB_RE = re.compile(
    "|".join(
        re.escape(term)
        for _, details, _ in extract_leaves(unitprocess_keywords, ignore_disposal=False)
        for term in details.get("alt_names", [])
        if term.strip()
    ),
    re.IGNORECASE,
)
DESC_PRIORITY_RE = re.compile(
    r"treatment process|consists of|septic|leach\s*field|comprised of|upgraded",
    re.IGNORECASE,
)
_VOCAB_COMBINED_RE = re.compile(
    WASTEWATER_VOCAB_RE.pattern + "|" + DESC_PRIORITY_RE.pattern,
    re.IGNORECASE,
)
# short header lines: start with cap/digit/letter-dot, not a page marker, ≤80 chars
RAW_HEADER_RE = re.compile(r"(?:^|\n)((?!===)[A-Za-z\d][^\n]{2,79})(?=\n)", re.MULTILINE)
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100

EFF_TERMS = [
    "historic eff",
    "historical eff",
    "effluent lim",
    "effluent mon",
    "influent mon",
    "groundwater qual",
    "groundwater mon",
    "effluent water qual",
    "parameter unit",
    "constituent unit",
    "effluent character",
    "discharge points and",
    "analytical method",
    "regulatory considerations",
]
APPLICABLE_PLANS = [
    "applicable plans",
    "plans, policies and regulations"
]
PLANNED_HEADER = [". Planned Changes", ". Planned upgrades"]
PLANNED_TEXT = ["planned changes", "planned upgrade"]
OTHER_PLANNED_END = [
    "receiving water",
    "hydrogeology",
    "site geology",
    "or anticipated noncompliance",
]

NOA_WDR_SPEC = {
    "context": "full",
    "strip_toc": False,
    "desc_end": APPLICABLE_PLANS + OTHER_PLANNED_END + EFF_TERMS + ["following table"],
    "changes_start": PLANNED_TEXT,
    "changes_end": APPLICABLE_PLANS + OTHER_PLANNED_END + EFF_TERMS,
}

SPEC = {
    "NPDES": {
        "context": "attachment",
        "strip_toc": True,
        "desc_end": APPLICABLE_PLANS + OTHER_PLANNED_END + PLANNED_HEADER + ["Table F-2"] + EFF_TERMS,
        "changes_start": PLANNED_TEXT,
        "changes_end": APPLICABLE_PLANS + OTHER_PLANNED_END + EFF_TERMS,
    },
    "NOA": NOA_WDR_SPEC,
    "WDR": NOA_WDR_SPEC,
}

CLUSTER_GAP = 500      # max chars between two vocab hits to count as clustered
LOOKBACK_HEADER = 600  # how far back from cluster start to look for a section header
MIN_CHANGES_VOCAB = 2  # min vocab hits in 1000 chars after "planned changes" phrase


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


def find_best_cluster_start(text):
    """Find the start of the best vocab cluster for a facility description.

    Prefers the earliest cluster that has a description signal (DESC_PRIORITY_RE) in
    the window leading up to and including the cluster. Falls back to densest cluster.
    """
    hits = list(_VOCAB_COMBINED_RE.finditer(text))
    if len(hits) < 2:
        return -1
    clusters = []
    current = [hits[0]]
    for h in hits[1:]:
        if h.start() - current[-1].end() <= CLUSTER_GAP:
            current.append(h)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = [h]
    if len(current) >= 2:
        clusters.append(current)
    if not clusters:
        return -1
    # prefer earliest cluster with a description signal nearby
    desc_clusters = [
        c for c in clusters
        if DESC_PRIORITY_RE.search(text[max(0, c[0].start() - LOOKBACK_HEADER):c[-1].end()])
    ]
    if desc_clusters:
        return desc_clusters[0][0].start()
    # fallback: densest cluster
    return max(clusters, key=len)[0].start()


def snap_back_to_header(text, cluster_start):
    """Slide cluster_start backward to the last section header within LOOKBACK_HEADER chars."""
    offset = max(0, cluster_start - LOOKBACK_HEADER)
    region = text[offset:cluster_start]
    headers = list(RAW_HEADER_RE.finditer(region))
    if not headers:
        return cluster_start
    return offset + headers[-1].start(1)


def find_changes_start(text, search_start, search_end, changes_re):
    """Find first 'planned changes' match with at least MIN_CHANGES_VOCAB hits nearby."""
    for match in changes_re.finditer(text, search_start, search_end):
        window = text[match.start():match.start() + 1000]
        if len(_VOCAB_COMBINED_RE.findall(window)) >= MIN_CHANGES_VOCAB:
            return match.start(), match.group(0).strip()
    return -1, None


def extract_section(text, start, attachment_page, spec, mode):
    end_pos, desc_end_phrase = first_match_after(text, spec["desc_end"], start + LOOKBACK_CHARS)
    end = end_pos if end_pos != -1 else len(text)

    changes_re = phrase_pattern(spec["changes_start"])
    changes_pos, changes_start_text = find_changes_start(text, start + LOOKBACK_CHARS, len(text), changes_re)
    if changes_pos == -1:
        # planned changes may precede the description (e.g. WDR Findings section)
        changes_pos, changes_start_text = find_changes_start(text, 0, start, changes_re)

    changes_text, planned_pos, changes_end_phrase = "", None, None
    if changes_pos != -1:
        planned_pos = changes_pos
        changes_end_pos, changes_end_phrase = first_match_after(text, spec["changes_end"], changes_pos + LOOKBACK_CHARS)
        changes_end = changes_end_pos if changes_end_pos != -1 else len(text)
        changes_text = text[changes_pos:changes_end].strip()
        if start <= changes_pos < end:
            end = changes_pos

    description = text[start:end].strip()
    return {
        "txt_section": description,
        "txt_changes": changes_text,
        "full_text": text,
        "metadata": {
            "mode": mode,
            "attachment_f_page": attachment_page,
            "start_pos": start,
            "planned_changes_pos": planned_pos,
            "txt_section_length": len(description),
            **({"txt_changes_length": len(changes_text)} if changes_text else {}),
            "desc_end_phrase": desc_end_phrase,
            "changes_start_phrase": changes_start_text,
            "changes_end_phrase": changes_end_phrase,
        },
    }


def extract_from_pdf(pdf_path, mode):
    if not os.path.exists(pdf_path):
        return None
    spec = SPEC[mode]

    _reader = PdfReader(pdf_path)
    _root = _reader.trailer['/Root'].get_object()
    is_portfolio = '/Collection' in _root
    page_parts = []
    if is_portfolio:
        for r in package_sub_readers(_reader):
            for page_num, page in enumerate(r.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    raw = "\n".join(page_parts)

    contexts = []
    if spec["context"] == "attachment":
        att_pos, attachment_page = find_attachment_f_page(raw)
        first_att = ATTACHMENT_F_RE.search(raw)
        if first_att and first_att.start() < 500:
            att_pos, attachment_page = 0, 0
        elif att_pos is None and first_att:
            att_pos, attachment_page = 0, 0
        if att_pos is not None:
            raw_att = raw[att_pos:]
            if spec["strip_toc"] and att_pos > 0:
                dot_hits = list(DOT_RE.finditer(raw_att[:20000]))
                if len(dot_hits) >= 2 and not DOT_RE.search(raw_att[dot_hits[-1].end():dot_hits[-1].end() + 500]):
                    raw_att = raw_att[dot_hits[-1].end():]
                elif (j := raw_att.lower().find("attachment f", 1000)) != -1:
                    raw_att = raw_att[j:]
            contexts.append((clean_excerpt(raw_att), attachment_page))
    if spec["context"] == "full":
        contexts.append((clean_excerpt(raw), None))

    for text, attachment_page in contexts:
        cluster_start = find_best_cluster_start(text)
        if cluster_start == -1:
            continue
        start = snap_back_to_header(text, cluster_start)
        result = extract_section(text, start, attachment_page, spec, mode)
        if result:
            return result

    full_text = contexts[-1][0] if contexts else ""
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
            "txt_section": section.strip(),
            "txt_changes": changes.strip(),
            "full_text": content,
            "metadata": {},
        }
    out = extract_from_pdf(str(pdf_path), mode=mode)
    if regenerate_text_excerpts and out and out.get("txt_section") is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            out["txt_section"] + SEP + out["txt_changes"],
            encoding="utf-8",
        )
    return out


def _extract_one(args):
    directory, pdf_file, j, total = args
    path = os.path.join(directory, pdf_file)
    return pdf_file, extract_permit_sections(path, regenerate_text_excerpts=True)


def main():
    rfr_data = f"npdes_permits/output/{DATE_FOLDER}/site_data.csv"
    directory = f"npdes_permits/output/{DATE_FOLDER}/npdes"

    site_df = pd.read_csv(rfr_data, dtype=str).fillna("")
    pdfs = site_df["PDF_File"].tolist()

    unique_pdfs = list(dict.fromkeys(p for p in pdfs if p))
    total = len(unique_pdfs)
    args = [(directory, pdf_file, j, total) for j, pdf_file in enumerate(unique_pdfs)]

    _page_marker_re = re.compile(r"===PAGE \d+===")
    flag_counts = {"unreadable": 0}
    phrase_counts = defaultdict(Counter)

    def _flag(pdf_file, reason):
        cache = Path(directory) / "text" / f"{Path(pdf_file).stem}.txt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(SEP, encoding="utf-8")
        flag_counts[reason] += 1

    with ProcessPoolExecutor(max_workers=20) as executor:
        for pdf_file, r in executor.map(_extract_one, args):
            print(f"Processed {pdf_file}")
            if r is None:
                continue
            txt = r.get("txt_section", "")
            full_text = r.get("full_text", "")
            if not txt and len(_page_marker_re.sub("", full_text).strip()) < 100:
                _flag(pdf_file, "unreadable")
            for key in ("desc_end_phrase", "changes_start_phrase", "changes_end_phrase"):
                val = (r.get("metadata") or {}).get(key)
                if val:
                    phrase_counts[key][re.sub(r"\s+", " ", val.lower().strip())] += 1

    _norm = lambda s: re.sub(r"\s+", " ", s.lower().strip())
    all_desc_end = list(dict.fromkeys(
        APPLICABLE_PLANS + OTHER_PLANNED_END + PLANNED_HEADER + ["Table F-2", "following table"] + EFF_TERMS
    ))
    ref_lists = {
        "desc_end_phrase":      ("Description end",       all_desc_end),
        "changes_start_phrase": ("Planned changes start", PLANNED_TEXT),
        "changes_end_phrase":   ("Planned changes end",   list(dict.fromkeys(APPLICABLE_PLANS + OTHER_PLANNED_END + EFF_TERMS))),
    }
    print(f"Non-machine-readable PDFs: {flag_counts['unreadable']}")
    print()
    for key, (label, ref) in ref_lists.items():
        counts = Counter({_norm(t): 0 for t in ref})
        counts.update(phrase_counts[key])
        print(f"{label}:")
        for term in sorted(ref, key=lambda t: -counts[_norm(t)]):
            print(f"  {counts[_norm(term)]:4d}  {term!r}")
        print()

    txt_dir = Path(directory) / "text"
    txt_files = list(txt_dir.glob("*.txt"))
    if txt_files:
        sizes = [(f, f.stat().st_size) for f in txt_files]
        sizes.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop text cache files in {txt_dir}:")
        for f, size in sizes[:10]:
            print(f"  {f.name}: {size} bytes")

    site_data = pd.read_csv(rfr_data, dtype=str).fillna("")
    site_data["Total_PDFs_Available"] = pd.to_numeric(
        site_data["Total_PDFs_Available"], errors="coerce"
    )
    reg_measures = site_data.groupby("Reg_Measure_ID")["Total_PDFs_Available"].max().reset_index()
    total_available = int(reg_measures["Total_PDFs_Available"].fillna(0).sum())
    print(f"Total permits: {len(reg_measures)}")
    print(f"Total PDFs available across permits: {total_available}")


if __name__ == "__main__":
    main()
