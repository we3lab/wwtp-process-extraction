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


def clean_excerpt(text):
    # keep page boundaries as [Page N] so source pages are traceable in excerpts
    text = re.compile(r"===PAGE (\d+)===\n?").sub(r"[Page \1]\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()

DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_F_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")
WASTEWATER_VOCAB_RE = re.compile(
    "|".join(
        re.escape(term)
        for _, details, _ in extract_leaves(unitprocess_keywords)
        for term in details.get("alt_names", [])
        if term.strip()
    ),
    re.IGNORECASE,
)
DESC_PRIORITY_RE = re.compile(
    r"treatment process|consists of|leach\s*field|comprised of|upgraded"
    r"|headworks|preliminary treatment|primary treatment|secondary treatment",
    re.IGNORECASE,
)
# compliance/regulatory boilerplate signals — these don't appear in factual facility descriptions
BOILERPLATE_RE = re.compile(
    r"pursuant to|shall comply|must be \w+|shall be \w+"
    r"|inter-tidal|intertidal|tide gate|kayak|interpretive center"
    r"|RPA for Discharge|WQBELs|Endpoint \d+\s+is established"
    r"|133\.10|Construction, Operation, and Maintenance Specifications",
    re.IGNORECASE,
)
# "pond" is a search-only clustering term (not a keyword-match alt_name — it was removed from
# Unspecified Lagoon). Pond-heavy paragraphs (e.g. capacity/freeboard tables) otherwise have no
# vocab hits, creating a gap that truncates the description before solids/biosolids sections.
VOCAB_COMBINED_RE = re.compile(
    WASTEWATER_VOCAB_RE.pattern + "|" + DESC_PRIORITY_RE.pattern + r"|\bponds?\b",
    re.IGNORECASE,
)
# section headers: uppercase or digit start, ≤80 chars (lowercase starts match sentence wraps too often)
RAW_HEADER_RE = re.compile(r"(?:^|\n)([A-Z\d][^\n]{2,79})(?=\n)", re.MULTILINE)
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100

CHANGES_PHRASES = ["planned changes", "planned upgrade", "proposed upgrade"]
CHANGES_RE = re.compile(r"planned\s+changes|planned\s+upgrade|proposed\s+upgrade", re.IGNORECASE)

SPEC = {
    "NPDES": {"context": "attachment", "strip_toc": True},
    "NOA":   {"context": "full",       "strip_toc": False},
}
SPEC["WDR"] = SPEC["NOA"]

CLUSTER_GAP = 500          # max chars between two vocab hits to count as clustered
CLUSTER_TRAIL = 400        # chars after last cluster hit to capture trailing sentence/paragraph
LOOKBACK_HEADER = 600      # boilerplate/desc-signal check window around each cluster
SNAP_BACK_CHARS = 2000     # how far back from cluster start to look for a section header
MIN_CHANGES_VOCAB = 2      # min vocab hits in 1000 chars after "planned changes" phrase
MAX_CLUSTER_DISTANCE = 10000  # max chars from first cluster; cuts off distant general-order boilerplate
CONTIG_GAP = 2500          # max gap between consecutive qualifying clusters to treat as one description
DIVERSITY_MIN = 6          # a cluster naming >= this many distinct processes qualifies as a description



def find_attachment_f_page(raw):
    """Find Attachment F page in delimited full text. Returns (lookback_char_pos, page_num+1) or (None, None).
    Starts a few pages early (LOOKBACK_PAGES) to capture any preamble before the title page."""
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


def find_desc_clusters(text):
    """Find all vocab clusters that have a description signal nearby.

    Returns a list of (start, end) char positions — one per qualifying cluster.
    Falls back to position-discounted densest cluster if none pass the boilerplate filter.
    """
    hits = list(VOCAB_COMBINED_RE.finditer(text))
    if len(hits) < 2:
        return []
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
        return []
    # prefer clusters with a description signal; exclude clusters that look like
    # O&M/compliance boilerplate (>=2 boilerplate markers in the window)
    desc_clusters = []
    for c in clusters:
        window = text[max(0, c[0].start() - LOOKBACK_HEADER):c[-1].end()]
        # skip table-of-contents clusters (dot leaders like "......")
        if len(DOT_RE.findall(window)) >= 3:
            continue
        # A process-dense cluster (many distinct processes) is a description regardless of nearby
        # phrasing — it qualifies even without a description signal in the lookback window (header
        # across a page break) and even if compliance language trips the boilerplate filter (a
        # real treatment description that also references a Cease and Desist Order, etc.).
        if len(set(h.group().lower() for h in c)) >= DIVERSITY_MIN:
            desc_clusters.append(c)
            continue
        # otherwise: require a description signal and reject compliance/O&M boilerplate. Admin
        # tables ("Facility Information") name few processes and fall here, staying excluded.
        if not DESC_PRIORITY_RE.search(window):
            continue
        # extend boilerplate check past the cluster — compliance phrases often
        # follow the vocab hits in the same paragraph
        bpl_window = text[max(0, c[0].start() - LOOKBACK_HEADER):c[-1].end() + LOOKBACK_HEADER]
        if len(BOILERPLATE_RE.findall(bpl_window)) >= 2:
            continue
        desc_clusters.append(c)
    if desc_clusters:
        n = len(text)
        # sort by unique-term diversity / position — deprioritizes single-category clusters
        # (e.g. "chlorination × 5") and late general-order sections over rich early descriptions.
        # extract_from_pdf processes best cluster first; its start_pos anchors MAX_CLUSTER_DISTANCE,
        # so nearby multi-facility clusters are still included in order.
        desc_clusters.sort(key=lambda c: -len(set(h.group().lower() for h in c)) / (1 + c[0].start() / n))
        return [(c[0].start(), c[-1].end()) for c in desc_clusters]
    # fallback: same score — unique-term diversity / position
    n = len(text)
    best = max(clusters, key=lambda c: len(set(h.group().lower() for h in c)) / (1 + c[0].start() / n))
    return [(best[0].start(), best[-1].end())]


SECTION_NUM_RE = re.compile(r"(?:^|\n)((?:\d+|[IVX]{2,5})\.\s+[A-Z])", re.MULTILINE)


def snap_back_to_header(text, cluster_start):
    """Slide cluster_start backward to the nearest paragraph break or section header.

    Priority: numbered section header (e.g. '8. Facility Description') > last blank-line
    paragraph break > last uppercase/digit-starting line > cluster_start.
    """
    offset = max(0, cluster_start - SNAP_BACK_CHARS)
    region = text[offset:cluster_start]
    # highest priority: a numbered section header like "8. Facility Description"
    section_headers = list(SECTION_NUM_RE.finditer(region))
    if section_headers:
        return offset + section_headers[-1].start(1)
    last_para = region.rfind("\n\n")
    if last_para != -1:
        return offset + last_para + 2
    headers = list(RAW_HEADER_RE.finditer(region))
    if not headers:
        return cluster_start
    return offset + headers[-1].start(1)


def find_changes_start(text, search_start, search_end):
    """Find first 'planned changes/upgrade' match with vocab hits nearby.
    Requires phrase at a line/sentence boundary; skips negated 'no planned changes'."""
    for match in CHANGES_RE.finditer(text, search_start, search_end):
        pre = text[max(0, match.start() - 10):match.start()].lower()
        if re.search(r'\bno\s+$', pre):
            continue
        # must start on a new line or after sentence punctuation, not mid-sentence
        pre2 = text[max(0, match.start() - 2):match.start()]
        if pre2 and not re.search(r'[\n.:( ]$', pre2):
            continue
        window = text[match.start():match.start() + 1000]
        if len(VOCAB_COMBINED_RE.findall(window)) >= MIN_CHANGES_VOCAB:
            return match.start(), match.group(0).strip()
    return -1, None


def find_changes_cluster_end(text, changes_pos):
    """Find end of planned changes section by extending to the vocab cluster end."""
    hits = list(VOCAB_COMBINED_RE.finditer(text, changes_pos, changes_pos + 5000))
    if not hits:
        return min(changes_pos + CLUSTER_TRAIL, len(text))
    cluster_end = hits[0].end()
    for h in hits[1:]:
        if h.start() - cluster_end <= CLUSTER_GAP:
            cluster_end = h.end()
        else:
            break
    trail_end = min(cluster_end + CLUSTER_TRAIL, len(text))
    next_para = text.find("\n\n", cluster_end, trail_end)
    return next_para + 2 if next_para != -1 else trail_end


def extract_section(text, start, cluster_end, attachment_page, spec, mode):
    # Extend to next paragraph break after cluster end, capped at CLUSTER_TRAIL chars
    trail_end = min(cluster_end + CLUSTER_TRAIL, len(text))
    next_para = text.find("\n\n", cluster_end, trail_end)
    end = next_para + 2 if next_para != -1 else trail_end

    changes_pos, changes_start_text = find_changes_start(text, start + LOOKBACK_CHARS, len(text))
    if changes_pos == -1:
        # planned changes may precede the description (e.g. WDR Findings section)
        changes_pos, changes_start_text = find_changes_start(text, 0, start)

    changes_text = ""
    if changes_pos != -1:
        changes_end = find_changes_cluster_end(text, changes_pos)
        changes_text = text[changes_pos:changes_end].strip()
        if start <= changes_pos < end:
            end = changes_pos

    description = text[start:end].strip()
    # If the section starts mid-page, prepend the current [Page N] marker so the
    # source page is always visible at the top of every excerpt.
    if not description.startswith("[Page"):
        prior_pages = PAGE_MARKER_RE.findall(text[:start])
        if prior_pages:
            description = f"[Page {prior_pages[-1]}]\n" + description
    return {
        "txt_section": description,
        "txt_changes": changes_text,
        "full_text": text,
        "metadata": {
            "mode": mode,
            "attachment_f_page": attachment_page,
            "start_pos": start,
            "end_pos": end,
            "planned_changes_pos": changes_pos if changes_pos != -1 else None,
            "txt_section_length": len(description),
            **({"txt_changes_length": len(changes_text)} if changes_text else {}),
            "changes_start_phrase": changes_start_text,
        },
    }


def extract_from_pdf(pdf_path, mode):
    if not os.path.exists(pdf_path):
        return None
    spec = SPEC[mode]

    # PDFs with no extractable text (scanned images, no text layer) will produce
    # empty page strings throughout — extract_permit_sections flags these as "unreadable".
    reader = PdfReader(pdf_path)
    root = reader.trailer['/Root'].get_object()
    is_portfolio = '/Collection' in root
    page_parts = []
    if is_portfolio:
        for r in package_sub_readers(reader):
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
        clusters = find_desc_clusters(text)
        if not clusters:
            # no vocab clusters found — likely image-only or no treatment description text
            continue
        # clusters are score-sorted; anchor on the best, then take a run of qualifying clusters
        # around it (document order). Asymmetric: extend BACKWARD only through contiguous clusters
        # (gap <= CONTIG_GAP) so a split description starts at its earliest part (e.g. SJSC's
        # headworks before the higher-scoring filters) while NOT pulling in a preceding
        # "PERMIT INFORMATION" admin table sitting across a page gap. Extend FORWARD to all
        # qualifying clusters within MAX_CLUSTER_DISTANCE of the anchor, capturing trailing
        # solids / advanced-treatment sections (e.g. Palo Alto's MBR/RO); non-description tables
        # (effluent limits, monitoring) lack a desc signal so they never qualify and aren't reached.
        anchor = clusters[0]
        bypos = sorted(clusters, key=lambda c: c[0])
        ai = bypos.index(anchor)
        lo = ai
        while lo > 0 and bypos[lo][0] - bypos[lo - 1][1] <= CONTIG_GAP:
            lo -= 1
        hi = ai
        while hi < len(bypos) - 1 and bypos[hi + 1][0] - anchor[0] <= MAX_CLUSTER_DISTANCE:
            hi += 1
        ordered = bypos[lo:hi + 1]
        sections = []
        prev_end = 0
        for cs, ce in ordered:
            if cs < prev_end:
                continue
            start = snap_back_to_header(text, cs)
            result = extract_section(text, start, ce, attachment_page, spec, mode)
            sections.append(result)
            prev_end = result["metadata"]["end_pos"]
        combined_txt = "\n\n".join(r["txt_section"] for r in sections if r["txt_section"])
        changes = next((r["txt_changes"] for r in sections if r["txt_changes"]), "")
        return {**sections[0], "txt_section": combined_txt, "txt_changes": changes}

    full_text = contexts[-1][0] if contexts else ""
    return {"txt_section": "", "txt_changes": "", "full_text": full_text, "metadata": {"mode": mode}}


def extract_permit_sections(pdf_path, regenerate_text_excerpts=False):
    pdf_path = Path(pdf_path)
    site_data = next(
        (p / "site_data_relevant.csv" for p in [pdf_path.parent] + list(pdf_path.parents) if (p / "site_data_relevant.csv").exists()),
        pdf_path.parent.parent / "site_data_relevant.csv",
    )
    # Mode controls which part of the PDF to search and where to stop extraction.
    # NPDES: Attachment F fact sheet only. NOA/WDR: full document.
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


def extract_one(args):
    directory, pdf_file = args
    path = os.path.join(directory, pdf_file)
    return pdf_file, extract_permit_sections(path, regenerate_text_excerpts=True)


def main():
    rfr_data = f"wwtp_process_extraction/output/site_data_relevant.csv"
    directory = f"wwtp_process_extraction/output/permits"

    site_data = pd.read_csv(rfr_data, dtype=str).fillna("")
    pdfs = site_data["PDF_File"].tolist()

    # Breakdown by Reg_Measure_Type (NPDES vs WDR) over facilities with ≥1 PDF
    has_pdf = site_data[site_data["PDF_File"] != ""]["Place ID"].unique()
    df_pdf = site_data.drop_duplicates(subset="Place ID")
    df_pdf = df_pdf[df_pdf["Place ID"].isin(has_pdf)]
    print(f"\n  Reg_Measure_Type breakdown (facilities with ≥1 PDF, n={len(df_pdf)}):")
    for rmt, count in df_pdf["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count} ({count / len(df_pdf):.1%})")

    unique_pdfs = list(dict.fromkeys(p for p in pdfs if p))
    args = [(directory, pdf_file) for pdf_file in unique_pdfs]

    page_marker_re = re.compile(PAGE_MARKER_RE.pattern + r"\n?")
    flag_counts = {"unreadable": 0}
    phrase_counts = defaultdict(Counter)

    def flag(pdf_file, reason):
        cache = Path(directory) / "text" / f"{Path(pdf_file).stem}.txt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(SEP, encoding="utf-8")
        flag_counts[reason] += 1

    with ProcessPoolExecutor(max_workers=20) as executor:
        for pdf_file, r in executor.map(extract_one, args):
            print(f"Processed {pdf_file}")
            if r is None:
                continue
            txt = r.get("txt_section", "")
            full_text = r.get("full_text", "")
            # flag as unreadable if no description AND very little non-marker text
            # (these are scanned image PDFs with no text layer — needs OCR)
            if not txt and len(page_marker_re.sub("", full_text).strip()) < 100:
                flag(pdf_file, "unreadable")
            val = (r.get("metadata") or {}).get("changes_start_phrase")
            if val:
                phrase_counts["changes_start_phrase"][re.sub(r"\s+", " ", val.lower().strip())] += 1

    norm = lambda s: re.sub(r"\s+", " ", s.lower().strip())
    ref_lists = {
        "changes_start_phrase": ("Planned changes start", CHANGES_PHRASES),
    }
    print(f"Non-machine-readable PDFs: {flag_counts['unreadable']}")
    print()
    for key, (label, ref) in ref_lists.items():
        counts = Counter({norm(t): 0 for t in ref})
        counts.update(phrase_counts[key])
        print(f"{label}:")
        for term in sorted(ref, key=lambda t: -counts[norm(t)]):
            print(f"  {counts[norm(term)]:4d}  {term!r}")
        print()

    txt_dir = Path(directory) / "text"
    txt_files = list(txt_dir.glob("*.txt"))
    if txt_files:
        sizes = [(f, f.stat().st_size) for f in txt_files]
        sizes.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop text cache files in {txt_dir}:")
        for f, size in sizes[:10]:
            print(f"  {f.name}: {size} bytes")

    site_data["Total_PDFs_Available"] = pd.to_numeric(
        site_data["Total_PDFs_Available"], errors="coerce"
    )
    reg_measures = site_data.groupby("Reg_Measure_ID")["Total_PDFs_Available"].max().reset_index()
    total_available = int(reg_measures["Total_PDFs_Available"].fillna(0).sum())
    print(f"Total permits: {len(reg_measures)}")
    print(f"Total PDFs available across permits: {total_available}")


if __name__ == "__main__":
    main()
