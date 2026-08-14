import os
import re
import csv
import pandas as pd
import pdfplumber
from collections import Counter, defaultdict
from PyPDF2 import PdfReader
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from helpers.utils import extract_leaves, SEP, unitprocess_keywords, package_sub_readers, normalize_text, is_general_order


def clean_excerpt(text):
    # keep page boundaries as [Page N] so source pages are traceable in excerpts
    text = re.sub(r"===PAGE (\d+)===\n?", r"[Page \1]\n", text)
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
# strong structural description signals: a sentence enumerating what a system "consists of" is a
# real description even when enrollment/compliance boilerplate follows it (common in small General
# Order permits, e.g. "the community system consists of septic tanks ... a community leachfield").
STRONG_DESC_RE = re.compile(r"consists? of|consisting of|comprised of|comprising", re.IGNORECASE)
# "pond" is a search-only clustering term (not a keyword-match alt_name — it was removed from
# Unspecified Lagoon). Pond-heavy paragraphs (e.g. capacity/freeboard tables) otherwise have no
# vocab hits, creating a gap that truncates the description before solids/biosolids sections.
VOCAB_COMBINED_RE = re.compile(
    WASTEWATER_VOCAB_RE.pattern + "|" + DESC_PRIORITY_RE.pattern + r"|\bponds?\b",
    re.IGNORECASE,
)
# section headers: uppercase or digit start, ≤80 chars (lowercase starts match sentence wraps too often)
RAW_HEADER_RE = re.compile(r"(?:^|\n)([A-Z\d][^\n]{2,79})(?=\n)", re.MULTILINE)
SECTION_NUM_RE = re.compile(r"(?:^|\n)((?:\d+|[IVX]{2,5})\.\s+[A-Z])", re.MULTILINE)
# recurring CA fact-sheet regulatory section headers — everything after one of these is
# compliance boilerplate (40 CFR, TBELs, 303(d), Ocean Plan), not facility description. Keyed on
# the section header, not inline phrases like "secondary treatment standards" which appear in real
# descriptions (e.g. r5-2024-0009's biosolids/percolation-pond text).
REG_HEADER_RE = re.compile(
    r"applicable plans,?\s+policies"
    r"|other plans,?\s+policies\s+and\s+regulations"
    r"|secondary treatment regulations",
    re.IGNORECASE,
)
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
FRAGMENT_GAP = 1000        # absorb a low-diversity raw cluster trailing a qualifying one if this close



def find_attachment_f_page(raw):
    """Return the char position to start Attachment F extraction at, or None.
    Starts a few pages early (LOOKBACK_PAGES) to capture any preamble before the title page."""
    page_re = re.compile(r"===PAGE (\d+)===\n")
    pages = list(page_re.finditer(raw))
    for page_index, page_match in enumerate(pages):
        page_num = int(page_match.group(1))
        if page_num < 10:
            continue
        next_start = pages[page_index + 1].start() if page_index + 1 < len(pages) else len(raw)
        page_text = raw[page_match.end():next_start]
        if ATTACHMENT_F_RE.search(page_text) and len(DOT_RE.findall(page_text)) < 3:
            lookback_index = max(0, page_index - LOOKBACK_PAGES)
            return pages[lookback_index].start()
    return None


def cluster_score(cluster, text_length):
    # unique-term diversity discounted by position — deprioritizes single-category clusters
    # (e.g. "chlorination × 5") and late sections over rich early descriptions
    diversity = len(set(hit.group().lower() for hit in cluster))
    return diversity / (1 + cluster[0].start() / text_length)


def find_desc_clusters(text, multi_facility=False):
    """Find all vocab clusters that have a description signal nearby.

    Returns a list of (start, end) char positions — one per qualifying cluster.
    Falls back to position-discounted densest cluster if none pass the boilerplate filter.
    """
    hits = list(VOCAB_COMBINED_RE.finditer(text))
    if len(hits) < 2:
        return []
    clusters = []
    current = [hits[0]]
    for hit in hits[1:]:
        if hit.start() - current[-1].end() <= CLUSTER_GAP:
            current.append(hit)
        else:
            if len(current) >= 2:
                clusters.append(current)
            current = [hit]
    if len(current) >= 2:
        clusters.append(current)
    if not clusters:
        return []
    # prefer clusters with a description signal; exclude clusters that look like
    # O&M/compliance boilerplate (>=2 boilerplate markers in the window)
    desc_clusters = []
    for cluster in clusters:
        window = text[max(0, cluster[0].start() - LOOKBACK_HEADER):cluster[-1].end()]
        # skip table-of-contents clusters (dot leaders like "......")
        if len(DOT_RE.findall(window)) >= 3:
            continue
        # A process-dense cluster (many distinct processes) is a description regardless of nearby
        # phrasing — it qualifies even without a description signal in the lookback window (header
        # across a page break) and even if compliance language trips the boilerplate filter (a
        # real treatment description that also references a Cease and Desist Order, etc.). A strong
        # structural signal ("consists of") qualifies the same way: it marks a real description even
        # when enrollment boilerplate follows, which would otherwise trip the boilerplate filter.
        if len(set(hit.group().lower() for hit in cluster)) >= DIVERSITY_MIN or STRONG_DESC_RE.search(window):
            desc_clusters.append(cluster)
            continue
        # otherwise: require a description signal and reject compliance/O&M boilerplate. Admin
        # tables ("Facility Information") name few processes and fall here, staying excluded.
        if not DESC_PRIORITY_RE.search(window):
            continue
        # extend boilerplate check past the cluster — compliance phrases often
        # follow the vocab hits in the same paragraph
        boilerplate_window = text[max(0, cluster[0].start() - LOOKBACK_HEADER):cluster[-1].end() + LOOKBACK_HEADER]
        if len(BOILERPLATE_RE.findall(boilerplate_window)) >= 2:
            continue
        desc_clusters.append(cluster)
    text_length = len(text)
    if desc_clusters:
        # sort best-first by cluster_score — extract_from_pdf processes the best cluster first;
        # its start_pos anchors MAX_CLUSTER_DISTANCE, so nearby multi-facility clusters are
        # still included in order.
        desc_clusters.sort(key=lambda cluster: -cluster_score(cluster, text_length))
        if not multi_facility:
            return [(cluster[0].start(), cluster[-1].end()) for cluster in desc_clusters]
        # multi-facility permits describe each plant in a single dense paragraph that often
        # has a low-diversity tail (disinfection/disposal-routing sentences) which individually
        # falls below DIVERSITY_MIN — absorb a trailing raw cluster within FRAGMENT_GAP rather
        # than drop it, but never swallow another already-qualifying cluster (that one gets its
        # own section, or merges via the normal CONTIG_GAP path in extract_from_pdf).
        desc_cluster_ids = {id(cluster) for cluster in desc_clusters}
        clusters_by_start = sorted(clusters, key=lambda cluster: cluster[0].start())
        spans = []
        for cluster in desc_clusters:
            start, end = cluster[0].start(), cluster[-1].end()
            next_index = clusters_by_start.index(cluster) + 1
            while (next_index < len(clusters_by_start)
                   and id(clusters_by_start[next_index]) not in desc_cluster_ids
                   and clusters_by_start[next_index][0].start() - end <= FRAGMENT_GAP):
                end = clusters_by_start[next_index][-1].end()
                next_index += 1
            spans.append((start, end))
        return spans
    # fallback: densest cluster by the same score
    densest_cluster = max(clusters, key=lambda cluster: cluster_score(cluster, text_length))
    return [(densest_cluster[0].start(), densest_cluster[-1].end())]


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
        preceding = text[max(0, match.start() - 10):match.start()].lower()
        if re.search(r'\bno\s+$', preceding):
            continue
        # must start on a new line or after sentence punctuation, not mid-sentence
        preceding_two = text[max(0, match.start() - 2):match.start()]
        if preceding_two and not re.search(r'[\n.:( ]$', preceding_two):
            continue
        window = text[match.start():match.start() + 1000]
        if len(VOCAB_COMBINED_RE.findall(window)) >= MIN_CHANGES_VOCAB:
            return match.start(), match.group(0).strip()
    return -1, None


def extend_to_paragraph_break(text, pos):
    """Extend pos to the next paragraph break (\\n\\n), capped at CLUSTER_TRAIL chars."""
    trail_end = min(pos + CLUSTER_TRAIL, len(text))
    next_para = text.find("\n\n", pos, trail_end)
    return next_para + 2 if next_para != -1 else trail_end


def find_changes_cluster_end(text, changes_pos):
    """Find end of planned changes section by extending to the vocab cluster end."""
    hits = list(VOCAB_COMBINED_RE.finditer(text, changes_pos, changes_pos + 5000))
    if not hits:
        return min(changes_pos + CLUSTER_TRAIL, len(text))
    cluster_end = hits[0].end()
    for hit in hits[1:]:
        if hit.start() - cluster_end <= CLUSTER_GAP:
            cluster_end = hit.end()
        else:
            break
    return extend_to_paragraph_break(text, cluster_end)


def extract_section(text, start, cluster_end, end_cap=None):
    # Extend to next paragraph break after cluster end, capped at CLUSTER_TRAIL chars
    end = extend_to_paragraph_break(text, cluster_end)
    # never let the excerpt bleed past a regulatory-section header
    if end_cap is not None:
        end = min(end, end_cap)

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
            "start_pos": start,
            "end_pos": end,
            "changes_start_phrase": changes_start_text,
        },
    }


def extract_from_pdf(pdf_path, mode, multi_facility=False):
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
        for sub_reader in package_sub_readers(reader):
            for page_num, page in enumerate(sub_reader.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    raw = "\n".join(page_parts)

    # OCR fallback for scanned PDFs with no text layer (105 documents, ~6000 pages). Left off:
    # step3 re-extracts every PDF on each run, so enabling it without the cache below would
    # re-OCR everything every time. Cache raw text (not sections) so section-logic changes
    # don't force re-OCR. 300 dpi is the verified-legible setting.
    # if len(re.sub(r"===PAGE \d+===\n?", "", raw).strip()) < 100:
    #     ocr_cache = Path(pdf_path).parent / "ocr" / f"{Path(pdf_path).stem}.txt"
    #     if ocr_cache.exists():
    #         raw = ocr_cache.read_text(encoding="utf-8")
    #     else:
    #         import io, fitz, pytesseract
    #         from PIL import Image
    #         doc = fitz.open(pdf_path)
    #         parts = []
    #         for page_num, page in enumerate(doc):
    #             parts.append(f"===PAGE {page_num}===")
    #             png = page.get_pixmap(dpi=300).tobytes("png")
    #             parts.append(pytesseract.image_to_string(Image.open(io.BytesIO(png))))
    #         doc.close()
    #         raw = "\n".join(parts)
    #         ocr_cache.parent.mkdir(parents=True, exist_ok=True)
    #         ocr_cache.write_text(raw, encoding="utf-8")

    # A statewide general order describes no single plant — its generic process list would be
    # attributed to every enrolled facility. Skip before any section extraction.
    if is_general_order(raw):
        return {"general_order": True}

    # Build the single text region to search. NPDES: the Attachment F fact sheet.
    # NOA/WDR: the full document.
    text = None
    if spec["context"] == "attachment":
        attachment_pos = find_attachment_f_page(raw)
        first_attachment_match = ATTACHMENT_F_RE.search(raw)
        if first_attachment_match and first_attachment_match.start() < 500:
            attachment_pos = 0
        elif attachment_pos is None and first_attachment_match:
            attachment_pos = 0
        if attachment_pos is not None:
            attachment_text = raw[attachment_pos:]
            if spec["strip_toc"] and attachment_pos > 0:
                dot_leader_hits = list(DOT_RE.finditer(attachment_text[:20000]))
                if len(dot_leader_hits) >= 2 and not DOT_RE.search(attachment_text[dot_leader_hits[-1].end():dot_leader_hits[-1].end() + 500]):
                    attachment_text = attachment_text[dot_leader_hits[-1].end():]
                elif (toc_restart := attachment_text.lower().find("attachment f", 1000)) != -1:
                    attachment_text = attachment_text[toc_restart:]
            text = clean_excerpt(attachment_text)
        if text is None:
            # No Attachment F reference anywhere (44 documents, mostly pre-dating the fact-sheet
            # convention) — the full document is all there is. Only reachable when the scoped
            # path found nothing, so it cannot change a document that already extracts.
            text = clean_excerpt(raw)
    else:
        text = clean_excerpt(raw)

    clusters = find_desc_clusters(text, multi_facility=multi_facility) if text is not None else []
    if clusters:
        # clusters are score-sorted; anchor on the best, then take a run of qualifying clusters
        # around it (document order). Asymmetric: extend BACKWARD only through contiguous clusters
        # (gap <= CONTIG_GAP) so a split description starts at its earliest part (e.g. SJSC's
        # headworks before the higher-scoring filters) while NOT pulling in a preceding
        # "PERMIT INFORMATION" admin table sitting across a page gap. Extend FORWARD to all
        # qualifying clusters within MAX_CLUSTER_DISTANCE of the anchor, capturing trailing
        # solids / advanced-treatment sections (e.g. Palo Alto's MBR/RO); non-description tables
        # (effluent limits, monitoring) lack a desc signal so they never qualify and aren't reached.
        clusters_by_position = sorted(clusters, key=lambda cluster: cluster[0])
        if multi_facility:
            # A multi-facility permit describes several plants in separate regions, so
            # the single-anchor window misses them — take every qualifying cluster.
            ordered = clusters_by_position
            reference_start = clusters_by_position[0][0]
        else:
            anchor = clusters[0]
            anchor_index = clusters_by_position.index(anchor)
            first_index = anchor_index
            while first_index > 0 and clusters_by_position[first_index][0] - clusters_by_position[first_index - 1][1] <= CONTIG_GAP:
                first_index -= 1
            last_index = anchor_index
            while last_index < len(clusters_by_position) - 1 and clusters_by_position[last_index + 1][0] - anchor[0] <= MAX_CLUSTER_DISTANCE:
                last_index += 1
            ordered = clusters_by_position[first_index:last_index + 1]
            reference_start = anchor[0]
        # Stop at the first regulatory-section header after the description start — everything
        # after it is compliance boilerplate (40 CFR, TBELs, 303(d)), not facility description.
        # Search past reference_start so the anchor itself is never dropped.
        reg_match = REG_HEADER_RE.search(text, reference_start + 1)
        reg_stop = reg_match.start() if reg_match else len(text)
        ordered = [(cs, ce) for cs, ce in ordered if cs < reg_stop]
        # merge clusters separated by <= CONTIG_GAP so contiguous description prose
        # (low-vocab continuation of the same paragraph) is kept whole, not dropped in the gap
        merged = []
        for cluster_start, cluster_end in ordered:
            if merged and cluster_start - merged[-1][1] <= CONTIG_GAP:
                merged[-1] = (merged[-1][0], cluster_end)
            else:
                merged.append((cluster_start, cluster_end))
        ordered = merged
        sections = []
        prev_end = 0
        for cluster_start, cluster_end in ordered:
            if cluster_start < prev_end:
                continue
            start = snap_back_to_header(text, cluster_start)
            section = extract_section(text, start, cluster_end, end_cap=reg_stop)
            sections.append(section)
            prev_end = section["metadata"]["end_pos"]
        combined_txt = "\n\n".join(section["txt_section"] for section in sections if section["txt_section"])
        changes = next((section["txt_changes"] for section in sections if section["txt_changes"]), "")
        return {**sections[0], "txt_section": combined_txt, "txt_changes": changes}

    # no vocab clusters found — likely image-only or no treatment description text
    return {"txt_section": "", "txt_changes": "", "full_text": text or "", "metadata": {}}


def extract_permit_sections(pdf_path):
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
        pdf_rows = [row for row in csv.DictReader(f) if (row.get("PDF_File") or "").strip() == pdf_path.name]
    mode = next(
        (mode_map.get((row.get("Reg_Measure_Type") or "").strip().upper(), "NPDES") for row in pdf_rows),
        "NPDES",
    )
    # Multi-facility permit: one PDF covering several distinct Place IDs (e.g. OCSD, IEUA).
    multi_facility = len({(row.get("Place ID") or "").strip() for row in pdf_rows} - {""}) > 1
    out = extract_from_pdf(str(pdf_path), mode=mode, multi_facility=multi_facility)
    cache = pdf_path.parent / "text" / f"{pdf_path.stem}.txt"
    if out and out.get("general_order"):
        cache.unlink(missing_ok=True)  # drop text left by an earlier run, before this was caught
    elif out:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(out["txt_section"] + SEP + out["txt_changes"], encoding="utf-8")
    return out


def extract_one(args):
    directory, pdf_file = args
    path = os.path.join(directory, pdf_file)
    return pdf_file, extract_permit_sections(path)


def main():
    relevant_sites_csv = "wwtp_process_extraction/output/site_data_relevant.csv"
    directory = "wwtp_process_extraction/output/permits"

    site_data = pd.read_csv(relevant_sites_csv, dtype=str).fillna("")
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
    flag_counts = {"unreadable": 0, "general_order": 0, "no_desc_in_attachment": 0}
    phrase_counts = defaultdict(Counter)

    def flag(pdf_file, reason):
        cache = Path(directory) / "text" / f"{Path(pdf_file).stem}.txt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(SEP, encoding="utf-8")
        flag_counts[reason] += 1

    with ProcessPoolExecutor(max_workers=24) as executor:
        for pdf_file, result in executor.map(extract_one, args):
            print(f"Processed {pdf_file}")
            if result is None:
                continue
            if result.get("general_order"):
                flag_counts["general_order"] += 1
                continue
            txt = result.get("txt_section", "")
            full_text = result.get("full_text", "")
            # flag as unreadable if no description AND very little non-marker text
            # (these are scanned image PDFs with no text layer — needs OCR)
            if not txt and len(page_marker_re.sub("", full_text).strip()) < 100:
                flag(pdf_file, "unreadable")
            elif not txt:
                # readable document that still yielded no description
                flag_counts["no_desc_in_attachment"] += 1
            val = (result.get("metadata") or {}).get("changes_start_phrase")
            if val:
                phrase_counts["changes_start_phrase"][normalize_text(val)] += 1

    ref_lists = {
        "changes_start_phrase": ("Planned changes start", CHANGES_PHRASES),
    }
    print(f"Non-machine-readable PDFs: {flag_counts['unreadable']}")
    print(f"Statewide general orders skipped: {flag_counts['general_order']}")
    print(f"Readable but no description found: {flag_counts['no_desc_in_attachment']}")
    print()
    for key, (label, ref) in ref_lists.items():
        counts = Counter({normalize_text(t): 0 for t in ref})
        counts.update(phrase_counts[key])
        print(f"{label}:")
        for term in sorted(ref, key=lambda t: -counts[normalize_text(t)]):
            print(f"  {counts[normalize_text(term)]:4d}  {term!r}")
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
