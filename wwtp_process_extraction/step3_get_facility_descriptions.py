import io
import os
import re
import pandas as pd
import pdfplumber
from collections import Counter
from PyPDF2 import PdfReader
from concurrent.futures import ProcessPoolExecutor
from helpers.utils import (
    leaves, SEP, normalize_text, is_general_order, OUTPUT_DIR, TXT_DIR,
    SITE_DATA_RELEVANT_CSV,
)


DOT_RE = re.compile(r"\.{5,}")
ATTACHMENT_F_RE = re.compile(r"ATTACHMENT\s+F\s*[-–—‐]\s*FACT\s+SHEET", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]\n?")
RAW_PAGE_MARKER_RE = re.compile(r"===PAGE (\d+)===\n?")
# A document's own order number from its title block. Anchoring on the "ORDER NO." label lifts
# accuracy from 68% to 90%; the remaining misses are scanned title pages
ORDER_NUM = r"(?:R\d{1,2}[A-Z]?[-\s])?(?:WQ[-\s])?(?:20\d{2}|9\d)[-\s]\d{3,4}(?:[-\s](?:DWQ|EXEC))?"
ORDER_LABELLED_RE = re.compile(r"ORDER\s*(?:NO\.?|NUMBER)?\s*[:\s]\s*(" + ORDER_NUM + r")", re.IGNORECASE)
ORDER_ANY_RE = re.compile(r"\b(" + ORDER_NUM + r")\b", re.IGNORECASE)
ORDER_HEAD_CHARS = 2500
WASTEWATER_VOCAB_RE = re.compile(
    "|".join(
        re.escape(term)
        for _, details, _ in leaves
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
# "consists of"-style sentences mark a real description even when boilerplate follows
# (common in small general-order permits)
STRONG_DESC_RE = re.compile(r"consists? of|consisting of|comprised of|comprising", re.IGNORECASE)
# "pond" is a clustering-only term (not an alt_name): without it, pond-heavy paragraphs have no
# vocab hits and cut the description off before its solids sections
VOCAB_COMBINED_RE = re.compile(
    WASTEWATER_VOCAB_RE.pattern + "|" + DESC_PRIORITY_RE.pattern + r"|\bponds?\b",
    re.IGNORECASE,
)
# section headers: uppercase or digit start, ≤80 chars (lowercase starts match sentence wraps too often)
RAW_HEADER_RE = re.compile(r"(?:^|\n)([A-Z\d][^\n]{2,79})(?=\n)", re.MULTILINE)
SECTION_NUM_RE = re.compile(r"(?:^|\n)((?:\d+|[IVX]{2,5})\.\s+[A-Z])", re.MULTILINE)
# Fact-sheet regulatory section headers; everything after one is compliance boilerplate. Keyed on
# headers, not phrases like "secondary treatment standards" that real descriptions also use
REG_HEADER_RE = re.compile(
    r"applicable plans,?\s+policies"
    r"|other plans,?\s+policies\s+and\s+regulations"
    r"|secondary treatment regulations",
    re.IGNORECASE,
)
LOOKBACK_PAGES = 2
LOOKBACK_CHARS = 100

CHANGES_PHRASES = ["planned changes", "planned upgrade", "proposed upgrade"]
CHANGES_RE = re.compile("|".join(phrase.replace(" ", r"\s+") for phrase in CHANGES_PHRASES), re.IGNORECASE)

# Reg_Measure_Types searched as a full document (NOA/WDR); every other type is searched as an
# NPDES permit, Attachment F fact sheet only.
FULL_DOCUMENT_TYPES = {"ENROLLEE - NPDES", "ENROLLEE - WDR", "WDR", "INDIVIDUAL MONITORING REQUIREM"}

PERMITS_DIR = OUTPUT_DIR / "permits"
MAX_WORKERS = 24

CLUSTER_GAP = 500          # max chars between two vocab hits to count as clustered
CLUSTER_TRAIL = 400        # chars after last cluster hit to capture trailing sentence/paragraph
LOOKBACK_HEADER = 600      # boilerplate/desc-signal check window around each cluster
SNAP_BACK_CHARS = 2000     # how far back from cluster start to look for a section header
MIN_CHANGES_VOCAB = 2      # min vocab hits in 1000 chars after "planned changes" phrase
MAX_CLUSTER_DISTANCE = 10000  # max chars from first cluster; cuts off distant general-order boilerplate
CONTIG_GAP = 2500          # max gap between consecutive qualifying clusters to treat as one description
DIVERSITY_MIN = 6          # a cluster naming >= this many distinct processes qualifies as a description
FRAGMENT_GAP = 1000        # absorb a low-diversity raw cluster trailing a qualifying one if this close


def clean_excerpt(text):
    # keep page boundaries as [Page N] so source pages are traceable in excerpts
    text = RAW_PAGE_MARKER_RE.sub(r"[Page \1]\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def find_attachment_f_page(raw):
    """Return the char position to start Attachment F extraction at, or None.
    Starts a few pages early (LOOKBACK_PAGES) to capture any preamble before the title page."""
    pages = list(RAW_PAGE_MARKER_RE.finditer(raw))
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
    start, _, n_terms = cluster
    return n_terms / (1 + start / text_length)


def find_desc_clusters(text, multi_facility=False):
    """Find all vocab clusters that have a description signal nearby.

    Returns a list of (start, end) char positions — one per qualifying cluster.
    Falls back to position-discounted densest cluster if none pass the boilerplate filter.
    """
    hits = list(VOCAB_COMBINED_RE.finditer(text))
    if len(hits) < 2:
        return []
    # runs of >= 2 hits with gaps <= CLUSTER_GAP, as (start, end, distinct terms), in text order
    clusters = []
    run = [hits[0]]
    for hit in hits[1:] + [None]:  # None closes the last run
        if hit and hit.start() - run[-1].end() <= CLUSTER_GAP:
            run.append(hit)
            continue
        if len(run) >= 2:
            clusters.append((run[0].start(), run[-1].end(), len({h.group().lower() for h in run})))
        run = [hit]
    if not clusters:
        return []
    # prefer clusters with a description signal; exclude clusters that look like
    # O&M/compliance boilerplate (>=2 boilerplate markers in the window)
    desc_clusters = []
    for cluster in clusters:
        start, end, n_terms = cluster
        window = text[max(0, start - LOOKBACK_HEADER):end]
        # skip table-of-contents clusters (dot leaders like "......")
        if len(DOT_RE.findall(window)) >= 3:
            continue
        # Many distinct processes, or a "consists of" sentence, qualifies on its own: no description
        # signal needed, and nearby compliance language doesn't disqualify it
        if n_terms >= DIVERSITY_MIN or STRONG_DESC_RE.search(window):
            desc_clusters.append(cluster)
            continue
        # otherwise: require a description signal and reject compliance/O&M boilerplate. Admin
        # tables ("Facility Information") name few processes and fall here, staying excluded.
        if not DESC_PRIORITY_RE.search(window):
            continue
        # extend boilerplate check past the cluster — compliance phrases often
        # follow the vocab hits in the same paragraph
        boilerplate_window = text[max(0, start - LOOKBACK_HEADER):end + LOOKBACK_HEADER]
        if len(BOILERPLATE_RE.findall(boilerplate_window)) >= 2:
            continue
        desc_clusters.append(cluster)
    text_length = len(text)
    if desc_clusters:
        # best first: extract_from_pdf anchors MAX_CLUSTER_DISTANCE on the first cluster
        desc_clusters.sort(key=lambda cluster: -cluster_score(cluster, text_length))
        if not multi_facility:
            return [(start, end) for start, end, _ in desc_clusters]
        # Multi-facility permits: absorb a low-diversity tail (disinfection/disposal sentences)
        # within FRAGMENT_GAP of a plant's paragraph, but never another qualifying cluster
        qualifying = set(desc_clusters)
        spans = []
        for cluster in desc_clusters:
            start, end, _ = cluster
            next_index = clusters.index(cluster) + 1
            while (next_index < len(clusters)
                   and clusters[next_index] not in qualifying
                   and clusters[next_index][0] - end <= FRAGMENT_GAP):
                end = clusters[next_index][1]
                next_index += 1
            spans.append((start, end))
        return spans
    # fallback: densest cluster by the same score
    start, end, _ = max(clusters, key=lambda cluster: cluster_score(cluster, text_length))
    return [(start, end)]


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
        if match.start() > 0 and text[match.start() - 1] not in "\n.:( ":
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
        "end_pos": end,
        "changes_start_phrase": changes_start_text,
    }


def extract_from_pdf(pdf_path, is_npdes, multi_facility):
    if not os.path.exists(pdf_path):
        return None

    # PDFs with no extractable text (scanned images, no text layer) will produce
    # empty page strings throughout — main flags these as "unreadable".
    root = PdfReader(pdf_path).trailer['/Root'].get_object()
    page_parts = []
    if '/Collection' in root:
        # PDF Portfolio: read each embedded PDF (page numbers restart in each)
        embedded = root['/Names'].get_object()['/EmbeddedFiles'].get_object()['/Names']
        for file_spec in embedded[1::2]:
            streams = file_spec.get_object()['/EF'].get_object()
            sub_reader = PdfReader(io.BytesIO((streams.get('/F') or streams['/UF']).get_object().get_data()))
            for page_num, page in enumerate(sub_reader.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    else:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_parts.append(f"===PAGE {page_num}===")
                page_parts.append(page.extract_text() or "")
    raw = "\n".join(page_parts)

    # OCR fallback for scanned PDFs (105 documents, ~6000 pages), left off. It caches raw text so
    # reruns don't re-OCR; 300 dpi is the verified-legible setting.
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

    # Search region: the Attachment F fact sheet for NPDES permits, the full document for NOA/WDR

    # The order number printed in the document's own title block, or ''
    head = RAW_PAGE_MARKER_RE.sub("", raw[:ORDER_HEAD_CHARS])
    order_match = ORDER_LABELLED_RE.search(head) or ORDER_ANY_RE.search(head)
    doc_order = order_match.group(1).upper().replace(" ", "-") if order_match else ""

    # No Attachment F reference anywhere (44 documents, mostly pre-dating the fact-sheet
    # convention) leaves the full document, as for NOA/WDR.
    search_text = raw
    if is_npdes:
        attachment_pos = find_attachment_f_page(raw)
        first_attachment_match = ATTACHMENT_F_RE.search(raw)
        if first_attachment_match and (first_attachment_match.start() < 500 or attachment_pos is None):
            attachment_pos = 0
        if attachment_pos is not None:
            search_text = raw[attachment_pos:]
            if attachment_pos > 0:
                # skip past a table of contents (dot leaders) to the fact sheet itself
                dot_leader_hits = list(DOT_RE.finditer(search_text[:20000]))
                if len(dot_leader_hits) >= 2 and not DOT_RE.search(search_text[dot_leader_hits[-1].end():dot_leader_hits[-1].end() + 500]):
                    search_text = search_text[dot_leader_hits[-1].end():]
                elif (toc_restart := search_text.lower().find("attachment f", 1000)) != -1:
                    search_text = search_text[toc_restart:]
    text = clean_excerpt(search_text)

    clusters = find_desc_clusters(text, multi_facility=multi_facility)
    if clusters:
        # Anchor on the best cluster. Extend backward only through contiguous clusters (keeps an
        # earlier part of a split description, not a preceding admin table); extend forward to
        # every qualifying cluster within MAX_CLUSTER_DISTANCE (trailing solids/advanced sections)
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
        # Stop at the first regulatory-section header after the description start (past it so the
        # anchor itself is never dropped)
        reg_match = REG_HEADER_RE.search(text, reference_start + 1)
        reg_stop = reg_match.start() if reg_match else len(text)
        ordered = [(cluster_start, cluster_end) for cluster_start, cluster_end in ordered if cluster_start < reg_stop]
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
            prev_end = section["end_pos"]
        combined_txt = "\n\n".join(section["txt_section"] for section in sections if section["txt_section"])
        changes = next((section["txt_changes"] for section in sections if section["txt_changes"]), "")
        return {**sections[0], "txt_section": combined_txt, "txt_changes": changes,
                "document_order_no": doc_order}

    # no vocab clusters found — likely image-only or no treatment description text
    return {"txt_section": "", "txt_changes": "", "full_text": text, "changes_start_phrase": None,
            "document_order_no": doc_order}


def extract_one(args):
    pdf_file, is_npdes, multi_facility = args
    pdf_path = PERMITS_DIR / pdf_file
    out = extract_from_pdf(str(pdf_path), is_npdes, multi_facility)
    cache = TXT_DIR / f"{pdf_path.stem}.txt"
    if out and out.get("general_order"):
        cache.unlink(missing_ok=True)  # drop text left by an earlier run, before this was caught
    elif out:
        cache.write_text(out["txt_section"] + SEP + out["txt_changes"], encoding="utf-8")
    return pdf_file, out


def main():
    site_data = pd.read_csv(SITE_DATA_RELEVANT_CSV, dtype=str).fillna("")

    # Breakdown by Reg_Measure_Type (NPDES vs WDR) over facilities with ≥1 PDF
    has_pdf = site_data[site_data["PDF_File"] != ""]["Place ID"].unique()
    df_pdf = site_data.drop_duplicates(subset="Place ID")
    df_pdf = df_pdf[df_pdf["Place ID"].isin(has_pdf)]
    print(f"\n  Reg_Measure_Type breakdown (facilities with ≥1 PDF, n={len(df_pdf)}):")
    for rmt, count in df_pdf["Reg_Measure_Type"].value_counts(dropna=False).items():
        print(f"    {rmt}: {count} ({count / len(df_pdf):.1%})")

    args = []
    for pdf_file, pdf_rows in site_data[site_data["PDF_File"] != ""].groupby("PDF_File", sort=False):
        # First matching row's type decides
        is_npdes = pdf_rows["Reg_Measure_Type"].iloc[0].strip().upper() not in FULL_DOCUMENT_TYPES
        # Multi-facility permit: one PDF covering several distinct Place IDs (e.g. OCSD, IEUA).
        multi_facility = len(set(pdf_rows["Place ID"].str.strip()) - {""}) > 1
        args.append((pdf_file, is_npdes, multi_facility))

    flag_counts = {"unreadable": 0, "general_order": 0, "no_desc_in_attachment": 0}
    doc_orders = {}
    phrase_counts = Counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pdf_file, result in executor.map(extract_one, args):
            print(f"Processed {pdf_file}")
            if result is None:
                continue
            if result.get("general_order"):
                flag_counts["general_order"] += 1
                continue
            txt = result["txt_section"]
            # flag as unreadable if no description AND very little non-marker text
            # (these are scanned image PDFs with no text layer — needs OCR)
            if not txt and len(PAGE_MARKER_RE.sub("", result["full_text"]).strip()) < 100:
                flag_counts["unreadable"] += 1
            elif not txt:
                # readable document that still yielded no description
                flag_counts["no_desc_in_attachment"] += 1
            doc_orders[pdf_file] = result["document_order_no"]
            if result["changes_start_phrase"]:
                phrase_counts[normalize_text(result["changes_start_phrase"])] += 1

    # Each document's own order number, to distinguish superseded order PDFs 
    site_data["document_order_no"] = site_data["PDF_File"].map(doc_orders).fillna("")
    site_data.to_csv(SITE_DATA_RELEVANT_CSV, index=False)

    print(f"Non-machine-readable PDFs: {flag_counts['unreadable']}")
    print(f"Statewide general orders skipped: {flag_counts['general_order']}")
    print(f"Readable but no description found: {flag_counts['no_desc_in_attachment']}")
    print("Planned changes start:")
    for term in sorted(CHANGES_PHRASES, key=lambda t: -phrase_counts[normalize_text(t)]):
        print(f"  {phrase_counts[normalize_text(term)]:4d}  {term!r}")
    print()

    print(f"\nTop text cache files in {TXT_DIR}:")
    for f in sorted(TXT_DIR.glob("*.txt"), key=lambda f: f.stat().st_size, reverse=True)[:10]:
        print(f"  {f.name}: {f.stat().st_size} bytes")

    site_data["Total_PDFs_Available"] = pd.to_numeric(
        site_data["Total_PDFs_Available"], errors="coerce"
    )
    reg_measures = site_data.groupby("Reg_Measure_ID")["Total_PDFs_Available"].max().reset_index()
    total_available = int(reg_measures["Total_PDFs_Available"].fillna(0).sum())
    print(f"Total permits: {len(reg_measures)}")
    print(f"Total PDFs available across permits: {total_available}")


if __name__ == "__main__":
    main()
