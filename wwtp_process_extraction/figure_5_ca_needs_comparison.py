# CA wastewater treatment capital needs, rebuilt from permit-extracted planned changes
# using EPA's own CWNS 2022 cost curves (Cost Estimation Tool Methods, Table 2-3).
# CWNS has no CA unit-process records, so it cannot produce a bottom-up CA estimate at all;
# this figure contrasts that structural zero with CWNS's reported documented need.

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.utils import extract_leaves, parse_status, PRESENT_STATUSES
from helpers.plotting import COLORS, save_and_close, set_thick_spines

DATA = Path("wwtp_process_extraction/data")
OUT = Path("wwtp_process_extraction/output")
CURVES_JSON = DATA / "cwns_cost_curves.json"
KEYWORDS_JSON = DATA / "unitprocess_keywords.json"
FLOW_CSV = DATA / "cwns/2022/FLOW.csv"
NEEDS_CSV = DATA / "cwns/2022/NEEDS_COST_BY_CATEGORY.csv"
MAPPING_CSV = DATA / "ciwqs_to_cwns.csv"
LLM_CSV = OUT / "unit_processes_by_facility_llm.csv"
# per-document, so a snapshot's document set can be costed on its own
PERDOC_CSV = OUT / "unit_processes_by_pdf_llm.csv"

TREATMENT_CATEGORIES = ["I", "II"]  # secondary + advanced treatment
# EPA's Table 2-3 curves are fitted below 5 MGD for new/replacement, 2.1 MGD for lagoon rehab
MAX_MGD = 5.0
EXPANSION_TOLERANCE = 0.05  # future design flow must exceed current by >5% to count as expansion
DISINFECTION = {"Chlorination", "UV Disinfection"}
# a facility takes the highest tier it contains
TIERS = ["lagoon", "aerated_lagoon", "secondary_mechanical", "advanced"]

CATEGORY_LABELS = {
    "new": "New",
    "system_expansion": "Expansion",
    "treatment_upgrade": "Upgrade",
    "rehabilitation": "Rehabilitation",
    "add_disinfection": "New (Disinfection)",
}

META_COLS = {"Place ID", "WDID", "Order_No", "NPDES No.", "Agency", "Facility Name", "County",
             "PDF_File"}

# One permit-extraction bar per snapshot. Only the yyyy-06-01 folders are comparable annual
# scrapes; a same-day run (e.g. 2026-08-13) can be partial and would read as a real decline.
SNAPSHOT_GLOB = "20??-06-01"

# Leaves whose taxonomy group would misclassify them. Nitrification sits under Activated
# Sludge but is ammonia removal (EPA Category II); Denitrification Filter sits under
# Biofiltration but is nutrient removal.
ADVANCED_EXTRA = {"UV-AOP", "Activated Carbon", "Ion Exchange", "Denitrification Filter", "Nitrification"}
SECONDARY_MECH_EXTRA = {
    "Trickling Filter", "Rotating Biological Contactor", "Moving Bed Biofilm Reactor",
    "Membrane Aerated Biofilm Reactor", "Unspecified FFR", "Biologically Active Filtration",
}


def system_type(processes, lookup):
    """Highest tier present among a facility's processes."""
    tiers = [lookup[p] for p in processes if p in lookup]
    return max(tiers, key=TIERS.index) if tiers else None


def evaluate_curve(curves, system, construction, mgd):
    """Return (cost, within_limit, basis) for one facility. Cost is Jan-2022 dollars."""
    spec = curves["systems"][system][construction]
    segment = next(s for s in spec["segments"] if s["max_mgd"] is None or mgd <= s["max_mgd"])
    base = segment["a"] * mgd + segment["b"] if segment["form"] == "linear" else segment["a"] * mgd ** segment["b"]
    scale = curves["cpi_adjustment"] * curves["location_factor"] / curves["national_average_location_factor"]
    return base * scale, mgd <= spec["limit_mgd"], segment["basis"]


def infer_construction(present, future, cur_mgd, fut_mgd, lookup):
    """Construction type from the PRESENT vs FUTURE process delta. First rule wins."""
    if not present:
        return "new", None
    planned_disinfection = future & DISINFECTION
    if planned_disinfection and not (future - DISINFECTION - {"Dechlorination"}):
        return "add_disinfection", ("uv" if "UV Disinfection" in planned_disinfection else "chlorine")
    present_tier = system_type(present, lookup)
    future_tier = system_type(present | future, lookup)
    if future_tier and present_tier and TIERS.index(future_tier) > TIERS.index(present_tier):
        return "treatment_upgrade", None
    if cur_mgd and fut_mgd and fut_mgd > cur_mgd * (1 + EXPANSION_TOLERANCE):
        return "system_expansion", None
    return "rehabilitation", None


def snapshot_documents(as_of):
    """(Place ID, PDF_File) pairs a facility held at one as-of date."""
    rel = pd.read_csv(OUT / "site_data" / as_of / "site_data_relevant.csv", dtype=str).fillna("")
    rel = rel[rel["PDF_File"].str.strip().ne("")]
    return set(zip(rel["Place ID"].str.strip(), rel["PDF_File"].str.strip()))


def facilities_as_of(perdoc, process_cols, doc_pairs):
    """Per-facility PRESENT/FUTURE process sets, unioned over that snapshot's documents only.

    A document's extraction never changes between years, so the year-over-year signal comes
    entirely from which documents were in force -- a permit renewal replacing an older order.

    A facility's order page often carries superseded orders as attachments, and those plan
    projects that later orders report as built: Pinole's 2018 order plans nitrification/MLE
    that its 2023 order lists as PRESENT. So a plain FUTURE is dropped when any sibling
    document reports the same process as present -- 27% of FUTURE claims at multi-document
    facilities. PRESENT_AND_FUTURE survives, being an expansion of something already there.
    """
    by_place = {}
    for _, row in perdoc.iterrows():
        key = (str(row["Place ID"]).strip(), str(row["PDF_File"]).strip())
        if key not in doc_pairs:
            continue
        fac = by_place.setdefault(key[0], {
            "Place ID": key[0], "WDID": row["WDID"], "Facility Name": row["Facility Name"],
            "present": set(), "expansion": set(), "planned": set(),
        })
        for col in process_cols:
            status = parse_status(row[col])
            if status in PRESENT_STATUSES:
                fac["present"].add(col)
            if status == "PRESENT_AND_FUTURE":
                fac["expansion"].add(col)
            elif status == "FUTURE":
                fac["planned"].add(col)

    for fac in by_place.values():
        fac["future"] = fac["expansion"] | (fac["planned"] - fac["present"])
    return list(by_place.values())


def main():
    curves = json.loads(CURVES_JSON.read_text())
    keywords = json.loads(KEYWORDS_JSON.read_text())

    top_of, group_of = {}, {}
    for top, val in keywords.items():
        for name, _details, group_id in extract_leaves({top: val}):
            top_of[name] = top
            group_of[name] = group_id

    lookup = {}
    for leaf in top_of:
        if leaf in ADVANCED_EXTRA or top_of[leaf] == "Nutrient Removal" or group_of[leaf] == "Membrane Process":
            lookup[leaf] = "advanced"
        elif leaf in SECONDARY_MECH_EXTRA or top_of[leaf] == "Activated Sludge":
            lookup[leaf] = "secondary_mechanical"
        elif leaf == "Aerated Lagoon":
            lookup[leaf] = "aerated_lagoon"
        elif top_of[leaf] == "Lagoon":
            lookup[leaf] = "lagoon"

    perdoc = pd.read_csv(PERDOC_CSV, dtype=str).fillna("")
    if "PDF_File" not in perdoc.columns:
        raise SystemExit(f"{PERDOC_CSV.name} has no PDF_File column — re-run step6 "
                         "(it now records each extraction's source document).")
    process_cols = [c for c in perdoc.columns if c not in META_COLS]

    mapping = pd.read_csv(MAPPING_CSV, dtype=str).fillna("")
    pid_to_cwns = (mapping[mapping["CWNS_ID"].str.strip().ne("") & mapping["CWNS_ID"].str.upper().ne("NA")]
                   .drop_duplicates("Place ID").set_index("Place ID")["CWNS_ID"].str.strip())

    flow = pd.read_csv(FLOW_CSV, dtype={"CWNS_ID": str})
    flow = flow[(flow["STATE_CODE"] == "CA") & (flow["FLOW_TYPE"] == "Total Flow")]
    assert flow["CWNS_ID"].is_unique, "expected one Total Flow row per CA CWNS_ID"
    for col in ("CURRENT_DESIGN_FLOW", "FUTURE_DESIGN_FLOW"):
        flow[col] = pd.to_numeric(flow[col], errors="coerce")
    flow = flow.set_index("CWNS_ID")[["CURRENT_DESIGN_FLOW", "FUTURE_DESIGN_FLOW"]]

    site = pd.read_csv(OUT / "site_data_all.csv", dtype=str)
    site["Design Flow"] = pd.to_numeric(site["Design Flow"], errors="coerce")
    site = site[site["Design Flow"] > 0]
    ciwqs_flow = site.drop_duplicates("WDID").set_index("WDID")["Design Flow"]

    needs = pd.read_csv(NEEDS_CSV, dtype={"CWNS_ID": str})
    needs = needs[needs["STATE_CODE"] == "CA"]
    treat = needs[needs["NEEDS_CATEGORY"].isin(TREATMENT_CATEGORIES)].copy()
    treat["OFFICIAL_AMOUNT"] = pd.to_numeric(treat["OFFICIAL_AMOUNT"], errors="coerce").fillna(0)
    needs_by_id = treat.groupby("CWNS_ID")["OFFICIAL_AMOUNT"].sum()
    has_advanced = set(needs.loc[needs["NEEDS_CATEGORY"] == "II", "CWNS_ID"])
    has_secondary = set(needs.loc[needs["NEEDS_CATEGORY"] == "I", "CWNS_ID"])

    def cost_facilities(facilities):
        """Cost one snapshot's facilities; returns the full per-facility frame."""
        rows = []
        for fac in facilities:
            present, future = fac["present"], fac["future"]
            if not future:
                continue

            cwns_id = pid_to_cwns.get(str(fac["Place ID"]).strip(), "")
            cur_mgd = flow["CURRENT_DESIGN_FLOW"].get(cwns_id)
            fut_mgd = flow["FUTURE_DESIGN_FLOW"].get(cwns_id)
            flow_source = "CWNS"
            if cur_mgd is None or pd.isna(cur_mgd):
                # no CWNS match; CIWQS gives one permitted design flow and no future value,
                # so expansion can't be inferred for these facilities
                cur_mgd = ciwqs_flow.get(str(fac["WDID"]).strip())
                fut_mgd = None
                flow_source = "CIWQS" if cur_mgd is not None and not pd.isna(cur_mgd) else ""

            construction, disinfectant = infer_construction(present, future, cur_mgd, fut_mgd, lookup)
            mgd = fut_mgd if construction in ("new", "system_expansion") else cur_mgd
            sys_type = system_type(present | future, lookup)

            row = {
                "Place ID": fac["Place ID"], "CWNS_ID": cwns_id,
                "Facility Name": fac["Facility Name"],
                "system_type": sys_type, "construction_type": construction,
                "disinfectant": disinfectant or "",
                "current_mgd": cur_mgd, "future_mgd": fut_mgd, "mgd_used": mgd,
                "flow_source": flow_source,
                "n_future_processes": len(future),
                "future_processes": "; ".join(sorted(future)),
            }
            if pd.isna(mgd) or mgd is None or mgd <= 0 or (sys_type is None and construction != "add_disinfection"):
                row.update({"cost_2022usd": None, "within_curve_limit": None, "curve_basis": "",
                            "excluded_reason": "no design flow" if (mgd is None or pd.isna(mgd) or mgd <= 0) else "unclassified system"})
            else:
                system_key = "add_disinfection" if construction == "add_disinfection" else sys_type
                curve_key = disinfectant if construction == "add_disinfection" else construction
                cost, in_limit, basis = evaluate_curve(curves, system_key, curve_key, mgd)
                row.update({"cost_2022usd": cost, "within_curve_limit": in_limit,
                            "curve_basis": basis, "excluded_reason": ""})
            rows.append(row)
        return pd.DataFrame(rows)

    def reportable(est):
        """The costed cohort: within EPA's fitted MGD limit and under the small-plant cutoff."""
        costed = est[est["cost_2022usd"].notna()]
        small = costed[costed["mgd_used"] < MAX_MGD]
        return small[small["within_curve_limit"]].copy()

    snapshots = sorted(p.name for p in (OUT / "site_data").glob(SNAPSHOT_GLOB))
    if not snapshots:
        raise SystemExit(f"no {SNAPSHOT_GLOB} snapshots under {OUT/'site_data'}")

    docs_by_year = {as_of: snapshot_documents(as_of) for as_of in snapshots}
    facs_by_year = {as_of: facilities_as_of(perdoc, process_cols, docs)
                    for as_of, docs in docs_by_year.items()}

    # Balanced cohort: only facilities with an extracted document in EVERY snapshot. Without it
    # the series is a coverage curve, not a trend -- the count of facilities holding an extracted
    # document grows steeply across snapshots while the share of them planning work stays flat
    # near 36%, so absolute totals would rise on document availability alone. Restricted this
    # way, year-over-year movement can only come from a permit renewal changing what is planned.
    # The cohort grows as step5 works through older documents; re-run this after step5/step6.
    cohort_pids = set.intersection(*({f["Place ID"] for f in facs} for facs in facs_by_year.values()))
    print(f"balanced cohort: {len(cohort_pids)} facilities with an extracted document in all "
          f"{len(snapshots)} snapshots "
          f"(of {max(len(f) for f in facs_by_year.values())} in the fullest single year)")

    per_year, year_rows = {}, []
    for as_of in snapshots:
        facs = [f for f in facs_by_year[as_of] if f["Place ID"] in cohort_pids]
        est_y = cost_facilities(facs)
        scored_y = reportable(est_y)
        per_year[as_of] = scored_y
        year_rows.append({
            "as_of": as_of, "documents_in_force": len(docs_by_year[as_of]),
            "cohort_facilities": len(cohort_pids),
            "facilities_with_planned_changes": len(est_y),
            "facilities_costed": len(scored_y),
            "total_2022usd": scored_y["cost_2022usd"].sum(),
        })
        print(f"  {as_of}: {len(docs_by_year[as_of]):5} documents in force, {len(est_y):3} cohort "
              f"facilities with planned changes, {len(scored_y):3} costed, "
              f"${scored_y['cost_2022usd'].sum()/1e6:,.0f}M")

    # The newest snapshot is the headline estimate and the like-for-like cohort for CWNS
    latest = snapshots[-1]
    est = cost_facilities([f for f in facs_by_year[latest] if f["Place ID"] in cohort_pids])
    scored = reportable(est)

    # CWNS reported need over the same facilities we costed, so the bars are like-for-like
    cohort = set(scored.loc[scored["CWNS_ID"].ne(""), "CWNS_ID"])
    cwns_reported = needs_by_id.reindex(sorted(cohort)).fillna(0).sum()

    est["cwns_reported_I_II_2022usd"] = est["CWNS_ID"].map(needs_by_id)
    est["cwns_has_advanced_need"] = est["CWNS_ID"].isin(has_advanced)
    est["cwns_has_secondary_need"] = est["CWNS_ID"].isin(has_secondary)

    # Validate the advanced/secondary split against EPA's own Category II (advanced treatment)
    # designation, over facilities CWNS actually assigned a treatment category.
    check = est[est["system_type"].notna() & (est["cwns_has_advanced_need"] | est["cwns_has_secondary_need"])]
    ours_adv = check["system_type"] == "advanced"
    agree = (ours_adv == check["cwns_has_advanced_need"]).sum()

    over = int((ours_adv & ~check["cwns_has_advanced_need"]).sum())
    under = int((~ours_adv & check["cwns_has_advanced_need"]).sum())
    print(f"advanced/secondary agreement vs CWNS Category II: {agree}/{len(check)} "
            f"({100*agree/len(check):.0f}%); ours advanced only {over}, CWNS Cat II only {under}")
    print("  (weak proxy: NEEDS_CATEGORY describes the funded project, not the plant's "
            "treatment level, so an advanced plant can carry a Category I need and vice versa)")

    needs_dir = OUT / "needs"
    needs_dir.mkdir(parents=True, exist_ok=True)
    est.sort_values(["system_type", "construction_type", "Facility Name"]).to_csv(
        needs_dir / "ca_needs_summary.csv", index=False)

    year_df = pd.DataFrame(year_rows)
    year_df.to_csv(needs_dir / "ca_needs_by_year.csv", index=False)

    plot(per_year, cwns_reported, len(cohort))
    print(f"\nwrote {needs_dir/'ca_needs_summary.csv'}, {needs_dir/'ca_needs_by_year.csv'} "
          f"and {OUT/'final'/'figure_5'}.png/.tiff")

TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 11
PANEL_FONTSIZE = 16
SPINE_WIDTH = 1.6

def plot(per_year, cwns_reported, n_cohort):
    """CWNS's documented need, then one permit-extraction bar per annual snapshot.

    Each permit bar is stacked by capital-need category; CWNS's is not, because it records
    dollars per (document, needs category) and change types per (facility, facility type) with
    no link between them, so a matching breakdown would be invented. Only the most recent bar
    carries segment labels -- repeating them on every year would just be a legend six times.
    """
    years = sorted(per_year)
    fig, ax = plt.subplots(figsize=(1.05 * (len(years) + 1) + 3.0, 5))

    order = ("new", "system_expansion", "treatment_upgrade", "rehabilitation", "add_disinfection")
    shades = ["#1f3b63", "#305993ff", "#5c82b8", "#8fabd2", "#c2d2e8"]

    ax.bar(0, cwns_reported / 1e6, width=0.55, color=COLORS["Clean Watershed Needs Survey"],
           edgecolor="black", linewidth=0.4)
    ax.text(0, cwns_reported / 1e6, f"${cwns_reported/1e6:,.0f}M",
            ha="center", va="bottom", fontsize=LEGEND_FONTSIZE)

    last_x, label_targets = len(years), []
    for xi, as_of in enumerate(years, start=1):
        by_type = per_year[as_of].groupby("construction_type")["cost_2022usd"].sum() / 1e6
        bottom = 0.0
        for name, shade in zip(order, shades):
            v = by_type.get(name, 0.0)
            if v <= 0:
                continue
            ax.bar(xi, v, bottom=bottom, width=0.55, color=shade,
                   edgecolor="black", linewidth=0.4)
            if as_of == years[-1]:
                label_targets.append((CATEGORY_LABELS.get(name, name), bottom + v / 2))
            bottom += v
        ax.text(xi, bottom, f"${bottom:,.0f}M", ha="center", va="bottom", fontsize=LEGEND_FONTSIZE)

    # Nudge labels apart where thin segments would overlap (New and Expansion are a few $M each),
    # keeping each leader line anchored on its own segment's midpoint.
    top = ax.get_ylim()[1]
    min_gap = 0.055 * top
    placed = []
    for name, mid in sorted(label_targets, key=lambda t: t[1]):
        y = mid if not placed else max(mid, placed[-1][2] + min_gap)
        placed.append((name, mid, y))
    for name, mid, y in placed:
        ax.annotate(name, xy=(last_x + 0.29, mid), xytext=(last_x + 0.42, y),
                    ha="left", va="center", fontsize=LEGEND_FONTSIZE,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35",
                                    shrinkA=0, shrinkB=0))

    ax.set_xticks(range(len(years) + 1))
    ax.set_xticklabels(["CWNS\nreported"] + [y[:4] for y in years], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.set_ylabel(f"CA treatment capital need, Cat. I+II\n"
                  f"(Jan-2022 $M, plants < {MAX_MGD:.0f} MGD, n = {n_cohort})",
                  fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Permit extraction, as of 1 June", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(-0.5, len(years) + 1.9)
    set_thick_spines(ax, linewidth=SPINE_WIDTH)
    fig.tight_layout()
    save_and_close(fig, OUT / "final" / "figure_5", dpi=300)


if __name__ == "__main__":
    main()
