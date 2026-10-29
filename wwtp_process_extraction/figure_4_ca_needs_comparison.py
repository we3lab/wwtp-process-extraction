# CA wastewater treatment capital needs, rebuilt from permit-extracted planned changes
# using EPA's own CWNS 2022 cost curves (Cost Estimation Tool Methods, Table 2-3).
# CWNS has no CA unit-process records, so it cannot produce a bottom-up CA estimate at all;
# this figure contrasts that structural zero with CWNS's reported documented need.

import json

import matplotlib.pyplot as plt
import pandas as pd

from helpers.paths import *
from helpers.utils import (
    parse_status, current_permit_mask, PRESENT_STATUSES, cwns_mapping, leaves, leaf_names,
    column_top_category,
)
from helpers.plotting import COLORS, save_and_close, set_thick_spines

CURVES_JSON = DATA_DIR / "cwns_cost_curves.json"
FLOW_CSV = DATA_DIR / "cwns/2022/FLOW.csv"
NEEDS_CSV = DATA_DIR / "cwns/2022/NEEDS_COST_BY_CATEGORY.csv"
NEEDS_DIR = OUTPUT_DIR / "needs"

TREATMENT_CATEGORIES = ["I", "II"]  # secondary + advanced treatment
# EPA's Table 2-3 curves are fitted below 5 MGD for new/replacement, 2.1 MGD for lagoon rehab
MAX_MGD = 5.0
EXPANSION_TOLERANCE = 0.05  # future design flow must exceed current by >5% to count as expansion
DISINFECTION = {"Chlorination", "UV Disinfection"}
# a facility takes the highest tier it contains
TIERS = ["lagoon", "aerated_lagoon", "secondary_mechanical", "advanced"]

# Facilities without a CWNS_ID, stacked as one grey band
UNMATCHED = "no_cwns_match"

CATEGORY_LABELS = {
    "new": "New facility",
    "system_expansion": "Existing facility expansion",
    "treatment_upgrade": "Existing facility upgrade",
    "rehabilitation": "Existing facility rehabilitation",
    UNMATCHED: "No CWNS match",
}

# add_disinfection is costed off its own EPA curve (UV via CAFCom, chlorine via Capdet) but is
# still a new-unit build, so the figure folds it into New. ca_needs_summary.csv keeps the split.
DISPLAY_GROUP = {"add_disinfection": "new"}

# One permit-extraction bar per snapshot. Only the yyyy-06-01 folders are comparable annual
# scrapes; a same-day run (e.g. 2026-08-13) can be partial and would read as a real decline.
SNAPSHOT_GLOB = "20??-06-01"
# CWNS 2022 is one survey, so its marker is compared to the permits in force that year
CWNS_SNAPSHOT = "2022-06-01"

# Leaves whose taxonomy group would misclassify them. Nitrification sits under Activated
# Sludge but is ammonia removal (EPA Category II); Denitrification Filter sits under
# Biofiltration but is nutrient removal.
ADVANCED_EXTRA = {"UV-AOP", "Activated Carbon", "Ion Exchange", "Denitrification Filter", "Nitrification"}
SECONDARY_MECH_EXTRA = {
    "Trickling Filter", "Rotating Biological Contactor", "Moving Bed Biofilm Reactor",
    "Membrane Aerated Biofilm Reactor", "Unspecified FFR", "Biologically Active Filtration",
}

TICK_FONTSIZE = 12
LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 11


def system_type(processes, lookup):
    """Highest tier present among a facility's processes."""
    tiers = [lookup[p] for p in processes if p in lookup]
    return max(tiers, key=TIERS.index) if tiers else None


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


def cost_facilities(facilities, curves, lookup, pid_to_cwns, flow, ciwqs_flow):
    """Cost one snapshot's facilities (Jan-2022 dollars); returns the full per-facility frame."""
    scale = curves["cpi_adjustment"] * curves["location_factor"] / curves["national_average_location_factor"]
    rows = []
    for fac in facilities:
        present, future = fac["present"], fac["future"]
        if not future:
            continue

        cwns_id = pid_to_cwns.get(fac["Place ID"], "")
        cur_mgd = flow["CURRENT_DESIGN_FLOW"].get(cwns_id)
        fut_mgd = flow["FUTURE_DESIGN_FLOW"].get(cwns_id)
        flow_source = "CWNS"
        if pd.isna(cur_mgd):
            # no CWNS match; CIWQS gives one permitted design flow and no future value,
            # so expansion can't be inferred for these facilities
            cur_mgd = ciwqs_flow.get(fac["WDID"])
            fut_mgd = None
            flow_source = "" if pd.isna(cur_mgd) else "CIWQS"

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
        no_flow = pd.isna(mgd) or mgd <= 0
        if no_flow or (sys_type is None and construction != "add_disinfection"):
            row.update({"cost_2022usd": None, "within_curve_limit": None, "curve_basis": "",
                        "excluded_reason": "no design flow" if no_flow else "unclassified system"})
        else:
            if construction == "add_disinfection":
                spec = curves["systems"]["add_disinfection"][disinfectant]
            else:
                spec = curves["systems"][sys_type][construction]
            segment = next(s for s in spec["segments"] if s["max_mgd"] is None or mgd <= s["max_mgd"])
            a, b = segment["a"], segment["b"]
            base = a * mgd + b if segment["form"] == "linear" else a * mgd ** b
            row.update({"cost_2022usd": base * scale, "within_curve_limit": mgd <= spec["limit_mgd"],
                        "curve_basis": segment["basis"], "excluded_reason": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    curves = json.loads(CURVES_JSON.read_text())

    group_of = {name: group_id for name, _, group_id in leaves}
    lookup = {}
    for leaf, top in column_top_category.items():
        if leaf in ADVANCED_EXTRA or top == "Nutrient Removal" or group_of[leaf] == "Membrane Process":
            lookup[leaf] = "advanced"
        elif leaf in SECONDARY_MECH_EXTRA or top == "Activated Sludge":
            lookup[leaf] = "secondary_mechanical"
        elif leaf == "Aerated Lagoon":
            lookup[leaf] = "aerated_lagoon"
        elif top == "Lagoon":
            lookup[leaf] = "lagoon"

    # per-document, so a snapshot's document set can be costed on its own
    perdoc = pd.read_csv(PDF_LLM_CSV, dtype=str).fillna("")

    # cwns_mapping is already ciwqs_to_cwns.csv filtered to rows with a real CWNS_ID
    pid_to_cwns = cwns_mapping.drop_duplicates("Place ID").set_index("Place ID")["CWNS_ID"]

    flow = pd.read_csv(FLOW_CSV, dtype={"CWNS_ID": str})
    flow = flow[(flow["STATE_CODE"] == "CA") & (flow["FLOW_TYPE"] == "Total Flow")]
    for col in ("CURRENT_DESIGN_FLOW", "FUTURE_DESIGN_FLOW"):
        flow[col] = pd.to_numeric(flow[col], errors="coerce")
    flow = flow.set_index("CWNS_ID")[["CURRENT_DESIGN_FLOW", "FUTURE_DESIGN_FLOW"]]

    site = pd.read_csv(SITE_DATA_ALL_CSV, dtype=str)
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

    snapshots = sorted(p.name for p in (OUTPUT_DIR / "site_data").glob(SNAPSHOT_GLOB))

    # Extracted documents each facility held at each as-of date, cut to the permits in force
    # then (superseded orders stay attached to a facility's CIWQS page), as in step6 and figure_5
    n_docs, facs_by_year = {}, {}
    for as_of in snapshots:
        rel = pd.read_csv(OUTPUT_DIR / "site_data" / as_of / "site_data_relevant.csv", dtype=str)
        pairs = set(zip(rel["Place ID"], rel["PDF_File"]))
        held = [pair in pairs for pair in zip(perdoc["Place ID"], perdoc["PDF_File"])]
        docs = perdoc[held]
        docs = docs[current_permit_mask(docs, as_of=as_of)]
        n_docs[as_of] = len(docs)

        # Per-facility PRESENT/FUTURE process sets, unioned over the documents in force. A document's
        # extraction never changes, so year-over-year change comes only from which documents are in force.
        by_place = {}
        for _, row in docs.iterrows():
            fac = by_place.setdefault(row["Place ID"], {
                "Place ID": row["Place ID"], "WDID": row["WDID"], "Facility Name": row["Facility Name"],
                "present": set(), "expansion": set(), "planned": set(),
            })
            for col in leaf_names:
                status = parse_status(row[col])
                if status in PRESENT_STATUSES:
                    fac["present"].add(col)
                if status == "PRESENT_AND_FUTURE":
                    fac["expansion"].add(col)
                elif status == "FUTURE":
                    fac["planned"].add(col)
        # A plain FUTURE is dropped when a sibling document reports the process as present;
        # PRESENT_AND_FUTURE stays, being an expansion of something already there
        for fac in by_place.values():
            fac["future"] = fac["expansion"] | (fac["planned"] - fac["present"])
        facs_by_year[as_of] = list(by_place.values())

    # Balanced cohort: only facilities with an extracted document in EVERY snapshot, so totals move on
    # permit renewals, not on document coverage growing. It grows as step5 works through older documents.
    cohort_pids = set.intersection(*({f["Place ID"] for f in facs} for facs in facs_by_year.values()))
    print(f"balanced cohort: {len(cohort_pids)} facilities with an extracted document in all "
          f"{len(snapshots)} snapshots "
          f"(of {max(len(f) for f in facs_by_year.values())} in the fullest single year)")

    per_year, year_rows = {}, []
    for as_of in snapshots:
        facs = [f for f in facs_by_year[as_of] if f["Place ID"] in cohort_pids]
        est = cost_facilities(facs, curves, lookup, pid_to_cwns, flow, ciwqs_flow)
        # The costed cohort: within EPA's fitted MGD limit and under the small-plant cutoff
        costed = est[est["cost_2022usd"].notna()]
        small = costed[costed["mgd_used"] < MAX_MGD]
        scored = small[small["within_curve_limit"]]
        per_year[as_of] = scored
        # Per-category columns cover only the CWNS-matched facilities, as the figure stacks them
        is_matched = scored["CWNS_ID"].ne("")
        by_cat = (scored[is_matched].replace({"construction_type": DISPLAY_GROUP})
                  .groupby("construction_type")["cost_2022usd"].sum())
        year_rows.append({
            "as_of": as_of, "documents_in_force": n_docs[as_of],
            "cohort_facilities": len(cohort_pids),
            "facilities_with_planned_changes": len(est),
            "facilities_costed": len(scored),
            "total_2022usd": scored["cost_2022usd"].sum(),
            "cwns_matched_facilities": int(is_matched.sum()),
            "cwns_matched_2022usd": scored.loc[is_matched, "cost_2022usd"].sum(),
            "no_cwns_match_facilities": int((~is_matched).sum()),
            "no_cwns_match_2022usd": scored.loc[~is_matched, "cost_2022usd"].sum(),
            **{f"{cat}_2022usd": by_cat[cat]
               for cat in ("new", "system_expansion", "treatment_upgrade", "rehabilitation")},
        })
        print(f"  {as_of}: {n_docs[as_of]:5} documents in force, {len(est):3} cohort "
              f"facilities with planned changes, {len(scored):3} costed, "
              f"${scored['cost_2022usd'].sum()/1e6:,.0f}M")
    # est is left holding the newest snapshot, the headline estimate

    # CWNS reported need over the same facilities we costed in the CWNS year, so the marker is like-for-like
    cwns_matched = per_year[CWNS_SNAPSHOT]
    cwns_matched = cwns_matched[cwns_matched["CWNS_ID"].ne("")]
    cwns_ids = set(cwns_matched["CWNS_ID"])
    cwns_reported = needs_by_id.reindex(sorted(cwns_ids)).fillna(0).sum()
    print(f"CWNS {CWNS_SNAPSHOT[:4]}: {len(cwns_ids)} matched facilities, ours "
          f"${cwns_matched['cost_2022usd'].sum()/1e6:,.0f}M "
          f"vs CWNS reported ${cwns_reported/1e6:,.0f}M")

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

    est.sort_values(["system_type", "construction_type", "Facility Name"]).to_csv(
        NEEDS_DIR / "ca_needs_summary.csv", index=False)

    pd.DataFrame(year_rows).to_csv(NEEDS_DIR / "ca_needs_by_year.csv", index=False)

    plot(per_year, cwns_reported, len(cwns_ids))
    print(f"\nwrote {NEEDS_DIR/'ca_needs_summary.csv'}, {NEEDS_DIR/'ca_needs_by_year.csv'} "
          f"and {FIGURES_DIR/'figure_4'}.png/.tiff")


def plot(per_year, cwns_reported, n_cohort):
    """Permit-extracted need as a stacked area over annual snapshots, with CWNS as one point.
    CWNS is one survey vintage, so it's one marker, not a line, and has no comparable breakdown."""
    keys = sorted(per_year)
    years = [int(k[:4]) for k in keys]
    fig, ax = plt.subplots(figsize=(6, 5))

    # bottom-to-top: biggest, steadiest category first so the thin ones ride on a flat base
    order = ("rehabilitation", "treatment_upgrade", "system_expansion", "new")
    stack_order = order + (UNMATCHED,)
    shades = {"rehabilitation": "#8fabd2", "treatment_upgrade": "#5c82b8",
              "system_expansion": "#305993", "new": "#1f3b63", UNMATCHED: "#b3b9c0"}

    by_cat = {cat: [] for cat in stack_order}
    for k in keys:
        scored = per_year[k].replace({"construction_type": DISPLAY_GROUP})
        # CWNS_ID is "" for facilities with no CWNS match
        is_matched = scored["CWNS_ID"].ne("")
        groups = {cat: scored[is_matched & (scored["construction_type"] == cat)] for cat in order}
        groups[UNMATCHED] = scored[~is_matched]
        for cat in stack_order:
            by_cat[cat].append(groups[cat]["cost_2022usd"].sum() / 1e6)
    # labels quote the final year's facility counts
    n_cat = {cat: len(group) for cat, group in groups.items()}

    ax.stackplot(years, *[by_cat[c] for c in stack_order],
                 colors=[shades[c] for c in stack_order], edgecolor="white")

    cwns_m = cwns_reported / 1e6
    cwns_year = int(CWNS_SNAPSHOT[:4])
    ax.plot(cwns_year, cwns_m, marker="o", markersize=11, zorder=5,
            color=COLORS["Clean Watershed Needs Survey"],
            markeredgecolor="white", markeredgewidth=1.4)
    ax.annotate(f"CWNS reported  ${cwns_m:,.0f}M (n = {n_cohort})", xy=(cwns_year, cwns_m),
                xytext=(8, 6), textcoords="offset points", va="bottom",
                fontsize=LEGEND_FONTSIZE, color=COLORS["Clean Watershed Needs Survey"])

    # Band edges at the final year, so labels can sit near the right edge inside their own band.
    edges, bottom = {}, 0.0
    for cat in stack_order:
        edges[cat] = (bottom, bottom + by_cat[cat][-1])
        bottom += by_cat[cat][-1]
    peak_total = max(sum(by_cat[c][i] for c in stack_order) for i in range(len(years)))
    label_x = years[-1] - 0.12

    # The thick bands hold their label inside; ink is chosen for contrast against the
    # fill, dark on the lighter shades and white on the darker one.
    inside_ink = {"rehabilitation": "#12263f", "treatment_upgrade": "#12263f",
                  "system_expansion": "white", UNMATCHED: "#12263f"}

    for cat, ink in inside_ink.items():
        band_bottom, band_top = edges[cat]
        ax.annotate(f"{CATEGORY_LABELS[cat]} (n = {n_cat[cat]})",
                    xy=(label_x, (band_bottom + band_top) / 2),
                    ha="right", va="center", fontsize=LEGEND_FONTSIZE, color=ink)

    # New is a thin band, so it is labelled just above its own top edge at the left, where the
    # stack is lowest -- close enough to read without a leader line crossing other bands.
    new_top_first = sum(by_cat[c][0] for c in order)
    ax.annotate(f"{CATEGORY_LABELS['new']} (n = {n_cat['new']})",
                xy=(years[0] + 0.08, new_top_first), xytext=(0, 4), textcoords="offset points",
                va="bottom", fontsize=LEGEND_FONTSIZE, color="#12263f")

    ax.set_xticks(years)
    ax.set_xticklabels(years, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    # No n on the axis: the coloured stack and the CWNS marker cover n_cohort matched plants,
    # the grey band adds the unmatched ones, so a single n would describe neither.
    ax.set_ylabel(f"Estimated Investment Need\n"
                  f"for CA WWTPs below {MAX_MGD:.0f} MGD"
                  f"\n(Million USD, 2022)",
                  fontsize=LABEL_FONTSIZE)
    ax.set_xlabel("Permit extraction, as of 1 June", fontsize=LABEL_FONTSIZE)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, max(cwns_m, peak_total) * 1.16)
    set_thick_spines(ax)
    fig.tight_layout()
    save_and_close(fig, FIGURES_DIR / "figure_4")


if __name__ == "__main__":
    main()
