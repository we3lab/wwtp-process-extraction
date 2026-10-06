# Adapted from https://github.com/jiananf2/US_WWTP_GHG/tree/main/GHG_accounting/WWTP_GHG_accounting.py
# by Abigayle Hodson, Abigayle_Hodson@lbl.gov
# Publication: https://eartharxiv.org/repository/view/7980/
# Modified by WE3Lab: California-specific comparison across three treatment process data sources.
#
#   Common set = Place IDs present in BOTH unit_processes_by_facility_cwns.csv AND llm output.

#
# NOTES:
#   - Biosolids emissions use per-TT 50th-percentile from MC distributions (not
#     facility-specific EPA biosolids data), consistent across all sources.
#   - Electricity uses facility-specific grid carbon intensity (WWTP balancing area).
#   - Biogas electricity (E-train variants) is absent from Source 2 only (BIOGAS_EL=0).
#   - TT assignment fallback uses El Abbadi's EPA region × flow-size bins (all CA = Region 9,
#     so only flow-size bins differentiate: <2,2-4,4-7,7-16,16-46,46-100,≥100 MGD).
#   - Fallback is applied to every source using the within-source assigned pool.

import ast
import io
import json
import urllib.request
from functools import cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from helpers.plotting import make_grouped_legend, save_and_close
from helpers.utils import (build_cwns_facility_processes, leaves,
                          CWNS_TABLE_CSV, CIWQS_TO_CWNS_CSV, DATA_DIR, OUTPUT_DIR,
                          FINAL_DIR, PRESENT_STATUSES, STATUS_TOKENS, current_permit_mask,
                          collapse_facility_processes)

# Local copies of the El Abbadi US_WWTP_GHG files, downloaded from GitHub (gitignored)
GHG_CACHE_DIR = DATA_DIR / 'ghg_cache'
MC_DIR = 'uncertainty_sensitivity_results/Monte_Carlo'
# The 50th-percentile factors distilled from the ~180 MB of upstream Monte Carlo workbooks.
# Cached because it is 49 rows of derived numbers; delete it to refetch and recompute.
MC_EF_CSV = DATA_DIR / 'ghg_mc_emission_factors.csv'
WERF_CODES_CSV = DATA_DIR / 'el_abbadi' / 'UNIT_PROCESS_EI_CODES_WERF_modified.csv'
LLM_PERDOC_CSV = OUTPUT_DIR / 'unit_processes_by_pdf_llm.csv'
GHG_OUTPUT_DIR = OUTPUT_DIR / 'ghg'

# Pinned upstream commit, so cached files and the treatment-train code can't drift
GHG_COMMIT = '7679ed497df02bfef3438706de0d1fac92d953bd'
GHG_GITHUB = f'https://raw.githubusercontent.com/jiananf2/US_WWTP_GHG/{GHG_COMMIT}'

MG_2_m3 = 3785.412  # m³/MG
kWh_2_MJ = 3.6
KG_PER_DAY_TO_KT_PER_YEAR = 365 / 1e6


def fetch_ghg_file(rel_path: str) -> io.BytesIO:
    """Read a US_WWTP_GHG file as BytesIO, downloading it into GHG_CACHE_DIR the first time."""
    local = GHG_CACHE_DIR / rel_path
    if not local.exists():
        with urllib.request.urlopen(f'{GHG_GITHUB}/{rel_path}', timeout=15) as r:
            data = r.read()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
    return io.BytesIO(local.read_bytes())


# Upstream's treatment_train_werf() (tt_assignment_2022.ipynb), run as-is by assign_treatment_trains
TRAIN_NOTEBOOK = json.load(fetch_ghg_file('treatment_train_assignment/tt_assignment_2022.ipynb'))
TRAIN_SOURCE = next(''.join(cell['source']) for cell in TRAIN_NOTEBOOK['cells']
                    if 'def treatment_train_werf' in ''.join(cell['source']))
TRAIN_NAMESPACE = {}
exec(TRAIN_SOURCE, TRAIN_NAMESPACE)
treatment_train_werf = TRAIN_NAMESPACE['treatment_train_werf']

# What each Tarallo train requires: the check list of each assign() call in treatment_train_werf
# (minus the BIOGAS_EL flag). Lagoons are assigned directly from their unit process.
# Used by the fallback to pick a train compatible with whatever unit processes a facility
# *did* report, rather than the bin's most common train outright.
TT_REQUIRES = {
    call.args[0].value: {e.value for e in call.args[1].elts if isinstance(e, ast.Constant)}
    for call in ast.walk(ast.parse(TRAIN_SOURCE))
    if isinstance(call, ast.Call) and getattr(call.func, 'id', None) == 'assign'
} | {lagoon: set() for lagoon in ('LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'LAGOON_UNCATEGORIZED')}
ALL_TT = sorted(TT_REQUIRES)
# E-train → non-E equivalent (fallback rule 2: biogas electricity is never imputed)
E_TO_BASE = {tt: tt[:-1] for tt in ALL_TT if tt.endswith('E')}

# Trains with no Monte Carlo workbook upstream, verified 404 rather than assumed: N1 is the
# only one of the 50, because El Abbadi's national assignment never produces it (0 of 15,863
# facilities; they find 11 MBR-BNR plants nationally and none with anaerobic digestion). We
# re-derive trains from permit text and can reach it, so N1 facilities emit nothing -- hence
# the runtime warning in calc_ghg rather than a silent zero.
EXPECTED_ABSENT_TT = frozenset({'N1'})

# Every CWNS-based source here describes the 2022 fleet: CWNS 2022 is a 2022 snapshot and
# El Abbadi's assignments are built from it. So the permit source is collapsed to the permits
# in force in 2022 rather than today's, and figure_5 does its own collapse instead of reading
# step6's table, which is deliberately "current" for other consumers. Era matters: on a 2026
# basis the same pipeline reads +22.0% N2O against their published total, on a 2022 basis
# +10.1%, and the total moves from +1.0% to -1.6%.
GHG_AS_OF = '2022-06-01'

# Nitrification is the N2O lever: trains needing one of these carry N2O factors around
# 0.24-0.27, the rest around 0.044 -- a ~6x cliff. Assigning across it by accident is the
# single largest error a fallback can make.
NITRIFYING_UPS = frozenset({'NIT', 'AS_BNR_N', 'AS_BNR_P'})
NITRIFYING_TT = frozenset(tt for tt, req in TT_REQUIRES.items() if req & NITRIFYING_UPS)
LAGOON_TT = frozenset(tt for tt in TT_REQUIRES if tt.startswith('LAGOON'))

# The unit processes El Abbadi key their partial-information fallback on, most specific first.
KEY_UPS = ['AS_BNR_P', 'AS_BNR_N', 'AS-PUREO', 'MBR-BNR', 'TF_ALL', 'MHI', 'FBI', 'LIME',
           'NIT', 'BASIC_AS', 'AND', 'AED', 'PRIMARY']

# Source keys, in plot order. CSV/print order follows SOURCE_NAMES.
PLOT_ORDER = ['cwns_only', 'cwns_external', 'published', 'permits', 'permits_external']
SOURCE_NAMES = {
    'cwns_external': 'CWNS + WEF + DOE Biogas + EPA Nitrification',
    'cwns_only': 'CWNS-only (no nitrification in CWNS)',
    'published': 'CWNS + WEF + DOE + EPA (published)',
    'permits': 'WE3Lab LLM',
    'permits_external': 'WE3Lab LLM + WEF + DOE + EPA',
}
# Labels printed by calc_ghg
CALC_LABELS = {
    'cwns_external': 'Source 1',
    'cwns_only': 'Source 2',
    'published': 'Source manual',
    'permits': 'Source 3',
    'permits_external': 'Source 4',
}
SOURCE_TICK_LABELS = {
    'cwns_external': '\nCWNS +\nWEF + DOE\n+ EPA',
    'cwns_only': '\nCWNS only',
    'published': '\nCWNS +\nWEF + DOE\n+ EPA\n+ corrections',
    'permits': '\nFacility\nPermits',
    'permits_external': '\nFacility\nPermits +\nWEF + DOE\n+ EPA',
}

COMPONENT_PALETTE = [
    '#C94040',  # CH4
    '#9B3080',  # N2O
    '#F2CFA0',  # Bio CO2
    '#D4924A',  # FNG combustion
    '#8B5E2E',  # Biosolids
    '#6E92B0',  # Electricity
    '#B8B8B8',  # NG upstream
]

# TT code → secondary treatment label (what drives N2O EF differences)
NO_NUTRIENT_REMOVAL = 'AS / TF (no nutrient removal)'  # EF ~0.044 — default for any unmapped TT
# G/I/H trains all share the same BNR N2O EF (~0.24) → collapsed to one label
TT_SECONDARY = {
    'E2':  'AS + Nitrification', 'E2P': 'AS + Nitrification',
    'F1':  'AS + Nitrif./Denitrif.', 'F1E': 'AS + Nitrif./Denitrif.',
    'I1':  'AS + BNR', 'I1E': 'AS + BNR',
    'I2':  'AS + BNR', 'I3':  'AS + BNR',
    'I5':  'AS + BNR', 'I6':  'AS + BNR',
    'G1':  'AS + BNR', 'G1E': 'AS + BNR',
    'G2':  'AS + BNR', 'G3':  'AS + BNR',
    'G5':  'AS + BNR', 'G6':  'AS + BNR',
    'H1':  'AS + BNR', 'H1E': 'AS + BNR',
    'N1':  'BNR-MBR', 'N1E': 'BNR-MBR', 'N2': 'BNR-MBR',
    'LAGOON_AER':           'Lagoon',
    'LAGOON_ANAER':         'Lagoon',
    'LAGOON_FAC':           'Lagoon',
    'LAGOON_UNCATEGORIZED': 'Lagoon',
}

SECONDARY_ORDER = [
    NO_NUTRIENT_REMOVAL,
    'AS + Nitrification',
    'AS + Nitrif./Denitrif.',
    'AS + BNR',
    'BNR-MBR',
    'Lagoon',
]

# N2O breakdown: warm sequential from light→dark with lagoon as contrasting sage
N2O_PALETTE = ['#FADA99', '#F5B560', '#E08030', '#C05820', '#8B3410', '#84B07A']
CATEGORY_COLORS = dict(zip(SECONDARY_ORDER, N2O_PALETTE))


@cache
def load_llm_facility_table():
    """Permit-derived processes collapsed per facility, as of GHG_AS_OF."""
    raw = pd.read_csv(LLM_PERDOC_CSV, dtype=str).fillna('').drop(columns=['County'])
    meta = {'Place ID', 'WDID', 'Order_No', 'NPDES No.', 'Agency', 'Facility Name',
            'PDF_File', 'document_order_no', 'Shared_PDF'}
    proc = [c for c in raw.columns if c not in meta]
    content = raw[proc].isin(STATUS_TOKENS).any(axis=1)
    keep = current_permit_mask(raw, as_of=GHG_AS_OF, content=content)
    facilities = collapse_facility_processes(
        raw[keep], key_cols=['Place ID'],
        meta_cols=['WDID', 'Order_No', 'NPDES No.', 'Agency', 'Facility Name', 'County',
                   'PDF_File', 'document_order_no'])
    print(f'  Permit basis {GHG_AS_OF}: {int(keep.sum())} of {len(raw)} documents '
          f'-> {len(facilities)} facilities')
    return facilities


def load_cwns_col_to_werf(automatic=False):
    """Column names from unitprocess_keywords.json → WERF codes (used by Sources 2 and 3).

    automatic=True (experimental): derive leaf -> WERF codes from each leaf's cwns_processes
    (FINAL_UNIT_PROCESS_NAME values) via the El Abbadi WERF CSV, instead of the curated
    JSON werf_codes. Falls back to the curated werf_codes only for leaves with no
    cwns_processes match (ontology/LLM-only concepts with no CWNS name to look up).
    """
    if automatic:
        werf_csv = pd.read_csv(WERF_CODES_CSV, dtype=str)
        csv_lookup = (werf_csv.groupby(werf_csv['FINAL_UNIT_PROCESS_NAME'].str.lower().str.strip())['WERF_CODE']
                      .apply(lambda s: sorted(set(s.dropna()))).to_dict())
    mapping = {}
    for name, details, _ in leaves:
        codes = set()
        if automatic:
            for cwns_name in details.get('cwns_processes') or []:
                codes.update(csv_lookup.get(cwns_name.lower().strip(), []))
        if not codes:
            codes = set(c for c in details.get('werf_codes', []) if c)
        if codes:
            mapping[name] = sorted(codes)
    return mapping


# helpers

def load_mc_ef():
    """50th-percentile emission factors per TT (kg CO2e / m³ wastewater, except elec_50
    which is kWh / MGD as per the original WWTP_GHG_accounting.py usage).

    Reads the cached CSV if present. Otherwise pulls each treatment train's Monte Carlo
    workbook through fetch_ghg_file (skipping EXPECTED_ABSENT_TT, which upstream never published),
    reduces it to its median, and writes the cache -- the workbooks are ~3.6 MB each and only their
    quantiles are ever used.
    """
    if MC_EF_CSV.exists():
        return pd.read_csv(MC_EF_CSV, index_col=0)

    records = {}
    for tt in ALL_TT:
        if tt in EXPECTED_ABSENT_TT:
            continue
        mc = pd.read_excel(fetch_ghg_file(f'{MC_DIR}/{tt}_MC.xlsx'))
        records[tt] = {
            'CH4_50':     mc['CH4'].quantile(0.5),
            'N2O_50':     mc['N2O'].quantile(0.5),
            'NC_CO2_50':  mc['NC_CO2'].quantile(0.5),
            'elec_50':    mc['elec_MC'].quantile(0.5),  # kWh/MGD — no MG_2_m3 needed
            'NG_comb_50': mc['NG_combustion'].quantile(0.5),
            'NG_up_50':   mc['NG_upstream'].quantile(0.5),
            'solids_50':  mc['solids'].quantile(0.5),
        }
    ef = pd.DataFrame(records).T
    ef.to_csv(MC_EF_CSV)
    print(f'  cached {len(ef)} treatment-train emission factors → {MC_EF_CSV.name}')
    return ef


def load_grid_carbon():
    """Series[CWNS_NUM → kg CO2 / kWh], from regional balancing area data."""
    upstream_emissions = {  # kg CO2e / MWh upstream fuel chain (from WWTP_GHG_accounting.py)
        'natural_gas': 24 / 1000 * kWh_2_MJ * 1000,
        'coal':        18 / 1000 * kWh_2_MJ * 1000,
        'nuclear':    1.9 / 1000 * kWh_2_MJ * 1000,
        'wind':       2.86 / 1000 * kWh_2_MJ * 1000,
        'solar':     10.48 / 1000 * kWh_2_MJ * 1000,
        'biomass':   19.02 / 1000 * kWh_2_MJ * 1000,
        'geothermal': 1.35 / 1000 * kWh_2_MJ * 1000,
        'hydro':      2.08 / 1000 * kWh_2_MJ * 1000,
    }
    ba = pd.read_excel(fetch_ghg_file('GHG_accounting/input_data/WWTP_baseline_trains_8.xlsx'),
                       sheet_name='Balance_Area')
    ba['CO2_kg_total'] = (
        ba['co2_gen_mmt'] * 1e9
        + (ba['gas-ct_MWh'] + ba['gas-cc_MWh']) * upstream_emissions['natural_gas']
        + ba['coal_MWh'] * upstream_emissions['coal']
        + ba['nuclear_MWh'] * upstream_emissions['nuclear']
        + (ba['wind-ons_MWh'] + ba['wind-ofs_MWh']) * upstream_emissions['wind']
        + (ba['csp_MWh'] + ba['upv_MWh'] + ba['distpv_MWh'] + ba['o-g-s_MWh']) * upstream_emissions['solar']
        + ba['biomass_MWh'] * upstream_emissions['biomass']
        + ba['geothermal_MWh'] * upstream_emissions['geothermal']
        + (ba['phs_MWh'] + ba['hydro_MWh']) * upstream_emissions['hydro']
    )
    ba['kg_CO2_kWh'] = ba['CO2_kg_total'] / ba['generation'] / 1000
    ba = ba[ba['t'] == 2020].copy()
    ba['r'] = ba['r'].str[1:].astype(int)
    ba_wwtp = pd.read_excel(fetch_ghg_file('GHG_accounting/input_data/WWTP_balancing_area.xlsx'))
    ba_wwtp = ba_wwtp.merge(ba[['r', 'kg_CO2_kWh']], left_on='balancing_area', right_on='r')
    # CWNS_NUMs are stored as integers — zero-pad to 11 digits to match string CWNS_IDs
    ba_wwtp['CWNS_NUM'] = ba_wwtp['CWNS_NUM'].astype(str).str.zfill(11)
    return ba_wwtp.set_index('CWNS_NUM')['kg_CO2_kWh']


def assign_treatment_trains(df):
    """
    Assign Tarallo et al. 2015 treatment train codes from binary WERF-code columns.
    Builds the grouped flags from unit_process_pivot() (tt_assignment_2022.ipynb cell 19), then
    runs upstream's own treatment_train_werf() (cell 20). BIOGAS_EL (the E-train flag) must
    already be set by the caller.
    Input df: one row per facility, binary columns for WERF codes.
    Returns df with added TT columns and TT_IDENTIFIED count.
    """
    d = df.copy()
    needed = ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P', 'AS-PUREO',
              'AS-SA', 'AS-SBR', 'AED', 'AND', 'BDENIT', 'BIO-P', 'BIODRY',
              'BIOGAS_CWNS', 'BNIT', 'BNR', 'BS_LAGOON', 'CHEM-P', 'DISINF-O3',
              'FBI', 'LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC',
              'LAND_TRT', 'LIME', 'MBR-BNR', 'MHI', 'NIT', 'PRIMARY',
              'STBL_POND', 'TF', 'TF-BF', 'TF-RBC']
    for col in needed:
        if col not in d.columns:
            d[col] = 0
    d = d.fillna(0)

    d['SUM_AS'] = d['AS'] + d['AS-A2O'] + d['AS-BDENIT'] + d['AS-EA'] + d['AS-P'] + d['AS-PUREO'] + d['AS-SA']
    d['BASIC_AS'] = ((d['AS'] + d['AS-EA'] + d['AS-SA'] + d['AS-OD'] + d['AS-SBR']) > 0).astype(int)
    d['AS_BNR_N'] = ((d['AS-A2O'] + d['AS-BDENIT'] > 0) | ((d['AS'] > 0) & (d['BNR'] > 0))).astype(int)
    d['TF_ALL'] = ((d['TF'] + d['TF-BF'] + d['TF-RBC']) > 0).astype(int)
    d['AS_BNR_P'] = (((d['SUM_AS'] > 0) & (d['BIO-P'] > 0)) | (d['AS-P'] == 1)).astype(int)
    for col in ('BASIC_AS', 'BDENIT', 'MHI', 'PRIMARY'):
        d.loc[d[col] > 0, col] = 1
    # BIOGAS_EL comes from the caller: WEF/DOE (Sources 1, 4), the AND + cogen proxy
    # (Sources 3, 4), or 0 (Source 2).

    # upstream names the flag BIOGAS_EL_<year> and adds a TT_ASSIGN_NOTE column we don't use
    d['BIOGAS_EL_0'] = d['BIOGAS_EL']
    d = treatment_train_werf(d, 0)
    return d.drop(columns=['BIOGAS_EL_0', 'TT_ASSIGN_NOTE'])


def calc_ghg(wwtp_df, ef, grid_carbon, source_label):
    """
    Compute per-facility GHG emissions.
    wwtp_df must have: CWNS_NUM (str), FLOW_2022_MGD_FINAL (float), TT_IDENTIFIED,
                       and binary TT columns (B1, C1, ..., LAGOON_UNCATEGORIZED).
    Drops facilities with TT_IDENTIFIED < 1.
    Returns wwtp_df with *_med emission columns (kg CO2e / day).
    """
    df = wwtp_df.copy()
    tt_present = [tt for tt in ALL_TT if tt in df.columns]
    df['TT_IDENTIFIED'] = df[tt_present].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    n_before = len(df)
    df = df[df['TT_IDENTIFIED'] >= 1].copy()
    print(f'  {source_label}: {n_before} in → {len(df)} with TT assignment')

    df['FLOW_2022_MGD_FINAL'] = pd.to_numeric(df['FLOW_2022_MGD_FINAL'], errors='coerce').fillna(0)

    # A train with no Monte Carlo workbook upstream contributes nothing at all -- not a wrong
    # number, an invisible facility. N1 (MBR-BNR without biogas recovery) is the only such
    # train: El Abbadi never assign it so never needed its factors, but we re-derive trains
    # from process codes and can reach it. Report the lost flow rather than substituting
    # factors -- the cogeneration credit is ~556 kWh/MGD in absolute terms across the eight
    # measurable E/base pairs, but N1E's demand (6023) is 2.5x any of their base trains, so no
    # scaling of a measured pair is defensible here.
    for tt in [t for t in tt_present if t not in ef.index]:
        flow_lost = (pd.to_numeric(df[tt], errors='coerce').fillna(0)
                     .div(df['TT_IDENTIFIED']).mul(df['FLOW_2022_MGD_FINAL']).sum())
        if flow_lost > 0:
            print(f'    WARNING {tt}: no upstream emission factors; {flow_lost:.1f} MGD '
                  f'contributes zero emissions')

    valid_tt = [tt for tt in tt_present if tt in ef.index]
    tt_mat = df[valid_tt].apply(pd.to_numeric, errors='coerce').fillna(0)
    tt_flow = tt_mat.div(df['TT_IDENTIFIED'], axis=0).mul(df['FLOW_2022_MGD_FINAL'], axis=0)
    ef_sub = ef.loc[valid_tt]

    # direct (kg CO2e / day), factors in kg CO2e / m³
    df['CH4_med']    = (tt_flow @ ef_sub['CH4_50'])   * MG_2_m3
    df['N2O_med']    = (tt_flow @ ef_sub['N2O_50'])   * MG_2_m3
    df['NC_CO2_med'] = (tt_flow @ ef_sub['NC_CO2_50']) * MG_2_m3
    df['NG_comb_med']= (tt_flow @ ef_sub['NG_comb_50']) * MG_2_m3
    df['NG_up_med']  = (tt_flow @ ef_sub['NG_up_50'])   * MG_2_m3
    df['solids_med'] = (tt_flow @ ef_sub['solids_50']) * MG_2_m3

    # electricity: elec_50 is kWh / MGD (no MG_2_m3 conversion)
    df = df.merge(grid_carbon.rename('kg_CO2_kWh'), left_on='CWNS_NUM', right_index=True, how='left')
    ca_avg_ci = grid_carbon[grid_carbon.index.str.startswith('06')].mean()
    df['kg_CO2_kWh'] = df['kg_CO2_kWh'].fillna(ca_avg_ci)
    df['elec_med'] = (tt_flow @ ef_sub['elec_50']) * df['kg_CO2_kWh']

    df['total_med'] = df[['CH4_med', 'N2O_med', 'NC_CO2_med',
                           'NG_comb_med', 'NG_up_med', 'elec_med', 'solids_med']].sum(axis=1)
    return df


def totals_from_df(df):
    """Totals dict (kt CO2e / year) from a df that already has *_med emission columns."""
    totals = {
        'n_facilities':   len(df),
        'total_flow_MGD': df['FLOW_2022_MGD_FINAL'].sum(),
        'CH4_ktyr':       df['CH4_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'N2O_ktyr':       df['N2O_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'NC_CO2_ktyr':    df['NC_CO2_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'NG_comb_ktyr':   df['NG_comb_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'solids_ktyr':    df['solids_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'elec_ktyr':      df['elec_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
        'NG_up_ktyr':     df['NG_up_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR,
    }
    totals['Scope1_ktyr'] = (totals['CH4_ktyr'] + totals['N2O_ktyr'] + totals['NC_CO2_ktyr']
                             + totals['NG_comb_ktyr'] + totals['solids_ktyr'])
    totals['Scope2_ktyr'] = totals['elec_ktyr']
    totals['Scope3_ktyr'] = totals['NG_up_ktyr']
    totals['total_ktyr']  = totals['Scope1_ktyr'] + totals['Scope2_ktyr'] + totals['Scope3_ktyr']
    return totals


# matching helpers

def load_common_place_ids():
    """
    Return the set of Place IDs present in both unit_processes_by_facility_cwns.csv and the
    LLM output — the same intersection used in figure_3_extraction_comparison.py.
    """
    cwns_out = pd.read_csv(CWNS_TABLE_CSV, dtype=str)
    ciwqs = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str)
    llm = load_llm_facility_table()

    cwns_pids = set(ciwqs[ciwqs['CWNS_ID'].isin(cwns_out['CWNS_ID'])]['Place ID'].dropna())
    llm_pids = set(llm['Place ID'].dropna())
    return cwns_pids & llm_pids


def pid_to_cwns_flow(common_pids):
    """
    For each Place ID in common_pids, return a DataFrame with one row per CWNS_NUM
    linking Place ID → CWNS_NUM → FLOW_2022_MGD_FINAL.
    One Place ID can map to multiple CWNS facilities; flow is per CWNS facility.
    """
    ciwqs = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str)
    ciwqs = ciwqs[ciwqs['Place ID'].isin(common_pids)].dropna(subset=['CWNS_ID']).copy()
    ciwqs['CWNS_NUM'] = ciwqs['CWNS_ID'].str.strip()

    tt = pd.read_csv(fetch_ghg_file('GHG_accounting/input_data/tt_assignments_2022.csv'),
                     dtype={'CWNS_NUM': str})
    tt_flow = tt[['CWNS_NUM', 'FLOW_2022_MGD_FINAL']].copy()
    tt_flow['FLOW_2022_MGD_FINAL'] = pd.to_numeric(tt_flow['FLOW_2022_MGD_FINAL'], errors='coerce')

    link = ciwqs[['Place ID', 'CWNS_NUM']].merge(tt_flow, on='CWNS_NUM', how='inner')
    # Drop duplicate CWNS_NUMs (one CWNS facility shouldn't count for multiple Place IDs)
    link = link.drop_duplicates(subset=['CWNS_NUM'])
    return link


# regional fallback

def rank_trains(pool, tt_cols, base_cols):
    """Candidate trains for a pool, most common first, E-trains folded into their base."""
    counts = {}
    for tt in tt_cols:
        n = int(pool[tt].sum())
        if n:
            counts[E_TO_BASE.get(tt, tt)] = counts.get(E_TO_BASE.get(tt, tt), 0) + n
    return [tt for tt, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if tt in base_cols]


def apply_regional_fallback(df):
    """Assign a train to facilities whose reported processes identified none.

    Mirrors the fallback in El Abbadi's tt_assignment_2022.ipynb so both the CWNS-derived and
    the permit-derived sources are built the same way -- they published theirs as
    TT_ASSIGN_NOTE, and 31% of their facilities (17.6% of flow) rely on it, so dropping the
    fallback would not make the comparison cleaner, only inconsistent.

    Three rules taken from their notebook, all missing from the earlier version, which simply
    took the bin's most common train and so ignored what the facility itself reported:

      1. Partial information. They compute a most common train *per key unit process*
         ("Most Common TT (AND)", "(NIT)", ...) and use the one matching a process the
         facility reported. Point Loma (240 MGD, only AND + PRIMARY known) is a fallback case
         in their data too, so this is the rule that decides it -- they land on O1E, while
         picking the bin mode outright landed it on G1E, six times the N2O.
      2. No E-trains. They collapse B1/B1E and friends before taking the mode, so biogas
         electricity is never imputed to a facility with no evidence of it.
      3. Nitrification guard. Their note: "if most common treatment train is a conventional
         activated sludge train, but the nitrification flag is on, override tt_common to nan".
         Applied symmetrically here -- a facility with no nitrification evidence does not get a
         nitrifying train either, which is the direction that actually misfired.

    Region is not a grouping key: every California facility is EPA Region 9, so their
    (size, region) grouping reduces to size alone.
    """
    tt_cols = [tt for tt in ALL_TT if tt in df.columns]
    flow_bins = [0, 2, 4, 7, 16, 46, 100, float('inf')]
    flow_labels = ['<2', '2-4', '4-7', '7-16', '16-46', '46-100', '≥100']
    df = df.copy()
    df['size_bin'] = pd.cut(
        pd.to_numeric(df['FLOW_2022_MGD_FINAL'], errors='coerce').fillna(0),
        bins=flow_bins, labels=flow_labels, right=False)
    assigned = df[df['TT_IDENTIFIED'] >= 1]
    base_cols = [tt for tt in tt_cols if tt not in E_TO_BASE]   # rule 2
    lag_present = assigned[[c for c in LAGOON_TT if c in assigned.columns]].sum(axis=1) > 0
    max_lagoon_flow = (assigned.loc[lag_present, 'FLOW_2022_MGD_FINAL'].max()
                       if lag_present.any() else float('inf'))

    n_fallback = 0
    for idx in df[df['TT_IDENTIFIED'] < 1].index:
        row = df.loc[idx]
        # the published source carries trains only, no unit-process columns
        present = {up for up in KEY_UPS if up in df.columns and row[up] == 1}
        nitrifies = bool(present & NITRIFYING_UPS)
        for pool_mask in [assigned['size_bin'] == row['size_bin'],
                          pd.Series(True, index=assigned.index)]:
            pool = assigned[pool_mask]
            if pool.empty:
                continue
            candidates = rank_trains(pool, tt_cols, base_cols)
            # rule 3 first: never cross the nitrification cliff
            candidates = [tt for tt in candidates if (tt in NITRIFYING_TT) == nitrifies]
            # Lagoons carry the highest N2O factor of any train (0.332) and are small-plant
            # technology. With only a handful of identified facilities in the largest size
            # bins, the bin mode is noise, and it put lagoons on 100-260 MGD plants. Cap
            # lagoon candidacy at the largest flow actually observed on an identified lagoon.
            if row['FLOW_2022_MGD_FINAL'] > max_lagoon_flow:
                candidates = [tt for tt in candidates if tt not in LAGOON_TT]
            # rule 1: prefer a train built on something the facility actually reported
            compatible = [tt for tt in candidates if TT_REQUIRES[tt] & present]
            choice = next(iter(compatible or candidates), None)
            if choice:
                df.at[idx, choice] = 1
                df.at[idx, 'TT_IDENTIFIED'] = 1
                n_fallback += 1
                break
    return df.drop(columns=['size_bin']), n_fallback


# biogas helper

def to_cwns_str(s):
    return str(int(float(s))).zfill(11)


def load_biogas_cwns_set():
    """CWNS_NUMs with confirmed biogas electricity generation from WEF + DOE databases."""
    werf = pd.read_csv(fetch_ghg_file('treatment_train_assignment/input_data/WERF_BIOGAS.csv'))
    elec_cols = ['Electricity_from_combustion-engine', 'Electricity_from_turbine',
                 'Electricity_from_microturbine', 'Electricity_from_fuelcell']
    elec_cols = [c for c in elec_cols if c in werf.columns]
    werf_elec = werf[elec_cols].apply(lambda x: x.str.strip().str.lower() == 'yes').any(axis=1)
    werf_cwns = set(werf.loc[werf_elec, 'CWNS_NUM'].dropna().apply(to_cwns_str))

    doe = pd.read_csv(fetch_ghg_file('treatment_train_assignment/input_data/doe_chpdb-WWTP.csv'))
    doe_cwns = set(doe.loc[doe['BIOGAS_DOE_2022'] == 1, 'CWNS_NUM'].dropna().apply(to_cwns_str))

    combined = werf_cwns | doe_cwns
    print(f'  Biogas set: {len(werf_cwns)} WEF + {len(doe_cwns)} DOE = {len(combined)} unique CWNS facilities')
    return combined


def load_epa_nitrification_set():
    """CWNS_NUMs whose nitrification comes from the EPA PRES_AMMONIA_REMOVAL (2024) field.

    A third external database alongside WEF and DOE, not a manual correction. El Abbadi merge
    it in their assignment notebook and record it in UP_ID_NOTE; the raw EPA field is not in
    the CWNS 2022 release, so the flag is read back from their published assignments.

    It matters far more than its 25 California facilities suggest: CWNS's own unit-process
    data reports nitrification for *none* of the 318 common facilities, and these 25 carry
    220.8 of the 556.7 kt CO2e/yr N2O in El Abbadi's published California total -- 40%. Without it
    the CWNS baseline has no nitrogen treatment at all, which is not a fair comparison for a
    permit-derived source that finds it at 114 facilities.
    """
    tt = pd.read_csv(fetch_ghg_file('treatment_train_assignment/output_data/tt_assignments_2022.csv'),
                     low_memory=False)
    key = tt['CWNS_NUM'].astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(11)
    from_epa = tt['UP_ID_NOTE'].fillna('').str.contains('AMMONIA', case=False)
    out = set(key[from_epa])
    print(f'  EPA nitrification set: {len(out)} CWNS facilities (PRES_AMMONIA_REMOVAL 2024)')
    return out


# WERF pivot (shared by CWNS and LLM sources)

def build_werf_pivot(facility_rows, col_to_werf):
    """
    Build a binary WERF-code pivot (one row per Place ID) from facility status columns.
    PRESENT_STATUSES selects which status strings count as present, for every source, so
    the comparison is like-for-like.
    col_to_werf is the curated mapping or, for the automatic-derivation experiment on the
    CWNS-based source, the CSV-derived one (see load_cwns_col_to_werf).
    Adds derived NIT and BIO-P indicators, then clears lagoon codes for any facility that
    also has secondary treatment. Returns (pivot, n_cleared).
    """
    status_cols = [c for c in facility_rows.columns if c in col_to_werf]
    records = []
    for _, row in facility_rows.iterrows():
        werf_present = set()
        for col in status_cols:
            if str(row[col]).strip().upper() in PRESENT_STATUSES:
                werf_present.update(col_to_werf[col])
        record = {'Place ID': str(row['Place ID']).strip()}
        for code in werf_present:
            record[code] = 1
        records.append(record)

    pivot = pd.DataFrame(records).fillna(0)

    nit_indicator = ['BNIT', 'BNR', 'AS-BDENIT', 'AS-A2O']
    pivot['NIT'] = (pivot[[c for c in nit_indicator if c in pivot.columns]]
                    .sum(axis=1) > 0).astype(int)
    # A2O is an anaerobic-anoxic-oxic configuration, i.e. biological P removal by design.
    # BNR is not: our "Unspecified Nutrient Removal" leaf maps to it and is PRESENT at 140
    # facilities, so including it credited 103 of 318 CA plants with biological phosphorus
    # removal where El Abbadi's CWNS data has 1 -- most of California has no P limit. Those
    # spurious BIO-P flags became AS_BNR_P and pushed plants into the G-series, whose N2O
    # factors are ~6x the non-nitrifying trains. BNR still derives NIT below, which it does
    # imply.
    bio_p_indicator = ['AS-A2O']
    pivot['BIO-P'] = (pivot[[c for c in bio_p_indicator if c in pivot.columns]]
                      .sum(axis=1) > 0).astype(int)

    secondary_werf = ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P',
                      'AS-PUREO', 'AS-SA', 'AS-SBR', 'TF', 'TF-BF', 'TF-RBC', 'MBR-BNR']
    lagoon_werf = ['LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'STBL_POND']
    has_secondary = pivot[[c for c in secondary_werf if c in pivot.columns]].sum(axis=1) > 0
    for col in lagoon_werf:
        if col in pivot.columns:
            pivot.loc[has_secondary, col] = 0
    return pivot, int(has_secondary.sum())


# Sources 1 & 2: El Abbadi (CWNS-based)

def load_cwns_source(common_pids, col_to_werf, include_manual=True):
    """
    Load El Abbadi CWNS-based treatment train data.

    include_manual=True  (Source 1): re-derives TT assignments as below, then adds the
        external databases El Abbadi use: DOE/WEF biogas E-train upgrades and EPA
        PRES_AMMONIA_REMOVAL nitrification.

    include_manual=False (Source 2): re-derives TT assignments from
        unit_processes_by_facility_cwns.csv using our pipeline (same as Source 3),
        with BIOGAS_EL=0. No manual corrections of any kind.

    col_to_werf: the curated mapping, or the experimental automatic one (derived live
        from the El Abbadi WERF CSV via each leaf's cwns_processes), to test
        comparability against El Abbadi's own methodology.
    """
    link = pid_to_cwns_flow(common_pids)

    if include_manual:
        biogas_cwns = load_biogas_cwns_set()

    # Re-derive from CWNS unit processes (both branches)
    ca_cwns = pd.read_csv(CWNS_TABLE_CSV, dtype=str)
    cwns_by_pid, _ = build_cwns_facility_processes(ca_cwns, target_facilities=common_pids)

    cwns_pivot, _ = build_werf_pivot(cwns_by_pid, col_to_werf)
    cwns_pivot['BIOGAS_EL'] = 0  # set after merge once CWNS_NUM is known

    merged = link.merge(cwns_pivot, on='Place ID', how='inner')

    if include_manual:
        # EPA PRES_AMMONIA_REMOVAL is the third external database, on the same footing as WEF
        # and DOE. CWNS unit-process data reports nitrification for none of these facilities,
        # so without it this source has no nitrogen treatment at all.
        epa_nit = merged['CWNS_NUM'].isin(load_epa_nitrification_set())
        merged.loc[epa_nit, 'NIT'] = 1
        print(f'  EPA nitrification applied to {int(epa_nit.sum())} of {len(merged)} facilities')
        wef_doe_mask = merged['CWNS_NUM'].isin(biogas_cwns)
        merged['BIOGAS_EL'] = wef_doe_mask.astype(int)
        merged['AND'] = ((merged['AND'] > 0) | wef_doe_mask).astype(int)

    df = assign_treatment_trains(merged)
    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON', 'STBL_POND']].max(axis=1)

    df, n_fallback = apply_regional_fallback(df)
    if include_manual:
        print(f'Source 1 (CWNS + WEF + DOE + EPA): {len(df)} CWNS rows, '
              f'{int((df["TT_IDENTIFIED"] >= 1).sum())} with TT assignment '
              f'({n_fallback} via regional fallback)')
    else:
        print(f'Source 2 (CWNS-only, re-derived): {len(df)} CWNS rows, '
              f'{int((df["TT_IDENTIFIED"] >= 1).sum())} with TT assignment '
              f'({n_fallback} via regional fallback)')
    return df


# Manual: El Abbadi & Feng published assignments

def load_source_manual(common_pids):
    """
    El Abbadi & Feng's published treatment-train assignments, read directly from
    tt_assignments_2022.csv. Includes CWNS unit processes, DOE/WEF biogas E-train
    upgrades, manual lagoon corrections, and manual TT overrides for large plants.
    One row per CWNS_NUM. TTs are pre-assigned, so we do not re-derive them.
    """
    link = pid_to_cwns_flow(common_pids)

    tt = pd.read_csv(fetch_ghg_file('GHG_accounting/input_data/tt_assignments_2022.csv'),
                     dtype={'CWNS_NUM': str})
    tt['CWNS_NUM'] = tt['CWNS_NUM'].str.strip()
    tt_cols = [c for c in ALL_TT if c in tt.columns]
    extra = ['LAGOON_OTHER', 'STBL_POND']
    tt = tt[['CWNS_NUM'] + tt_cols + extra].drop_duplicates(subset=['CWNS_NUM'])

    df = link.merge(tt, on='CWNS_NUM', how='inner')
    for c in tt_cols + extra:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON_OTHER', 'STBL_POND']].max(axis=1)
    tt_present = [c for c in ALL_TT if c in df.columns]
    df['TT_IDENTIFIED'] = df[tt_present].sum(axis=1)

    df, n_fallback = apply_regional_fallback(df)
    print(f'Source manual (CWNS + WEF + DOE biogas + manual corrections): {len(df)} CWNS rows, '
          f'{int((df["TT_IDENTIFIED"] >= 1).sum())} with TT assignment '
          f'({n_fallback} via regional fallback)')
    return df


# Sources 3 & 4: LLM permit extraction

def load_llm_source(common_pids, col_to_werf, add_wef_biogas=False):
    """
    LLM-extracted unit processes collapsed from unit_processes_by_pdf_llm.csv (as of GHG_AS_OF),
    mapped to WERF codes via unitprocess_keywords.json (same mapping as the CWNS sources).
    One row per CWNS_NUM.

    add_wef_biogas=False (Source 3): BIOGAS_EL comes from the LLM AND/cogen proxy only.
    add_wef_biogas=True  (Source 4): additionally flags biogas electricity for any facility
        the WEF/DOE databases confirm, testing whether that resolves the FNG discrepancy.
    """
    llm = load_llm_facility_table()
    llm = llm[llm['Place ID'].isin(common_pids)].copy()
    # PRESENT_STATUSES, matching the CWNS sources. Treatment trains describe the plant as it
    # runs today, so a planned process must not reassign it -- Hyperion's 2035 recycled-water
    # MBR was putting a 215 MGD high-purity-oxygen plant into the N (membrane) series, and N1
    # is the one train of 50 with no emission factors, so its emissions silently went to zero.
    # The old {'PRESENT', 'FUTURE'} set also omitted PRESENT_AND_FUTURE, dropping processes
    # that are both in service and being expanded.
    llm_pivot, n_cleared = build_werf_pivot(llm, col_to_werf)
    if n_cleared:
        print(f'  Cleared lagoon WERF codes for {n_cleared} facilities with secondary treatment')

    link = pid_to_cwns_flow(common_pids)
    merged = link.merge(llm_pivot, on='Place ID', how='inner')

    and_col = merged['AND']
    cogen_col = merged['BIOGAS_CWNS']
    # Digestion alone is not energy recovery -- biogas is often flared. El Abbadi treat recovery
    # as a small subset of digestion (315 of 2644 AD plants, 11.9%, in tt_assignments_2022.csv),
    # so an OR here promoted every digesting plant to an E-train, ~8x their share. Gas
    # utilisation/cogeneration is the signal; digestion is a precondition, not evidence.
    llm_biogas = (cogen_col > 0) & (and_col > 0)
    if add_wef_biogas:
        # LLM data takes priority; WEF only adds where LLM was silent.
        # OR logic: keep all LLM-detected AND/BIOGAS_EL, additionally set both for any
        # facility the WEF database confirms biogas utilization (implying a digester exists).
        epa_nit = merged['CWNS_NUM'].isin(load_epa_nitrification_set())
        merged.loc[epa_nit, 'NIT'] = 1
        wef_mask = merged['CWNS_NUM'].isin(load_biogas_cwns_set())
        merged['BIOGAS_EL'] = (llm_biogas | wef_mask).astype(int)
        merged['AND'] = ((and_col > 0) | wef_mask).astype(int)
        n_wef_added = int((wef_mask & ~llm_biogas).sum())
        print(f'  Source 4: WEF biogas data adds {n_wef_added} facilities not already detected by LLM')
    else:
        merged['BIOGAS_EL'] = llm_biogas.astype(int)

    df = assign_treatment_trains(merged)
    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON', 'STBL_POND']].max(axis=1)
    df, n_fallback = apply_regional_fallback(df)
    label = 'Source 4 (LLM + WEF/DOE biogas + EPA nitrification)' if add_wef_biogas else 'Source 3 (WE3Lab LLM)'
    print(f'{label}: {len(df)} CWNS rows, '
          f'{int((df["TT_IDENTIFIED"] >= 1).sum())} with TT assignment '
          f'({n_fallback} via regional fallback)')
    return df


# comparison plot

def plot_comparison(results):
    components = {
        'CH₄ (Scope 1)':            ('CH4_ktyr',     COMPONENT_PALETTE[0]),
        'N₂O (Scope 1)':            ('N2O_ktyr',     COMPONENT_PALETTE[1]),
        'Bio CO₂ (Scope 1)':        ('NC_CO2_ktyr',  COMPONENT_PALETTE[2]),
        'FNG combustion (Scope 1)': ('NG_comb_ktyr', COMPONENT_PALETTE[3]),
        'Biosolids (Scope 1)':      ('solids_ktyr',  COMPONENT_PALETTE[4]),
        'Electricity (Scope 2)':    ('elec_ktyr',    COMPONENT_PALETTE[5]),
        'NG upstream (Scope 3)':    ('NG_up_ktyr',   COMPONENT_PALETTE[6]),
    }

    x = np.arange(len(PLOT_ORDER))
    width = 0.5
    fig, ax = plt.subplots(figsize=(6, 5))
    bottoms = np.zeros(len(PLOT_ORDER))
    for label, (key, color) in components.items():
        vals = np.array([results[s][key] for s in PLOT_ORDER])
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=color, edgecolor='black', linewidth=0.4)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_TICK_LABELS[s] for s in PLOT_ORDER], fontsize=10)
    first = results[PLOT_ORDER[0]]
    ax.set_ylabel(f'kt CO₂e / year\nn={first["n_facilities"]} facilities, {first["total_flow_MGD"]:.0f} MGD', fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=10, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save_and_close(fig, FINAL_DIR / 'figure_5', dpi=300)


# N2O breakdown by secondary treatment

def n2o_by_secondary(df, ef, label):
    """Return dict[secondary_label → N2O kt CO2e/yr] for one source dataframe."""
    valid_tt = [tt for tt in ALL_TT if tt in df.columns and tt in ef.index]
    tt_mat = df[valid_tt].apply(pd.to_numeric, errors='coerce').fillna(0)
    tt_flow = tt_mat.div(df['TT_IDENTIFIED'], axis=0).mul(df['FLOW_2022_MGD_FINAL'], axis=0)
    groups = {}
    tt_n2o = {}
    for tt in valid_tt:
        n2o = (tt_flow[tt] * ef.loc[tt, 'N2O_50'] * MG_2_m3 * KG_PER_DAY_TO_KT_PER_YEAR).sum()
        tt_n2o[tt] = n2o
        secondary = TT_SECONDARY.get(tt, NO_NUTRIENT_REMOVAL)
        groups[secondary] = groups.get(secondary, 0) + n2o
    dup_cwns = df['CWNS_NUM'].duplicated().sum()
    top = sorted(tt_n2o.items(), key=lambda x: -x[1])[:8]
    top_str = ', '.join(f'{t}={v:.1f}' for t, v in top if v > 0)
    print(f'  [{label}] n={len(df)} rows, {dup_cwns} dup CWNS_NUMs, '
          f'total N2O={sum(groups.values()):.1f} kt/yr | top TTs: {top_str}')
    return groups


def plot_n2o_breakdown(dfs, ef):
    print('N2O breakdown diagnostics:')
    breakdowns = {s: n2o_by_secondary(df, ef, label=s) for s, df in dfs.items()}
    present_cats = [c for c in SECONDARY_ORDER
                    if any(breakdowns[s].get(c, 0) > 0 for s in PLOT_ORDER)]

    first_df = list(dfs.values())[0]
    n_fac = len(first_df)
    tot_flow = first_df['FLOW_2022_MGD_FINAL'].sum()
    ylabel = f'N₂O  kt CO₂e / year\nn={n_fac} facilities, {tot_flow:.0f} MGD'
    x = np.arange(len(PLOT_ORDER))
    width = 0.45

    fig, ax = plt.subplots(figsize=(8.6, 5))
    bottoms = np.zeros(len(PLOT_ORDER))
    for cat in present_cats:
        vals = np.array([breakdowns[s].get(cat, 0) for s in PLOT_ORDER])
        ax.bar(x, vals, width, bottom=bottoms,
               color=CATEGORY_COLORS[cat], edgecolor='black', linewidth=0.4)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_TICK_LABELS[s] for s in PLOT_ORDER], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    # Scale to the tallest STACK, not the largest single segment: taking the max over
    # individual categories left one tick at 1000 with bars topping out near 900.
    y_top = max(sum(breakdowns[s].get(c, 0) for c in present_cats) for s in PLOT_ORDER)
    step = next(s for s in (25, 50, 100, 200, 250, 500, 1000, 2000) if y_top / s <= 6)
    ax.set_yticks(np.arange(0, np.ceil(y_top / step) * step + step / 2, step))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cat_handles = [
        Patch(facecolor=CATEGORY_COLORS[cat], edgecolor='black', linewidth=0.5, label=cat)
        for cat in present_cats
    ]
    make_grouped_legend(ax, [{'header': 'Treatment Type', 'items': list(reversed(cat_handles))}],
                        bbox_to_anchor=(1.01, 1), fontsize=10)
    fig.tight_layout()
    fig.savefig(GHG_OUTPUT_DIR / 'ca_n2o_by_secondary_color.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


# main

if __name__ == '__main__':
    print('Loading MC emission factors...')
    ef = load_mc_ef()
    print(f'  Loaded EFs for {len(ef)} treatment trains')

    print('Loading grid carbon intensity...')
    grid_carbon = load_grid_carbon()

    print('\nBuilding common facility set (Place ID intersection: CWNS ∩ LLM)...')
    common_pids = load_common_place_ids()
    print(f'  {len(common_pids)} common Place IDs')

    werf_curated = load_cwns_col_to_werf()
    werf_automatic = load_cwns_col_to_werf(automatic=True)
    raw = {
        'cwns_external': load_cwns_source(common_pids, werf_curated, include_manual=True),
        'cwns_only': load_cwns_source(common_pids, werf_curated, include_manual=False),
        'published': load_source_manual(common_pids),
        'permits': load_llm_source(common_pids, werf_curated),
        'permits_external': load_llm_source(common_pids, werf_curated, add_wef_biogas=True),
    }

    # 1:1 matching: filter the other sources to the same CWNS_NUMs that Source 3 assigned
    assigned_cwns = set(raw['permits'].loc[raw['permits']['TT_IDENTIFIED'] >= 1, 'CWNS_NUM'])
    print(f'\nFiltering CWNS-based sources to the {len(assigned_cwns)} CWNS facilities '
          f'where LLM assigned a TT...')
    matched = {key: df if key == 'permits' else df[df['CWNS_NUM'].isin(assigned_cwns)]
               for key, df in raw.items()}

    print('\n── GHG calculation ──')
    ghg = {key: calc_ghg(df, ef, grid_carbon, CALC_LABELS[key]) for key, df in matched.items()}

    # Enforce intersection across all sources
    cwns_sets = {key: set(df['CWNS_NUM']) for key, df in ghg.items()}
    common_cwns = set.intersection(*cwns_sets.values())
    print(f'\n── intersection ──')
    print(f'  S1: {len(cwns_sets["cwns_external"])} | S2: {len(cwns_sets["cwns_only"])} '
          f'| Manual: {len(cwns_sets["published"])} '
          f'| S3: {len(cwns_sets["permits"])} | S4: {len(cwns_sets["permits_external"])}')
    print(f'  Final intersection: {len(common_cwns)} CWNS facilities')
    ghg = {key: df[df['CWNS_NUM'].isin(common_cwns)].copy() for key, df in ghg.items()}
    results = {key: totals_from_df(df) for key, df in ghg.items()}

    print('GHG Emissions Source Comparison (for matched facilities)')
    rows = []
    for s, name in SOURCE_NAMES.items():
        r = results[s]
        flow = r['total_flow_MGD']
        rows.append({
            'Source': name,
            # Where each source's treatment trains come from. Only the published source imports El
            # Abbadi & Feng's published assignments; every other source re-derives the train
            # from unit-process/WERF codes, which is the only path new process data can affect.
            'TT assignment': 'published' if s == 'published' else 're-derived',
            'N CWNS': r['n_facilities'],
            'Flow (MGD)': f"{flow:.0f}",
            'CH4 (kt/yr)': f"{r['CH4_ktyr']:.1f}",
            'N2O (kt/yr)': f"{r['N2O_ktyr']:.1f}",
            'NC_CO2 (kt/yr)': f"{r['NC_CO2_ktyr']:.1f}",
            'NG comb (kt/yr)': f"{r['NG_comb_ktyr']:.1f}",
            'Biosolids (kt/yr)': f"{r['solids_ktyr']:.1f}",
            'Elec (kt/yr)': f"{r['elec_ktyr']:.1f}",
            'NG up (kt/yr)': f"{r['NG_up_ktyr']:.1f}",
            'Scope 1': f"{r['Scope1_ktyr']:.1f}",
            'Scope 2': f"{r['Scope2_ktyr']:.1f}",
            'Scope 3': f"{r['Scope3_ktyr']:.1f}",
            'TOTAL (kt/yr)': f"{r['total_ktyr']:.1f}",
            'per MGD': f"{r['total_ktyr']/flow:.3f}" if flow > 0 else 'N/A',
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(GHG_OUTPUT_DIR / 'ca_ghg_summary.csv', index=False)

    for baseline_key, heading, comparisons in [
        ('cwns_only', 'CWNS-only', [('cwns_external', 'El Abbadi (all sources)'),
                                    ('permits', 'WE3Lab LLM'),
                                    ('permits_external', 'WE3Lab LLM + WEF Biogas')]),
        ('cwns_external', 'CWNS + WEF + DOE Biogas', [('permits', 'WE3Lab LLM'),
                                                      ('permits_external', 'WE3Lab LLM + WEF Biogas')]),
    ]:
        base = results[baseline_key]
        print(f'\n── % change relative to El Abbadi ({heading}) ──')
        for key, name in comparisons:
            r = results[key]
            n2o_pct = (r['N2O_ktyr'] - base['N2O_ktyr']) / base['N2O_ktyr'] * 100
            fng_pct = (r['NG_comb_ktyr'] - base['NG_comb_ktyr']) / base['NG_comb_ktyr'] * 100
            tot_pct = (r['total_ktyr'] - base['total_ktyr']) / base['total_ktyr'] * 100
            print(f'  {name}: N2O {n2o_pct:+.1f}%,  FNG {fng_pct:+.1f}%,  Total {tot_pct:+.1f}%')

    print('\n── EXPERIMENT: automatic WERF mapping (CSV-derived) vs curated JSON werf_codes ──')
    print('  (CWNS-based sources only, both compared against Source manual = El Abbadi & Feng published)')
    automatic_raw = {
        'cwns_external': load_cwns_source(common_pids, werf_automatic, include_manual=True),
        'cwns_only': load_cwns_source(common_pids, werf_automatic, include_manual=False),
    }
    automatic_results = {}
    for key, df in automatic_raw.items():
        auto_ghg = calc_ghg(df[df['CWNS_NUM'].isin(assigned_cwns)], ef, grid_carbon,
                            f'{CALC_LABELS[key]} (auto WERF)')
        automatic_results[key] = totals_from_df(auto_ghg[auto_ghg['CWNS_NUM'].isin(common_cwns)])

    published = results['published']
    for label, r in [('Source 1, curated werf_codes', results['cwns_external']),
                     ('Source 1, automatic WERF csv', automatic_results['cwns_external']),
                     ('Source 2, curated werf_codes', results['cwns_only']),
                     ('Source 2, automatic WERF csv', automatic_results['cwns_only'])]:
        n2o_d = (r['N2O_ktyr'] - published['N2O_ktyr']) / published['N2O_ktyr'] * 100
        tot_d = (r['total_ktyr'] - published['total_ktyr']) / published['total_ktyr'] * 100
        print(f'  {label}: N2O {n2o_d:+.1f}% vs manual,  Total {tot_d:+.1f}% vs manual  '
              f'(Total={r["total_ktyr"]:.1f}, N2O={r["N2O_ktyr"]:.1f})')

    plot_comparison(results)
    plot_n2o_breakdown(ghg, ef)
