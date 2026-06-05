# Adapted from https://github.com/jiananf2/US_WWTP_GHG/tree/main/GHG_accounting/WWTP_GHG_accounting.py
# by Abigayle Hodson, Abigayle_Hodson@lbl.gov
# Publication: https://eartharxiv.org/repository/view/7980/
# Modified by WE3Lab: California-specific comparison across three treatment process data sources.
#
#   Common set = Place IDs present in BOTH unit_processes_by_facility_cwns.csv AND llm output.
#
#   Source 1 - El Abbadi all-sources: tt_assignments_2022.csv for matched CA facilities.
#   Source 2 - El Abbadi CWNS-only: Source 1 without non-CWNS supplemental data
#   Source 3 - unit processes from unit_processes_by_facility_llm.csv,
#
# NOTES:
#   - Biosolids emissions use per-TT 50th-percentile from MC distributions (not
#     facility-specific EPA biosolids data), consistent across all three sources.
#   - Electricity uses facility-specific grid carbon intensity (WWTP balancing area).
#   - Biogas electricity (E-train variants) is present in Source 1 only.
#   - TT assignment fallback uses El Abbadi's EPA region × flow-size bins (all CA = Region 9,
#     so only flow-size bins differentiate: <2,2-4,4-7,7-16,16-46,46-100,≥100 MGD).
#   - Fallback is NOT applied to Source 3; facilities without a TT are simply excluded.

import io
import json
import sys
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'wwtp_process_extraction'))
from helpers.plotting import COLORS
from helpers.utils import build_cwns_facility_processes
GHG_ROOT = Path('/home/daly/git/US_WWTP_GHG')  # local fallback
MC_DIR = GHG_ROOT / 'uncertainty_sensitivity_results/Monte_Carlo'  # MC files stay local
GHG_INPUT = GHG_ROOT / 'GHG_accounting/input_data'
OUTPUT_DIR = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

GHG_GITHUB = 'https://raw.githubusercontent.com/jiananf2/US_WWTP_GHG/main'


def _fetch(rel_path: str) -> io.BytesIO:
    """Fetch a file as BytesIO: GitHub first, local GHG_ROOT fallback."""
    url = f'{GHG_GITHUB}/{rel_path}'
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return io.BytesIO(r.read())
    except Exception:
        local = GHG_ROOT / rel_path
        if local.exists():
            return io.BytesIO(local.read_bytes())
        raise FileNotFoundError(f'Could not fetch {rel_path} from GitHub or {local}')

MG_2_m3 = 3785.412  # m³/MG
kWh_2_MJ = 3.6

# Treatment train crosswalk: internal code → Tarallo et al. 2015 publication code
# From WWTP_GHG_accounting.py lines 71-120
crosswalk = {
    'B1':'*A1', 'B1E':'*A1e', 'B2':'*A3', 'B3':'*A4', 'B4':'*A2',
    'B5':'*A5', 'B6':'*A6', 'C1':'A1', 'C1E':'A1e', 'C2':'A3',
    'C3':'A4', 'C5':'A5', 'C6':'A6', 'D1':'*C1', 'D1E':'*C1e',
    'D2':'*C3', 'D3':'*C4', 'D5':'*C5', 'D6':'*C6', 'E2':'E3',
    'E2P':'*E3', 'F1':'*E1', 'F1E':'*E1e', 'G1':'*G1', 'G1E':'*G1e',
    'G2':'*G3', 'G3':'*G4', 'G5':'*G5', 'G6':'*G6',
    'H1':'*G1-p', 'H1E':'*G1e-p', 'I1':'F1', 'I1E':'F1e',
    'I2':'F3', 'I3':'F4', 'I5':'F5', 'I6':'F6',
    'LAGOON_AER':'L-a', 'LAGOON_ANAER':'L-n', 'LAGOON_FAC':'L-f',
    'LAGOON_UNCATEGORIZED':'L-u', 'N1':'*D1', 'N1E':'*D1e',
    'N2':'*D3', 'O1':'*B1', 'O1E':'*B1e', 'O2':'*B3',
    'O3':'*B4', 'O5':'*B5', 'O6':'*B6',
}
ALL_TT = list(crosswalk.keys())

# E-train → non-E equivalent (for Source 2: strip biogas-derived E assignments)
E_TO_BASE = {
    'B1E': 'B1', 'C1E': 'C1', 'D1E': 'D1', 'F1E': 'F1',
    'G1E': 'G1', 'H1E': 'H1', 'I1E': 'I1', 'N1E': 'N1', 'O1E': 'O1',
}

ONTOLOGY_TXT = REPO_ROOT / 'wwtp_process_extraction/data/llm_extraction/input/ontology.txt'
ONTOLOGY_JSON_DIR = REPO_ROOT / 'wwtp_process_extraction/output/2026-5-25/llm_postprocess_ontology'


def build_ontology_graph():
    """Parse ontology.txt IS-A graph into three dicts for ancestry traversal."""
    proc_parents = {}
    equip_parents = {}
    equip_procs = {}  # equipment → processes it performs
    section = None
    with open(ONTOLOGY_TXT) as f:
        for line in f:
            line = line.strip()
            if line == 'EQUIPMENTS:':
                section = 'equip'
            elif line == 'PROCESSES:':
                section = 'proc'
            elif not line or line.startswith('#'):
                continue
            elif section == 'proc':
                parts = [p.strip() for p in line.split('|')]
                name = parts[0]
                for part in parts[1:]:
                    if part.startswith('SubProcess:'):
                        parents = [p.strip() for p in part[len('SubProcess:'):].split(',')]
                        proc_parents[name] = parents
            elif section == 'equip':
                parts = [p.strip() for p in line.split('|')]
                name = parts[0]
                for part in parts[1:]:
                    if part.startswith('SubEquip:'):
                        parents = [p.strip() for p in part[len('SubEquip:'):].split(',')]
                        equip_parents[name] = parents
                    elif part.startswith('Process:'):
                        procs = [p.replace('Process-', '').strip()
                                 for p in part[len('Process:'):].split(',')]
                        equip_procs[name] = procs
    return proc_parents, equip_parents, equip_procs


PROC_PARENTS, EQUIP_PARENTS, EQUIP_PROCS = build_ontology_graph()


def load_ontology_to_werf():
    """Load ontology_to_werf.csv and invert to {ontology_name: [werf_code, ...]}."""
    csv_path = SCRIPT_DIR / 'ontology_to_werf.csv'
    df = pd.read_csv(csv_path)
    name_to_werf = {}
    for _, row in df.iterrows():
        werf = row['werf_code']
        for name in str(row['ontology_names']).split(';'):
            name = name.strip()
            if name:
                name_to_werf.setdefault(name, []).append(werf)
    return name_to_werf


ONTOLOGY_TO_WERF = load_ontology_to_werf()

# CWNS column names (from unitprocess_keywords.json) → WERF codes.
# Used by load_cwns_source (Source 2); Source 3 uses the ontology JSON path instead.
def load_cwns_col_to_werf():
    kw_path = REPO_ROOT / 'wwtp_process_extraction/data/unitprocess_keywords.json'
    with open(kw_path) as f:
        keywords = json.load(f)
    mapping = {}
    def walk(node):
        if not isinstance(node, dict):
            return
        for name, details in node.items():
            if isinstance(details, dict) and 'alt_names' in details:
                codes = [c for c in details.get('werf_codes', []) if c]
                if codes:
                    mapping[name] = codes
            else:
                walk(details)
    walk(keywords)
    return mapping

CWNS_COL_TO_WERF = load_cwns_col_to_werf()

PRESENT_STATUSES = {'PRESENT', 'PRESENT_AND_FUTURE'}


def get_ontology_ancestors(name):
    """BFS over ontology IS-A graph; returns all ancestor process/equipment names."""
    seen = set()
    queue = [name]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(PROC_PARENTS.get(current, []))
        queue.extend(EQUIP_PARENTS.get(current, []))
        queue.extend(EQUIP_PROCS.get(current, []))
    return seen


# ── helpers ───────────────────────────────────────────────────────────────────

def load_mc_ef():
    """50th-percentile emission factors per TT (kg CO2e / m³ wastewater, except elec_50
    which is kWh / MGD as per the original WWTP_GHG_accounting.py usage)."""
    records = {}
    for tt in ALL_TT:
        mc_file = MC_DIR / f'{tt}_MC.xlsx'
        if not mc_file.exists():
            continue
        mc = pd.read_excel(mc_file)
        records[tt] = {
            'CH4_50':     mc['CH4'].quantile(0.5),
            'N2O_50':     mc['N2O'].quantile(0.5),
            'NC_CO2_50':  mc['NC_CO2'].quantile(0.5),
            'elec_50':    mc['elec_MC'].quantile(0.5),  # kWh/MGD — no MG_2_m3 needed
            'NG_comb_50': mc['NG_combustion'].quantile(0.5),
            'NG_up_50':   mc['NG_upstream'].quantile(0.5),
            'solids_50':  mc['solids'].quantile(0.5),
        }
    return pd.DataFrame(records).T


def load_grid_carbon():
    """Series[CWNS_NUM → kg CO2 / kWh], from regional balancing area data."""
    UEG = {  # kg CO2e / MWh upstream fuel chain (from WWTP_GHG_accounting.py)
        'natural_gas': 24 / 1000 * kWh_2_MJ * 1000,
        'coal':        18 / 1000 * kWh_2_MJ * 1000,
        'nuclear':    1.9 / 1000 * kWh_2_MJ * 1000,
        'wind':       2.86 / 1000 * kWh_2_MJ * 1000,
        'solar':     10.48 / 1000 * kWh_2_MJ * 1000,
        'biomass':   19.02 / 1000 * kWh_2_MJ * 1000,
        'geothermal': 1.35 / 1000 * kWh_2_MJ * 1000,
        'hydro':      2.08 / 1000 * kWh_2_MJ * 1000,
    }
    ba = pd.read_excel(_fetch('GHG_accounting/input_data/WWTP_baseline_trains_8.xlsx'),
                       sheet_name='Balance_Area')
    ba['CO2_kg_total'] = (
        ba['co2_gen_mmt'] * 1e9
        + (ba['gas-ct_MWh'] + ba['gas-cc_MWh']) * UEG['natural_gas']
        + ba['coal_MWh'] * UEG['coal']
        + ba['nuclear_MWh'] * UEG['nuclear']
        + (ba['wind-ons_MWh'] + ba['wind-ofs_MWh']) * UEG['wind']
        + (ba['csp_MWh'] + ba['upv_MWh'] + ba['distpv_MWh'] + ba['o-g-s_MWh']) * UEG['solar']
        + ba['biomass_MWh'] * UEG['biomass']
        + ba['geothermal_MWh'] * UEG['geothermal']
        + (ba['phs_MWh'] + ba['hydro_MWh']) * UEG['hydro']
    )
    ba['kg_CO2_kWh'] = ba['CO2_kg_total'] / ba['generation'] / 1000
    ba = ba[ba['t'] == 2020].copy()
    ba['r'] = ba['r'].str[1:].astype(int)
    ba_wwtp = pd.read_excel(_fetch('GHG_accounting/input_data/WWTP_balancing_area.xlsx'))
    ba_wwtp = ba_wwtp.merge(ba[['r', 'kg_CO2_kWh']], left_on='balancing_area', right_on='r')
    # CWNS_NUMs are stored as integers — zero-pad to 11 digits to match string CWNS_IDs
    ba_wwtp['CWNS_NUM'] = ba_wwtp['CWNS_NUM'].astype(str).str.zfill(11)
    return ba_wwtp.set_index('CWNS_NUM')['kg_CO2_kWh']


def assign_treatment_trains(df):
    """
    Assign Tarallo et al. 2015 treatment train codes from binary WERF-code columns.
    Directly adapted from treatment_train_werf() in tt_assignment_2022.ipynb (cell 20).
    BIOGAS_EL is fixed to 0: E-train variants require the DOE/WEF biogas databases
    which are not replicated here.
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
    # BIOGAS_EL: pre-set from caller if available (e.g. Source 3 derives from AND/cogen);
    # otherwise 0 (Source 2 strips it; Source 1 reads it from tt_assignments_2022.csv).
    if 'BIOGAS_EL' not in d.columns:
        d['BIOGAS_EL'] = 0

    def assign(name, check, exceptions=None):
        if exceptions is None:
            exceptions = []
        d[name] = sum(d[c] for c in check)
        d.loc[d[name] != len(check), name] = 0
        d.loc[d[name] == len(check), name] = 1
        for exc in exceptions:
            d.loc[d[exc] == 1, name] = 0

    assign('N1E', ['MBR-BNR', 'AND', 'BIOGAS_EL'])
    assign('N1',  ['MBR-BNR', 'AND'],                   ['N1E'])
    assign('N2',  ['MBR-BNR', 'AED'])
    assign('H1E', ['AS_BNR_P', 'AND', 'CHEM-P', 'BIOGAS_EL'])
    assign('H1',  ['AS_BNR_P', 'AND', 'CHEM-P'],        ['H1E'])
    assign('G6',  ['AS_BNR_P', 'FBI'])
    assign('G5',  ['AS_BNR_P', 'MHI'])
    assign('G3',  ['AS_BNR_P', 'LIME'])
    assign('G2',  ['AS_BNR_P', 'AED'])
    assign('G1E', ['AS_BNR_P', 'AND', 'BIOGAS_EL'],     ['H1E', 'H1'])
    assign('G1',  ['AS_BNR_P', 'AND'],                  ['G1E', 'H1E', 'H1'])
    assign('I6',  ['AS_BNR_N', 'FBI'],                  ['G6'])
    assign('I5',  ['AS_BNR_N', 'MHI'],                  ['G5'])
    assign('I3',  ['AS_BNR_N', 'LIME'],                 ['G3'])
    assign('I2',  ['AS_BNR_N', 'AED'],                  ['G2'])
    assign('I1E', ['AS_BNR_N', 'AND', 'BIOGAS_EL'],     ['H1', 'H1E', 'G1', 'G1E'])
    assign('I1',  ['AS_BNR_N', 'AND'],                  ['I1E', 'H1', 'H1E', 'G1', 'G1E'])
    assign('F1E', ['BASIC_AS', 'AND', 'NIT', 'BIOGAS_EL'],
                                                         ['AS_BNR_N', 'G1', 'G1E', 'H1', 'H1E', 'I1', 'I1E'])
    assign('F1',  ['BASIC_AS', 'AND', 'NIT'],            ['AS_BNR_N', 'F1E', 'G1', 'G1E', 'H1', 'H1E', 'I1', 'I1E'])
    assign('E2P', ['BASIC_AS', 'AED', 'NIT', 'PRIMARY'], ['AS_BNR_N', 'G2', 'I2'])
    assign('E2',  ['BASIC_AS', 'AED', 'NIT'],            ['AS_BNR_N', 'G2', 'I2', 'E2P'])
    assign('O5',  ['AS-PUREO', 'MHI'],                  ['G5', 'I5'])
    assign('O6',  ['AS-PUREO', 'FBI'],                  ['G6', 'I6'])
    assign('O3',  ['AS-PUREO', 'LIME'],                 ['G3', 'I3'])
    assign('O2',  ['AS-PUREO', 'AED'],                  ['G2', 'I2', 'E2', 'E2P'])
    assign('O1E', ['AS-PUREO', 'AND', 'BIOGAS_EL'],     ['F1', 'F1E', 'G1E', 'G1', 'I1', 'I1E', 'H1', 'H1E'])
    assign('O1',  ['AS-PUREO', 'AND'],                  ['F1', 'F1E', 'O1E', 'G1E', 'G1', 'I1', 'I1E', 'H1', 'H1E'])
    assign('D5',  ['TF_ALL', 'MHI'])
    assign('D6',  ['TF_ALL', 'FBI'])
    assign('D3',  ['TF_ALL', 'LIME'])
    assign('D2',  ['TF_ALL', 'AED'])
    assign('D1E', ['TF_ALL', 'AND', 'BIOGAS_EL'])
    assign('D1',  ['TF_ALL', 'AND'],                    ['D1E'])
    assign('B6',  ['BASIC_AS', 'FBI', 'PRIMARY'],        ['G6', 'I6', 'O6'])
    assign('B5',  ['BASIC_AS', 'MHI', 'PRIMARY'],        ['AS-PUREO', 'G5', 'I5', 'O5'])
    assign('B4',  ['BASIC_AS', 'AND', 'BIODRY', 'PRIMARY'])
    assign('B3',  ['BASIC_AS', 'LIME', 'PRIMARY'],       ['G3', 'I3', 'O3'])
    assign('B2',  ['BASIC_AS', 'AED', 'PRIMARY'],        ['E2', 'E2P', 'G2', 'I2', 'N2', 'O2'])
    assign('B1E', ['BASIC_AS', 'AND', 'PRIMARY', 'BIOGAS_EL'],
                                                         ['AS_BNR_N', 'AS-PUREO', 'B4', 'F1', 'F1E',
                                                          'G1', 'G1E', 'H1', 'H1E', 'I1', 'I1E', 'N1', 'N1E', 'O1', 'O1E'])
    assign('B1',  ['BASIC_AS', 'AND', 'PRIMARY'],        ['AS_BNR_N', 'AS-PUREO', 'B1E', 'B4', 'F1', 'F1E',
                                                          'G1', 'G1E', 'H1', 'H1E', 'I1', 'I1E', 'N1', 'N1E', 'O1', 'O1E'])
    assign('C5',  ['BASIC_AS', 'MHI'],                  ['B5', 'G5', 'I5', 'O5'])
    assign('C6',  ['BASIC_AS', 'FBI'],                  ['B6', 'G6', 'I6', 'O6'])
    assign('C3',  ['BASIC_AS', 'LIME'],                 ['B3', 'G3', 'I3', 'O3'])
    assign('C2',  ['BASIC_AS', 'AED'],                  ['B2', 'E2', 'E2P', 'G2', 'I2', 'N2', 'O2'])
    assign('C1E', ['BASIC_AS', 'AND', 'BIOGAS_EL'],     ['B1', 'B1E', 'B4', 'F1', 'F1E',
                                                          'G1', 'G1E', 'H1', 'H1E', 'I1E', 'I1', 'N1E', 'N1', 'O1', 'O1E'])
    assign('C1',  ['BASIC_AS', 'AND'],                  ['B1', 'B1E', 'B4', 'C1E', 'F1', 'F1E',
                                                          'G1', 'G1E', 'H1', 'H1E', 'I1', 'I1E', 'N1', 'N1E', 'O1', 'O1E'])
    d['TT_IDENTIFIED'] = sum(
        d[tt] for tt in ('LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC',
                         'STBL_POND', 'I1E', 'G6', 'I6', 'O5', 'O6', 'O3', 'O1E',
                         'G5', 'I5', 'C5', 'C6', 'O2', 'O1', 'N1', 'N1E', 'N2',
                         'I3', 'I2', 'I1', 'H1', 'H1E', 'G3', 'G2', 'G1', 'G1E',
                         'F1', 'F1E', 'E2', 'E2P', 'D5', 'D6', 'D1', 'D1E', 'D3',
                         'D2', 'C3', 'C2', 'C1', 'C1E', 'B6', 'B5', 'B4', 'B3',
                         'B1E', 'B1', 'B2'))
    return d


def calc_ghg(wwtp_df, ef, grid_carbon, source_label=''):
    """
    Compute per-facility GHG emissions and CA totals.
    wwtp_df must have: CWNS_NUM (str), FLOW_2022_MGD_FINAL (float), TT_IDENTIFIED,
                       and binary TT columns (B1, C1, ..., LAGOON_UNCATEGORIZED).
    Drops facilities with TT_IDENTIFIED < 1.
    Returns (wwtp_df_with_emissions, totals_dict).
    """
    df = wwtp_df.copy()
    if 'LAGOON_UNCATEGORIZED' not in df.columns:
        lo = df['LAGOON_OTHER'].values if 'LAGOON_OTHER' in df.columns else 0
        sp = df['STBL_POND'].values if 'STBL_POND' in df.columns else 0
        df['LAGOON_UNCATEGORIZED'] = pd.DataFrame({'lo': lo, 'sp': sp}).max(axis=1).values

    tt_present = [tt for tt in ALL_TT if tt in df.columns]
    df['TT_IDENTIFIED'] = df[tt_present].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    n_before = len(df)
    df = df[df['TT_IDENTIFIED'] >= 1].copy()
    if source_label:
        print(f'  {source_label}: {n_before} in → {len(df)} with TT assignment')

    df['FLOW_2022_MGD_FINAL'] = pd.to_numeric(df['FLOW_2022_MGD_FINAL'], errors='coerce').fillna(0)

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

    MT = 365 / 1e6  # kg/day → MT/year
    totals = {
        'n_facilities':   len(df),
        'total_flow_MGD': df['FLOW_2022_MGD_FINAL'].sum(),
        'CH4_MTyr':       df['CH4_med'].sum() * MT,
        'N2O_MTyr':       df['N2O_med'].sum() * MT,
        'NC_CO2_MTyr':    df['NC_CO2_med'].sum() * MT,
        'NG_comb_MTyr':   df['NG_comb_med'].sum() * MT,
        'solids_MTyr':    df['solids_med'].sum() * MT,
        'elec_MTyr':      df['elec_med'].sum() * MT,
        'NG_up_MTyr':     df['NG_up_med'].sum() * MT,
    }
    totals['Scope1_MTyr'] = (totals['CH4_MTyr'] + totals['N2O_MTyr'] + totals['NC_CO2_MTyr']
                             + totals['NG_comb_MTyr'] + totals['solids_MTyr'])
    totals['Scope2_MTyr'] = totals['elec_MTyr']
    totals['Scope3_MTyr'] = totals['NG_up_MTyr']
    totals['total_MTyr']  = totals['Scope1_MTyr'] + totals['Scope2_MTyr'] + totals['Scope3_MTyr']
    return df, totals


def totals_from_df(df):
    """Recompute totals dict from a df that already has *_med emission columns."""
    MT = 365 / 1e6
    totals = {
        'n_facilities':   len(df),
        'total_flow_MGD': df['FLOW_2022_MGD_FINAL'].sum(),
        'CH4_MTyr':       df['CH4_med'].sum() * MT,
        'N2O_MTyr':       df['N2O_med'].sum() * MT,
        'NC_CO2_MTyr':    df['NC_CO2_med'].sum() * MT,
        'NG_comb_MTyr':   df['NG_comb_med'].sum() * MT,
        'solids_MTyr':    df['solids_med'].sum() * MT,
        'elec_MTyr':      df['elec_med'].sum() * MT,
        'NG_up_MTyr':     df['NG_up_med'].sum() * MT,
    }
    totals['Scope1_MTyr'] = (totals['CH4_MTyr'] + totals['N2O_MTyr'] + totals['NC_CO2_MTyr']
                             + totals['NG_comb_MTyr'] + totals['solids_MTyr'])
    totals['Scope2_MTyr'] = totals['elec_MTyr']
    totals['Scope3_MTyr'] = totals['NG_up_MTyr']
    totals['total_MTyr']  = totals['Scope1_MTyr'] + totals['Scope2_MTyr'] + totals['Scope3_MTyr']
    return totals


# ── matching helpers ──────────────────────────────────────────────────────────

def load_common_place_ids():
    """
    Return the set of Place IDs present in both unit_processes_by_facility_cwns.csv and the
    LLM output — the same intersection used in figure_4_source_comparison.py.
    """
    cwns_out = pd.read_csv(
        REPO_ROOT / 'wwtp_process_extraction/output/unit_processes_by_facility_cwns.csv', dtype=str)
    ciwqs = pd.read_csv(
        REPO_ROOT / 'wwtp_process_extraction/data/ciwqs_to_cwns.csv', dtype=str)
    llm = pd.read_csv(
        REPO_ROOT / 'wwtp_process_extraction/output/2026-5-25/unit_processes_by_facility_llm.csv',
        dtype=str)

    cwns_pids = set(ciwqs[ciwqs['CWNS_ID'].isin(cwns_out['CWNS_ID'])]['Place ID'].dropna())
    llm_pids = set(llm['Place ID'].dropna())
    return cwns_pids & llm_pids


def pid_to_cwns_flow(common_pids):
    """
    For each Place ID in common_pids, return a DataFrame with one row per CWNS_NUM
    linking Place ID → CWNS_NUM → FLOW_2022_MGD_FINAL.
    One Place ID can map to multiple CWNS facilities; flow is per CWNS facility.
    """
    ciwqs = pd.read_csv(
        REPO_ROOT / 'wwtp_process_extraction/data/ciwqs_to_cwns.csv', dtype=str)
    ciwqs = ciwqs[ciwqs['Place ID'].isin(common_pids)].dropna(subset=['CWNS_ID']).copy()
    ciwqs['CWNS_NUM'] = ciwqs['CWNS_ID'].str.strip()

    tt = pd.read_csv(_fetch('GHG_accounting/input_data/tt_assignments_2022.csv'),
                     dtype={'CWNS_NUM': str})
    tt_flow = tt[['CWNS_NUM', 'FLOW_2022_MGD_FINAL']].copy()
    tt_flow['FLOW_2022_MGD_FINAL'] = pd.to_numeric(tt_flow['FLOW_2022_MGD_FINAL'], errors='coerce')

    link = ciwqs[['Place ID', 'CWNS_NUM']].merge(tt_flow, on='CWNS_NUM', how='inner')
    # Drop duplicate CWNS_NUMs (one CWNS facility shouldn't count for multiple Place IDs)
    link = link.drop_duplicates(subset=['CWNS_NUM'])
    return link


# ── Sources 1 & 2: El Abbadi (CWNS-based) ────────────────────────────────────

def load_cwns_source(common_pids, include_manual=True):
    """
    Load El Abbadi CWNS-based treatment train data.

    include_manual=True  (Source 1): reads pre-assigned TTs directly from
        tt_assignments_2022.csv, which includes DOE/WEF biogas E-train upgrades,
        manual lagoon corrections, and manual TT overrides for large plants.

    include_manual=False (Source 2): re-derives TT assignments from
        unit_processes_by_facility_cwns.csv using our pipeline (same as Source 3),
        with BIOGAS_EL=0. No manual corrections of any kind.
    """
    link = pid_to_cwns_flow(common_pids)

    if include_manual:
        tt = pd.read_csv(_fetch('GHG_accounting/input_data/tt_assignments_2022.csv'),
                         dtype={'CWNS_NUM': str})
        for col in tt.columns:
            if col != 'CWNS_NUM':
                tt[col] = pd.to_numeric(tt[col], errors='coerce').fillna(0)
        df = link[['CWNS_NUM', 'FLOW_2022_MGD_FINAL']].merge(
            tt.drop(columns=['FLOW_2022_MGD_FINAL'], errors='ignore'),
            on='CWNS_NUM', how='inner')
        if 'LAGOON_OTHER' not in df.columns:
            df['LAGOON_OTHER'] = 0
        df['LAGOON_UNCATEGORIZED'] = df[['LAGOON_OTHER', 'STBL_POND']].max(axis=1)
        print(f'Source 1 (El Abbadi, all sources): {len(df)} CWNS rows')
        return df

    # include_manual=False: re-derive from CWNS unit processes
    ca_cwns = pd.read_csv(
        REPO_ROOT / 'wwtp_process_extraction/output/unit_processes_by_facility_cwns.csv',
        dtype=str)
    cwns_by_pid, _ = build_cwns_facility_processes(ca_cwns, target_facilities=common_pids)
    cwns_by_pid = cwns_by_pid[cwns_by_pid['Place ID'].isin(common_pids)].copy()

    status_cols = [c for c in cwns_by_pid.columns if c in CWNS_COL_TO_WERF]
    records = []
    for _, row in cwns_by_pid.iterrows():
        werf_present = set()
        for col in status_cols:
            if str(row.get(col, '')).strip().upper() in PRESENT_STATUSES:
                werf_present.update(CWNS_COL_TO_WERF[col])
        rec = {'Place ID': str(row['Place ID']).strip()}
        for wc in werf_present:
            rec[wc] = 1
        records.append(rec)

    cwns_pivot = pd.DataFrame(records).fillna(0)
    nit_indicator = ['BNIT', 'BNR', 'AS-BDENIT', 'AS-A2O']
    cwns_pivot['NIT'] = (cwns_pivot[[c for c in nit_indicator if c in cwns_pivot.columns]]
                         .sum(axis=1) > 0).astype(int)
    bio_p_indicator = ['AS-A2O', 'BNR']
    cwns_pivot['BIO-P'] = (cwns_pivot[[c for c in bio_p_indicator if c in cwns_pivot.columns]]
                           .sum(axis=1) > 0).astype(int)
    cwns_pivot['BIOGAS_EL'] = 0

    secondary_werf = ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P',
                      'AS-PUREO', 'AS-SA', 'AS-SBR', 'TF', 'TF-BF', 'TF-RBC', 'MBR-BNR']
    lagoon_werf = ['LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'STBL_POND']
    has_secondary = cwns_pivot[[c for c in secondary_werf if c in cwns_pivot.columns]].sum(axis=1) > 0
    for col in lagoon_werf:
        if col in cwns_pivot.columns:
            cwns_pivot.loc[has_secondary, col] = 0

    merged = link.merge(cwns_pivot, on='Place ID', how='inner')
    df = assign_treatment_trains(merged)
    df['LAGOON_OTHER'] = df['LAGOON'].values if 'LAGOON' in df.columns else 0
    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON_OTHER', 'STBL_POND']].max(axis=1) \
        if 'STBL_POND' in df.columns else df['LAGOON_OTHER']

    # Regional fallback using El Abbadi flow-size bins (all CA = EPA Region 9, so only
    # size matters for within-CA pool). Bins match tt_assignment_2022.ipynb exactly.
    tt_cols = [tt for tt in ALL_TT if tt in df.columns]
    flow_bins = [0, 2, 4, 7, 16, 46, 100, float('inf')]
    flow_labels = ['<2', '2-4', '4-7', '7-16', '16-46', '46-100', '≥100']
    df['_size_bin'] = pd.cut(
        pd.to_numeric(df['FLOW_2022_MGD_FINAL'], errors='coerce').fillna(0),
        bins=flow_bins, labels=flow_labels, right=False)

    assigned = df[df['TT_IDENTIFIED'] >= 1].copy()
    no_tt_mask = df['TT_IDENTIFIED'] < 1
    n_fallback = 0
    for idx in df[no_tt_mask].index:
        size = df.at[idx, '_size_bin']
        # narrow → wide fallback: size → all assigned (all CA is EPA Region 9)
        for pool_mask in [
            assigned['_size_bin'] == size,
            pd.Series(True, index=assigned.index),
        ]:
            pool = assigned[pool_mask]
            if pool.empty:
                continue
            tt_sums = pool[tt_cols].sum()
            best_tt = tt_sums.idxmax()
            if tt_sums[best_tt] > 0:
                df.at[idx, best_tt] = 1
                df.at[idx, 'TT_IDENTIFIED'] = 1
                n_fallback += 1
                break

    df = df.drop(columns=['_size_bin'])
    print(f'Source 2 (CWNS-only, re-derived): {len(df)} CWNS rows, '
          f'{int((df["TT_IDENTIFIED"] >= 1).sum())} with TT assignment '
          f'({n_fallback} via regional fallback)')
    return df


# ── Source 3: permit scraping ─────────────────────────────────────────────────────

def load_ontology_pivot(common_pids):
    """Read per-facility ontology JSONs and map detected processes to WERF codes.

    Uses IS-A inheritance so e.g. MLE → AO → ActivatedSludge → AS, giving
    BASIC_AS=True even when only a specific variant is named in the permit text.
    Returns a DataFrame with one row per Place ID, binary WERF code columns.
    """
    # Place ID is encoded as the last underscore-separated number in each filename
    pid_to_jsons = {}
    for json_path in ONTOLOGY_JSON_DIR.glob('*.json'):
        last_part = json_path.stem.rsplit('_', 1)[-1]
        if last_part.isdigit():
            pid_to_jsons.setdefault(last_part, []).append(json_path)

    present_implementations = {'present', 'planned', 'future'}
    records = []
    for pid in common_pids:
        werf_codes_found = set()
        for json_path in pid_to_jsons.get(str(pid), []):
            with open(json_path) as f:
                data = json.load(f)
            for item in data.get('items', []):
                if str(item.get('Implementation', '')).lower() not in present_implementations:
                    continue
                # Process field uses "Process-X" IRI prefix; strip it
                procs = [p.replace('Process-', '') for p in (item.get('Process') or [])]
                equip = item.get('Equipment')
                roles = item.get('Role') or []

                # Expand each detected process/equipment via ontology IS-A ancestors
                candidates = procs + ([equip] if equip else [])
                for candidate in candidates:
                    for ancestor in get_ontology_ancestors(candidate):
                        werf_codes_found.update(ONTOLOGY_TO_WERF.get(ancestor, []))

                # Lagoons appear as Equipment=Pond; type comes from the Role field
                if equip == 'Pond':
                    role_str = ' '.join(roles).lower()
                    if 'aerobic' in role_str or 'aerated' in role_str:
                        werf_codes_found.add('LAGOON_AER')
                    elif 'anaerobic' in role_str:
                        werf_codes_found.add('LAGOON_ANAER')
                    elif 'stabilization' in role_str or 'facultative' in role_str:
                        werf_codes_found.add('LAGOON_FAC')
                    else:
                        werf_codes_found.add('LAGOON')

        rec = {'Place ID': str(pid)}
        for wc in werf_codes_found:
            rec[wc] = 1
        records.append(rec)

    return pd.DataFrame(records).fillna(0)


def load_source3(common_pids):
    """
    LLM-extracted unit processes for common_pids, mapped to WERF codes via ontology
    inheritance, then to treatment trains using the Tarallo et al. 2015 assignment logic.
    One row per CWNS_NUM (facilities with multiple CWNS_IDs under one Place ID are
    expanded so that each CWNS_NUM gets the same LLM unit processes but its own flow).
    """
    llm_pivot = load_ontology_pivot(common_pids)

    # NIT flag: set if nitrification-type process present
    nit_indicator = ['BNIT', 'BNR', 'AS-BDENIT', 'AS-A2O']
    llm_pivot['NIT'] = (llm_pivot[[c for c in nit_indicator if c in llm_pivot.columns]]
                        .sum(axis=1) > 0).astype(int)

    # BIO-P flag: A2O and multi-stage Bardenpho perform biological P removal
    bio_p_indicator = ['AS-A2O', 'BNR']
    llm_pivot['BIO-P'] = (llm_pivot[[c for c in bio_p_indicator if c in llm_pivot.columns]]
                          .sum(axis=1) > 0).astype(int)

    # BIOGAS_EL: proxy from anaerobic digestion and/or cogeneration detection
    and_col = llm_pivot['AND'] if 'AND' in llm_pivot.columns else 0
    cogen_col = llm_pivot['BIOGAS_CWNS'] if 'BIOGAS_CWNS' in llm_pivot.columns else 0
    llm_pivot['BIOGAS_EL'] = ((and_col > 0) | (cogen_col > 0)).astype(int)

    # Clear lagoon WERF codes before TT assignment if any secondary treatment is present
    # (lagoons in LLM output are often storage/effluent ponds, not treatment lagoons).
    # Done pre-assignment so TT_IDENTIFIED is computed correctly from the start.
    secondary_werf = ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P',
                      'AS-PUREO', 'AS-SA', 'AS-SBR', 'TF', 'TF-BF', 'TF-RBC', 'MBR-BNR']
    lagoon_werf = ['LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'STBL_POND']
    has_secondary = llm_pivot[[c for c in secondary_werf if c in llm_pivot.columns]].sum(axis=1) > 0
    for col in lagoon_werf:
        if col in llm_pivot.columns:
            llm_pivot.loc[has_secondary, col] = 0
    n_cleared = int(has_secondary.sum())
    if n_cleared:
        print(f'  Cleared lagoon WERF codes for {n_cleared} facilities with secondary treatment')

    # expand to one row per CWNS_NUM (so each gets its own flow)
    link = pid_to_cwns_flow(common_pids)
    merged = link.merge(llm_pivot, on='Place ID', how='inner')

    df = assign_treatment_trains(merged)

    df['LAGOON_OTHER'] = df['LAGOON'].values if 'LAGOON' in df.columns else 0
    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON_OTHER', 'STBL_POND']].max(axis=1) \
        if 'STBL_POND' in df.columns else df['LAGOON_OTHER']
    n_with_tt = int((df['TT_IDENTIFIED'] >= 1).sum())
    print(f'Source 3 (WE3Lab LLM): {len(df)} CWNS rows, {n_with_tt} with TT assignment')
    print(f'  ({len(df) - n_with_tt} without sufficient unit processes for TT assignment — excluded from GHG calc)')
    return df


# ── comparison plot ───────────────────────────────────────────────────────────
src_labels = {
    'source1': 'El Abbadi\n(CWNS +\nManual Corrections)',
    'source2': 'El Abbadi\n(CWNS-only)',
    'source3': 'New Data\n(Permit Scraping)',
}


def plot_comparison(results, output_dir):
    sources = ['source2', 'source1', 'source3']

    components = {
        'CH₄ (Scope 1)':            ('CH4_MTyr',     cb_palette[0]),
        'N₂O (Scope 1)':            ('N2O_MTyr',     cb_palette[1]),
        'Bio CO₂ (Scope 1)':        ('NC_CO2_MTyr',  cb_palette[2]),
        'FNG combustion (Scope 1)': ('NG_comb_MTyr', cb_palette[3]),
        'Biosolids (Scope 1)':      ('solids_MTyr',  cb_palette[4]),
        'Electricity (Scope 2)':    ('elec_MTyr',    cb_palette[5]),
        'NG upstream (Scope 3)':    ('NG_up_MTyr',   cb_palette[6]),
    }

    x = np.arange(len(sources))
    width = 0.5
    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = np.zeros(len(sources))
    for label, (key, color) in components.items():
        vals = np.array([results[s][key] for s in sources])
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=color)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([src_labels[s] for s in sources], fontsize=10)
    ax.set_ylabel(f'kt CO₂e / year\nn={r["n_facilities"]} facilities, {r["total_flow_MGD"]:.0f} MGD', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(output_dir / 'ca_ghg_comparison_stacked.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── N2O breakdown by secondary treatment ─────────────────────────────────────

# TT code → secondary treatment label (what drives N2O EF differences)
_NO_NR = 'AS / TF (no nutrient removal)'  # EF ~0.044 — default for any unmapped TT
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
    _NO_NR,
    'AS + Nitrification',
    'AS + Nitrif./Denitrif.',
    'AS + BNR',
    'BNR-MBR',
    'Lagoon',
]


def n2o_by_secondary(df, ef, label=''):
    """Return dict[secondary_label → N2O MT/yr] for one source dataframe."""
    MT = 365 / 1e6
    valid_tt = [tt for tt in ALL_TT if tt in df.columns and tt in ef.index]
    tt_mat = df[valid_tt].apply(pd.to_numeric, errors='coerce').fillna(0)
    tt_identified = pd.to_numeric(df['TT_IDENTIFIED'], errors='coerce').clip(lower=1)
    tt_flow = tt_mat.div(tt_identified, axis=0).mul(
        pd.to_numeric(df['FLOW_2022_MGD_FINAL'], errors='coerce').fillna(0), axis=0
    )
    groups = {}
    tt_n2o = {}
    for tt in valid_tt:
        n2o = (tt_flow[tt] * ef.loc[tt, 'N2O_50'] * MG_2_m3 * MT).sum()
        tt_n2o[tt] = n2o
        label_ = TT_SECONDARY.get(tt, _NO_NR)
        groups[label_] = groups.get(label_, 0) + n2o
    if label:
        dup_cwns = df['CWNS_NUM'].duplicated().sum()
        top = sorted(tt_n2o.items(), key=lambda x: -x[1])[:8]
        top_str = ', '.join(f'{t}={v:.1f}' for t, v in top if v > 0)
        print(f'  [{label}] n={len(df)} rows, {dup_cwns} dup CWNS_NUMs, '
              f'total N2O={sum(groups.values()):.1f} MT/yr | top TTs: {top_str}')
    return groups


# colorblind-safe palette mapped to treatment categories
cb_palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#CC79A7', '#D55E00']
CATEGORY_COLORS = dict(zip(SECONDARY_ORDER, cb_palette))


def _grouped_legend(ax, header_items_pairs):
    """Grouped legend with bold headers. header_items_pairs: [(header, [Patch,...]), ...]"""
    from matplotlib.patches import Patch
    handles = []
    header_labels = set()
    for header, items in header_items_pairs:
        handles.append(Patch(color='none', label=header))
        header_labels.add(header)
        handles.extend(items)
    leg = ax.legend(handles=handles, loc='upper left', fontsize=10, frameon=False,
                    bbox_to_anchor=(1.01, 1), borderaxespad=0)
    for h, t in zip(leg.legend_handles, leg.get_texts()):
        if h.get_label() in header_labels:
            h.set_visible(False)
            t.set_fontweight('bold')


def plot_n2o_breakdown(dfs, ef, output_dir):
    from matplotlib.patches import Patch
    sources = ['source2', 'source1', 'source3']

    print('N2O breakdown diagnostics:')
    breakdowns = {s: n2o_by_secondary(df, ef, label=s) for s, df in dfs.items()}
    present_cats = [c for c in SECONDARY_ORDER
                    if any(breakdowns[s].get(c, 0) > 0 for s in sources)]

    first_df = list(dfs.values())[0]
    n_fac = len(first_df)
    tot_flow = pd.to_numeric(first_df['FLOW_2022_MGD_FINAL'], errors='coerce').sum()
    ylabel = f'N₂O  kt CO₂e / year\nn={n_fac} facilities, {tot_flow:.0f} MGD'
    x = np.arange(len(sources))
    width = 0.45

    def _style_ax(ax):
        ax.set_xticks(x)
        ax.set_xticklabels([src_labels[s] for s in sources], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='y', labelsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bottoms = np.zeros(len(sources))
    for cat in present_cats:
        vals = np.array([breakdowns[s].get(cat, 0) for s in sources])
        ax.bar(x, vals, width, bottom=bottoms,
               color=CATEGORY_COLORS[cat], edgecolor='black', linewidth=0.4)
        bottoms += vals
    _style_ax(ax)
    cat_handles = [
        Patch(facecolor=CATEGORY_COLORS[cat], edgecolor='black', linewidth=0.5, label=cat)
        for cat in present_cats
    ]
    _grouped_legend(ax, [('Treatment Type', list(reversed(cat_handles)))])
    fig.tight_layout()
    fig.savefig(output_dir / 'ca_n2o_by_secondary_color.png', dpi=150, bbox_inches='tight')
    plt.close(fig)



# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading MC emission factors...')
    ef = load_mc_ef()
    print(f'  Loaded EFs for {len(ef)} treatment trains')

    print('Loading grid carbon intensity...')
    grid_carbon = load_grid_carbon()

    print('\nBuilding common facility set (Place ID intersection: CWNS ∩ LLM)...')
    common_pids = load_common_place_ids()
    print(f'  {len(common_pids)} common Place IDs')

    df1_raw = load_cwns_source(common_pids, include_manual=True)
    df2_raw = load_cwns_source(common_pids, include_manual=False)
    df3_raw = load_source3(common_pids)

    # 1:1 matching: filter Sources 1 and 2 to the same CWNS_NUMs that Source 3 assigned
    assigned_cwns = set(df3_raw.loc[df3_raw['TT_IDENTIFIED'] >= 1, 'CWNS_NUM'])
    print(f'\nFiltering Sources 1 & 2 to the {len(assigned_cwns)} CWNS facilities '
          f'where LLM assigned a TT...')
    df1 = df1_raw[df1_raw['CWNS_NUM'].isin(assigned_cwns)].copy()
    df2 = df2_raw[df2_raw['CWNS_NUM'].isin(assigned_cwns)].copy()
    df3 = df3_raw.copy()

    print('\n── GHG calculation ──')
    df1_ghg, _ = calc_ghg(df1, ef, grid_carbon, 'Source 1')
    df2_ghg, _ = calc_ghg(df2, ef, grid_carbon, 'Source 2')
    df3_ghg, _ = calc_ghg(df3, ef, grid_carbon, 'Source 3')

    # Enforce 3-way intersection: only rows present in ALL sources
    s1_cwns = set(df1_ghg['CWNS_NUM'])
    s2_cwns = set(df2_ghg['CWNS_NUM'])
    s3_cwns = set(df3_ghg['CWNS_NUM'])
    common_cwns = s1_cwns & s2_cwns & s3_cwns
    print(f'\n── 3-way intersection ──')
    print(f'  Source 1: {len(s1_cwns)} | Source 2: {len(s2_cwns)} | Source 3: {len(s3_cwns)}')
    print(f'  Dropped by Source 2 only: {len(s1_cwns & s3_cwns - s2_cwns)} CWNS facilities')
    print(f'  Dropped by Source 3 only: {len(s1_cwns & s2_cwns - s3_cwns)} CWNS facilities')
    print(f'  Final 3-way intersection: {len(common_cwns)} CWNS facilities')
    df1_ghg = df1_ghg[df1_ghg['CWNS_NUM'].isin(common_cwns)].copy()
    df2_ghg = df2_ghg[df2_ghg['CWNS_NUM'].isin(common_cwns)].copy()
    df3_ghg = df3_ghg[df3_ghg['CWNS_NUM'].isin(common_cwns)].copy()

    totals1 = totals_from_df(df1_ghg)
    totals2 = totals_from_df(df2_ghg)
    totals3 = totals_from_df(df3_ghg)

    results = {'source1': totals1, 'source2': totals2, 'source3': totals3}
    source_names = {
        'source1': 'El Abbadi (all sources)',
        'source2': 'El Abbadi (CWNS-only)',
        'source3': 'WE3Lab LLM',
    }

    print('\n════════════════════════════════════════════════════════════════')
    print('California WWTP GHG Emissions — Source Comparison (matched facilities)')
    print('════════════════════════════════════════════════════════════════')
    rows = []
    for s, name in source_names.items():
        r = results[s]
        flow = r['total_flow_MGD']
        rows.append({
            'Source': name,
            'N CWNS': r['n_facilities'],
            'Flow (MGD)': f"{flow:.0f}",
            'CH4 (kt/yr)': f"{r['CH4_MTyr']:.1f}",
            'N2O (kt/yr)': f"{r['N2O_MTyr']:.1f}",
            'NC_CO2 (kt/yr)': f"{r['NC_CO2_MTyr']:.1f}",
            'NG comb (kt/yr)': f"{r['NG_comb_MTyr']:.1f}",
            'Biosolids (kt/yr)': f"{r['solids_MTyr']:.1f}",
            'Elec (kt/yr)': f"{r['elec_MTyr']:.1f}",
            'NG up (kt/yr)': f"{r['NG_up_MTyr']:.1f}",
            'Scope 1': f"{r['Scope1_MTyr']:.1f}",
            'Scope 2': f"{r['Scope2_MTyr']:.1f}",
            'Scope 3': f"{r['Scope3_MTyr']:.1f}",
            'TOTAL (kt/yr)': f"{r['total_MTyr']:.1f}",
            'per MGD': f"{r['total_MTyr']/flow:.3f}" if flow > 0 else 'N/A',
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / 'ca_ghg_summary.csv', index=False)

    print('\n── % change relative to El Abbadi (CWNS-only) ──')
    base = totals2
    for key, name in [('source1', 'El Abbadi (all sources)'), ('source3', 'WE3Lab LLM')]:
        r = results[key]
        n2o_pct = (r['N2O_MTyr'] - base['N2O_MTyr']) / base['N2O_MTyr'] * 100
        tot_pct = (r['total_MTyr'] - base['total_MTyr']) / base['total_MTyr'] * 100
        print(f'  {name}: N2O {n2o_pct:+.1f}%,  Total {tot_pct:+.1f}%')
    df1_ghg.to_csv(OUTPUT_DIR / 'source1_facility_emissions.csv', index=False)
    df2_ghg.to_csv(OUTPUT_DIR / 'source2_facility_emissions.csv', index=False)
    df3_ghg.to_csv(OUTPUT_DIR / 'source3_facility_emissions.csv', index=False)

    print('\nGenerating plots...')
    plot_comparison(results, OUTPUT_DIR)
    plot_n2o_breakdown(
        {'source1': df1_ghg, 'source2': df2_ghg, 'source3': df3_ghg},
        ef, OUTPUT_DIR,
    )
    print('Done.')
