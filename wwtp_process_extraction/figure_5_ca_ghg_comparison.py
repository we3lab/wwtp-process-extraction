# Adapted from https://github.com/jiananf2/US_WWTP_GHG/tree/main/GHG_accounting/WWTP_GHG_accounting.py
# by Abigayle Hodson, Abigayle_Hodson@lbl.gov
# Publication: https://eartharxiv.org/repository/view/7980/
# Modified by WE3Lab: CA GHG under five unit-process data sources, on the Place IDs in both
# unit_processes_by_facility_cwns.csv and the LLM output.
# 
# Biosolids use per-train MC medians. electricity uses each WWTP's balancing-area grid carbon.
# Re-derived sources get El Abbadi's flow-size-bin fallback (all CA is EPA Region 9). Published assignments need none.

import ast
import io
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from helpers.plotting import make_grouped_legend, save_and_close
from helpers.paths import *
from helpers.utils import (
    build_cwns_facility_processes, collapse_facility_processes, current_permit_mask, is_present,
    leaves,
)

# Local copies of the El Abbadi US_WWTP_GHG files, downloaded from GitHub (gitignored)
GHG_CACHE_DIR = DATA_DIR / 'ghg_cache'
MC_DIR = 'uncertainty_sensitivity_results/Monte_Carlo'
# The 50th-percentile factors distilled from the ~180 MB of upstream Monte Carlo workbooks.
# Cached because it is 49 rows of derived numbers; delete it to refetch and recompute.
MC_EF_CSV = DATA_DIR / 'ghg_mc_emission_factors.csv'
WERF_CODES_CSV = DATA_DIR / 'el_abbadi' / 'UNIT_PROCESS_EI_CODES_WERF_modified.csv'

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

LAGOON_TRAINS = ['LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'LAGOON_UNCATEGORIZED']
# Unit processes each train requires: the check list of each upstream assign() call (minus the
# BIOGAS_EL flag); lagoons come straight from their unit process. The fallback uses it to pick
# a train compatible with what a facility did report.
TT_REQUIRES = {
    call.args[0].value: {e.value for e in call.args[1].elts if isinstance(e, ast.Constant)}
    for call in ast.walk(ast.parse(TRAIN_SOURCE))
    if isinstance(call, ast.Call) and call.func.id == 'assign'
} | {lagoon: set() for lagoon in LAGOON_TRAINS}
ALL_TT = sorted(TT_REQUIRES)
# E-train → non-E equivalent (fallback rule 2: biogas electricity is never imputed)
E_TO_BASE = {tt: tt[:-1] for tt in ALL_TT if tt.endswith('E')}

# The one train with no upstream MC workbook (verified 404): El Abbadi assign it to 0 of 15,863
# facilities, but permits can reach it, so calc_ghg warns rather than silently zeroing it
EXPECTED_ABSENT_TT = frozenset({'N1'})

# CWNS 2022 and El Abbadi's assignments describe the 2022 fleet, so permits are collapsed as of 2022
# here instead of reading step6's current table. Against their published total, N2O is +22.0% on a
# 2026 basis and +10.1% on a 2022 basis; the total goes from +1.0% to -1.6%.
GHG_AS_OF = '2022-06-01'

# Nitrification is the N2O lever: trains needing one of these carry N2O factors ~0.24-0.27, the
# rest ~0.044 (a ~6x cliff), so crossing it by accident is the worst error a fallback can make
NITRIFYING_UPS = frozenset({'NIT', 'AS_BNR_N', 'AS_BNR_P'})
NITRIFYING_TT = frozenset(tt for tt, req in TT_REQUIRES.items() if req & NITRIFYING_UPS)

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

# Emission component → (legend label, color), in stack order
COMPONENTS = {
    'CH4': ('CH₄ (Scope 1)', '#C94040'),
    'N2O': ('N₂O (Scope 1)', '#9B3080'),
    'NC_CO2': ('Bio CO₂ (Scope 1)', '#F2CFA0'),
    'NG_comb': ('FNG combustion (Scope 1)', '#D4924A'),
    'solids': ('Biosolids (Scope 1)', '#8B5E2E'),
    'elec': ('Electricity (Scope 2)', '#6E92B0'),
    'NG_up': ('NG upstream (Scope 3)', '#B8B8B8'),
}

# Secondary treatment (what drives the N2O factor) → (color, trains), in stack order. Unlisted
# trains are NO_NUTRIENT_REMOVAL (EF ~0.044); G/I/H share the BNR EF (~0.24). Warm light→dark,
# lagoon a contrasting sage.
NO_NUTRIENT_REMOVAL = 'AS / TF (no nutrient removal)'
SECONDARY_GROUPS = {
    NO_NUTRIENT_REMOVAL: ('#FADA99', []),
    'AS + Nitrification': ('#F5B560', ['E2', 'E2P']),
    'AS + Nitrif./Denitrif.': ('#E08030', ['F1', 'F1E']),
    'AS + BNR': ('#C05820', ['G1', 'G1E', 'G2', 'G3', 'G5', 'G6', 'H1', 'H1E',
                             'I1', 'I1E', 'I2', 'I3', 'I5', 'I6']),
    'BNR-MBR': ('#8B3410', ['N1', 'N1E', 'N2']),
    'Lagoon': ('#84B07A', LAGOON_TRAINS),
}
TT_SECONDARY = {tt: group for group, (_, trains) in SECONDARY_GROUPS.items() for tt in trains}


def load_llm_facility_table():
    """Permit-derived processes collapsed per facility, as of GHG_AS_OF."""
    raw = pd.read_csv(PDF_LLM_CSV, dtype=str).fillna('')
    keep = current_permit_mask(raw, as_of=GHG_AS_OF)
    facilities = collapse_facility_processes(
        raw[keep], key_cols=['Place ID'],
        meta_cols=['WDID', 'Order_No', 'NPDES No.', 'Agency', 'Facility Name', 'County',
                   'PDF_File', 'document_order_no'])
    print(f'  Permit basis {GHG_AS_OF}: {int(keep.sum())} of {len(raw)} documents '
          f'-> {len(facilities)} facilities')
    return facilities


def load_cwns_col_to_werf(automatic=False):
    """Leaf column → WERF codes, from the curated werf_codes. automatic=True (experiment) looks them
    up via each leaf's cwns_processes in the El Abbadi WERF CSV, curated codes only where none match."""
    if automatic:
        werf_csv = pd.read_csv(WERF_CODES_CSV, dtype=str)
        csv_lookup = (werf_csv.groupby(werf_csv['FINAL_UNIT_PROCESS_NAME'].str.lower().str.strip())
                      ['WERF_CODE'].apply(lambda s: sorted(set(s.dropna()))).to_dict())
    mapping = {}
    for name, details, _ in leaves:
        codes = set()
        if automatic:
            for cwns_name in details['cwns_processes']:
                codes.update(csv_lookup.get(cwns_name.lower().strip(), []))
        mapping[name] = sorted(codes or details['werf_codes'])
    return mapping


def load_mc_ef():
    """Median emission factors per train, kg CO2e / m³ (elec_50 is kWh / MGD). Read from the cached
    CSV, else from each train's ~3.6 MB upstream MC workbook, skipping EXPECTED_ABSENT_TT."""
    if MC_EF_CSV.exists():
        return pd.read_csv(MC_EF_CSV, index_col=0)
    # factor → upstream workbook column; elec_MC is already kWh/MGD, no MG_2_m3 needed
    columns = {'CH4_50': 'CH4', 'N2O_50': 'N2O', 'NC_CO2_50': 'NC_CO2', 'elec_50': 'elec_MC',
               'NG_comb_50': 'NG_combustion', 'NG_up_50': 'NG_upstream', 'solids_50': 'solids'}
    records = {}
    for tt in ALL_TT:
        if tt not in EXPECTED_ABSENT_TT:
            mc = pd.read_excel(fetch_ghg_file(f'{MC_DIR}/{tt}_MC.xlsx'))
            records[tt] = {name: mc[col].quantile(0.5) for name, col in columns.items()}
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


def assign_treatment_trains(d):
    """Tarallo et al. 2015 trains from binary WERF-code columns: upstream's unit_process_pivot()
    flags (notebook cell 19), then its treatment_train_werf() (cell 20). The caller sets BIOGAS_EL."""
    for col in ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P', 'AS-PUREO', 'AS-SA', 'AS-SBR',
                'AED', 'AND', 'BIO-P', 'BIODRY', 'BNR', 'CHEM-P', 'FBI', 'LAGOON', 'LAGOON_AER',
                'LAGOON_ANAER', 'LAGOON_FAC', 'LIME', 'MBR-BNR', 'MHI', 'NIT', 'PRIMARY',
                'STBL_POND', 'TF', 'TF-BF', 'TF-RBC']:
        if col not in d.columns:
            d[col] = 0
    d['SUM_AS'] = (d['AS'] + d['AS-A2O'] + d['AS-BDENIT'] + d['AS-EA'] + d['AS-P'] + d['AS-PUREO']
                   + d['AS-SA'])
    d['BASIC_AS'] = ((d['AS'] + d['AS-EA'] + d['AS-SA'] + d['AS-OD'] + d['AS-SBR']) > 0).astype(int)
    d['AS_BNR_N'] = ((d['AS-A2O'] + d['AS-BDENIT'] > 0) | ((d['AS'] > 0) & (d['BNR'] > 0))).astype(int)
    d['TF_ALL'] = ((d['TF'] + d['TF-BF'] + d['TF-RBC']) > 0).astype(int)
    d['AS_BNR_P'] = (((d['SUM_AS'] > 0) & (d['BIO-P'] > 0)) | (d['AS-P'] == 1)).astype(int)
    # upstream names the flag BIOGAS_EL_<year> and adds a TT_ASSIGN_NOTE column we don't use
    d['BIOGAS_EL_0'] = d['BIOGAS_EL']
    return treatment_train_werf(d, 0).drop(columns=['BIOGAS_EL_0', 'TT_ASSIGN_NOTE'])


def calc_ghg(wwtp_df, ef, grid_carbon, source_label):
    """Per-facility emissions (*_med, kg CO2e / day) of the facilities with a train; a facility's
    flow is split evenly over its trains."""
    df = wwtp_df.assign(TT_IDENTIFIED=wwtp_df[ALL_TT].sum(axis=1))
    df = df[df['TT_IDENTIFIED'] >= 1].copy()
    print(f'  {source_label}: {len(wwtp_df)} in → {len(df)} with TT assignment')

    # A train without factors makes its facilities invisible, not wrong, so report the lost flow
    # rather than borrow factors: cogeneration saves ~556 kWh/MGD across the eight measurable
    # E/base pairs, but N1E's demand (6023) is 2.5x any of their base trains
    for tt in EXPECTED_ABSENT_TT:
        flow_lost = df[tt].div(df['TT_IDENTIFIED']).mul(df['FLOW_2022_MGD_FINAL']).sum()
        if flow_lost > 0:
            print(f'    WARNING {tt}: no upstream emission factors; {flow_lost:.1f} MGD '
                  f'contributes zero emissions')

    tt_flow = df[ef.index].div(df['TT_IDENTIFIED'], axis=0).mul(df['FLOW_2022_MGD_FINAL'], axis=0)
    # direct (kg CO2e / day), factors in kg CO2e / m³
    for c in ['CH4', 'N2O', 'NC_CO2', 'NG_comb', 'NG_up', 'solids']:
        df[f'{c}_med'] = (tt_flow @ ef[f'{c}_50']) * MG_2_m3
    # electricity: elec_50 is kWh / MGD (no MG_2_m3 conversion), times the WWTP's kg CO2 / kWh
    df['elec_med'] = (tt_flow @ ef['elec_50']) * grid_carbon.loc[df['CWNS_NUM']].to_numpy()
    return df


def totals_from_df(df):
    """Totals dict (kt CO2e / year) from a df that already has *_med emission columns."""
    totals = {'n_facilities': len(df), 'total_flow_MGD': df['FLOW_2022_MGD_FINAL'].sum()}
    totals |= {f'{c}_ktyr': df[f'{c}_med'].sum() * KG_PER_DAY_TO_KT_PER_YEAR for c in COMPONENTS}
    totals['Scope1_ktyr'] = sum(totals[f'{c}_ktyr'] for c in ['CH4', 'N2O', 'NC_CO2', 'NG_comb', 'solids'])
    totals['Scope2_ktyr'] = totals['elec_ktyr']
    totals['Scope3_ktyr'] = totals['NG_up_ktyr']
    totals['total_ktyr'] = totals['Scope1_ktyr'] + totals['Scope2_ktyr'] + totals['Scope3_ktyr']
    return totals


def rank_trains(pool):
    """Trains in a pool, most common first, E-trains folded into their base."""
    counts = {}
    for tt in ALL_TT:
        n = int(pool[tt].sum())
        if n:
            base = E_TO_BASE.get(tt, tt)
            counts[base] = counts.get(base, 0) + n
    return sorted(counts, key=counts.get, reverse=True)


def apply_regional_fallback(df):
    """Fill in place a train for facilities whose reported processes identified none, as El Abbadi's
    notebook does (31% of their facilities, 17.6% of flow); returns how many were filled. Pools: the
    facility's flow-size bin (all CA is EPA Region 9), then every assigned facility. Their rules:
    1) prefer a train built on a process the facility reported (Point Loma, only AND + PRIMARY known,
    lands on O1E as theirs does; the bin mode G1E has 6x the N2O); 2) no E-trains; 3) never cross
    the nitrification cliff, applied both ways here."""
    size_bin = pd.cut(df['FLOW_2022_MGD_FINAL'], bins=[0, 2, 4, 7, 16, 46, 100, float('inf')],
                      labels=['<2', '2-4', '4-7', '7-16', '16-46', '46-100', '≥100'], right=False)
    assigned = df['TT_IDENTIFIED'] >= 1
    bin_ranks = {b: rank_trains(df[assigned & (size_bin == b)]) for b in size_bin.cat.categories}
    all_ranks = rank_trains(df[assigned])
    is_lagoon = df[LAGOON_TRAINS].sum(axis=1) > 0
    max_lagoon_flow = df.loc[assigned & is_lagoon, 'FLOW_2022_MGD_FINAL'].max()

    n_fallback = 0
    for idx, row in df[~assigned].iterrows():
        present = {up for up in KEY_UPS if row[up] == 1}
        nitrifies = bool(present & NITRIFYING_UPS)
        for ranks in [bin_ranks[size_bin[idx]], all_ranks]:
            # rule 3 first: never cross the nitrification cliff
            candidates = [tt for tt in ranks if (tt in NITRIFYING_TT) == nitrifies]
            # Lagoons carry the highest N2O factor (0.332) and are small-plant technology; the thin
            # large-flow bins put them on 100-260 MGD plants, so cap at the largest identified lagoon
            if row['FLOW_2022_MGD_FINAL'] > max_lagoon_flow:
                candidates = [tt for tt in candidates if tt not in LAGOON_TRAINS]
            # rule 1: prefer a train built on something the facility actually reported
            compatible = [tt for tt in candidates if TT_REQUIRES[tt] & present]
            choice = next(iter(compatible or candidates), None)
            if choice:
                df.at[idx, choice] = 1
                df.at[idx, 'TT_IDENTIFIED'] = 1
                n_fallback += 1
                break
    return n_fallback


def to_cwns_str(s):
    return str(int(float(s))).zfill(11)


def load_biogas_cwns_set():
    """CWNS_NUMs with confirmed biogas electricity generation from WEF + DOE databases."""
    werf = pd.read_csv(fetch_ghg_file('treatment_train_assignment/input_data/WERF_BIOGAS.csv'))
    elec_cols = ['Electricity_from_combustion-engine', 'Electricity_from_turbine',
                 'Electricity_from_microturbine', 'Electricity_from_fuelcell']
    werf_elec = werf[elec_cols].eq('yes').any(axis=1)
    werf_cwns = set(werf.loc[werf_elec, 'CWNS_NUM'].dropna().apply(to_cwns_str))

    doe = pd.read_csv(fetch_ghg_file('treatment_train_assignment/input_data/doe_chpdb-WWTP.csv'))
    doe_cwns = set(doe.loc[doe['BIOGAS_DOE_2022'] == 1, 'CWNS_NUM'].dropna().apply(to_cwns_str))

    combined = werf_cwns | doe_cwns
    print(f'  Biogas set: {len(werf_cwns)} WEF + {len(doe_cwns)} DOE = {len(combined)} unique '
          f'CWNS facilities')
    return combined


def load_epa_nitrification_set():
    """CWNS_NUMs with nitrification from EPA PRES_AMMONIA_REMOVAL (2024), read back from El Abbadi's
    UP_ID_NOTE since CWNS 2022 lacks it. CWNS shows no nitrification at any of the 318 common
    facilities; these 25 carry 220.8 of the 556.7 kt CO2e/yr N2O in El Abbadi's published CA total."""
    tt = pd.read_csv(fetch_ghg_file('treatment_train_assignment/output_data/tt_assignments_2022.csv'))
    key = tt['CWNS_NUM'].astype(str).str.removesuffix('.0').str.zfill(11)
    from_epa = tt['UP_ID_NOTE'].fillna('').str.contains('AMMONIA', case=False, regex=False)
    out = set(key[from_epa])
    print(f'  EPA nitrification set: {len(out)} CWNS facilities (PRES_AMMONIA_REMOVAL 2024)')
    return out


def build_werf_pivot(facility_rows, col_to_werf):
    """Binary WERF-code pivot, one row per Place ID, from PRESENT_STATUSES only for every source.
    Adds derived NIT and BIO-P, then clears lagoon codes where there is also secondary treatment."""
    # Present only: counting FUTURE put Hyperion's planned 2035 recycled-water MBR on a 215 MGD
    # pure-oxygen plant, moving it into N1, the one train with no emission factors
    status_cols = [c for c in facility_rows.columns if c in col_to_werf]
    records = []
    for _, row in facility_rows.iterrows():
        record = {'Place ID': row['Place ID']}
        for col in status_cols:
            if is_present(row[col]):
                record |= dict.fromkeys(col_to_werf[col], 1)
        records.append(record)
    pivot = pd.DataFrame(records).fillna(0)

    nit_cols = [c for c in ['BNIT', 'BNR', 'AS-BDENIT', 'AS-A2O'] if c in pivot.columns]
    pivot['NIT'] = (pivot[nit_cols].sum(axis=1) > 0).astype(int)
    # Bio-P from A2O only (bio-P by design). BNR ("Unspecified Nutrient Removal") credited 103 of 318
    # CA plants where El Abbadi's CWNS data has 1, pushing them into the ~6x-N2O G-series; most of CA
    # has no P limit. BNR still derives NIT above.
    pivot['BIO-P'] = (pivot['AS-A2O'] > 0).astype(int) if 'AS-A2O' in pivot.columns else 0

    secondary = ['AS', 'AS-A2O', 'AS-BDENIT', 'AS-EA', 'AS-OD', 'AS-P', 'AS-PUREO', 'AS-SA',
                 'AS-SBR', 'TF', 'TF-BF', 'TF-RBC', 'MBR-BNR']
    lagoons = ['LAGOON', 'LAGOON_AER', 'LAGOON_ANAER', 'LAGOON_FAC', 'STBL_POND']
    has_secondary = pivot[[c for c in secondary if c in pivot.columns]].sum(axis=1) > 0
    pivot.loc[has_secondary, [c for c in lagoons if c in pivot.columns]] = 0
    return pivot


def build_source(label, facility_rows, link, col_to_werf, external=None, cogen_proxy=False):
    """One re-derived source: WERF pivot, flow link, biogas/EPA flags, trains, regional fallback.
    external = (EPA nitrification set, WEF/DOE biogas set), the databases El Abbadi add."""
    df = link.merge(build_werf_pivot(facility_rows, col_to_werf), on='Place ID')
    # Permit E-train proxy (off for CWNS): digestion alone is not energy recovery, as biogas is often
    # flared; El Abbadi have recovery at 315 of 2644 AD plants (11.9%), so require gas utilisation too
    biogas = (df['BIOGAS_CWNS'] > 0) & (df['AND'] > 0) & cogen_proxy
    if external:
        epa_nit, wef_doe_biogas = external
        on_epa = df['CWNS_NUM'].isin(epa_nit)
        df.loc[on_epa, 'NIT'] = 1
        wef_doe = df['CWNS_NUM'].isin(wef_doe_biogas)
        print(f'  {label}: EPA nitrification on {int(on_epa.sum())} of {len(df)} facilities, '
              f'WEF/DOE biogas adds {int((wef_doe & ~biogas).sum())}')
        # a confirmed biogas plant implies a digester
        df['AND'] = ((df['AND'] > 0) | wef_doe).astype(int)
        biogas = biogas | wef_doe
    df['BIOGAS_EL'] = biogas.astype(int)
    df = assign_treatment_trains(df)
    df['LAGOON_UNCATEGORIZED'] = df[['LAGOON', 'STBL_POND']].max(axis=1)
    n_fallback = apply_regional_fallback(df)
    print(f'{label}: {len(df)} CWNS rows, {int((df["TT_IDENTIFIED"] >= 1).sum())} with TT '
          f'assignment ({n_fallback} via regional fallback)')
    return df


def stacked_bars(segments, figsize, width, ylabel):
    """One stacked bar per source in PLOT_ORDER; segments are (label, color, per-source values)."""
    x = np.arange(len(PLOT_ORDER))
    fig, ax = plt.subplots(figsize=figsize)
    bottoms = np.zeros(len(PLOT_ORDER))
    for label, color, values in segments:
        values = np.array(values)
        ax.bar(x, values, width, bottom=bottoms, label=label, color=color, edgecolor='black', linewidth=0.4)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_TICK_LABELS[s] for s in PLOT_ORDER], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def plot_comparison(results, n_label):
    segments = [(label, color, [results[s][f'{c}_ktyr'] for s in PLOT_ORDER])
                for c, (label, color) in COMPONENTS.items()]
    fig, ax = stacked_bars(segments, (6, 5), 0.5, f'kt CO₂e / year\n{n_label}')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=10, frameon=False,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    save_and_close(fig, FIGURES_DIR / 'figure_5')


def n2o_by_secondary(df, ef, label):
    """Return dict[secondary group → N2O kt CO2e/yr] for one source dataframe."""
    tt_flow = df[ef.index].div(df['TT_IDENTIFIED'], axis=0).mul(df['FLOW_2022_MGD_FINAL'], axis=0)
    groups = {}
    tt_n2o = {}
    for tt in ef.index:
        n2o = (tt_flow[tt] * ef.loc[tt, 'N2O_50'] * MG_2_m3 * KG_PER_DAY_TO_KT_PER_YEAR).sum()
        tt_n2o[tt] = n2o
        group = TT_SECONDARY.get(tt, NO_NUTRIENT_REMOVAL)
        groups[group] = groups.get(group, 0) + n2o
    top = sorted(tt_n2o.items(), key=lambda x: -x[1])[:8]
    top_str = ', '.join(f'{t}={v:.1f}' for t, v in top if v > 0)
    print(f'  [{label}] n={len(df)} rows, total N2O={sum(groups.values()):.1f} kt/yr | top TTs: {top_str}')
    return groups


def plot_n2o_breakdown(dfs, ef, n_label):
    print('N2O breakdown diagnostics:')
    breakdowns = {s: n2o_by_secondary(df, ef, label=s) for s, df in dfs.items()}
    present = [g for g in SECONDARY_GROUPS if any(breakdowns[s][g] > 0 for s in PLOT_ORDER)]
    segments = [(g, SECONDARY_GROUPS[g][0], [breakdowns[s][g] for s in PLOT_ORDER]) for g in present]
    fig, ax = stacked_bars(segments, (8.6, 5), 0.45, f'N₂O  kt CO₂e / year\n{n_label}')
    # Scale to the tallest stack, not the largest single segment
    y_top = max(sum(breakdowns[s][g] for g in present) for s in PLOT_ORDER)
    step = next(s for s in (25, 50, 100, 200, 250, 500, 1000, 2000) if y_top / s <= 6)
    ax.set_yticks(np.arange(0, np.ceil(y_top / step) * step + step / 2, step))
    legend_items = [(g, {'facecolor': SECONDARY_GROUPS[g][0]}) for g in reversed(present)]
    make_grouped_legend(ax, [{'header': 'Treatment Type', 'items': legend_items}], fontsize=10)
    fig.tight_layout()
    save_and_close(fig, FIGURES_DIR / 'figure_s5')


def pct_change(r, base, key):
    return (r[key] - base[key]) / base[key] * 100


def main():
    print('Loading MC emission factors...')
    ef = load_mc_ef()
    print(f'  Loaded EFs for {len(ef)} treatment trains')
    print('Loading grid carbon intensity...')
    grid_carbon = load_grid_carbon()

    # Common set: Place IDs in both the CWNS table and the permit output, as in figure_3
    print('\nBuilding common facility set (Place ID intersection: CWNS ∩ LLM)...')
    cwns_table = pd.read_csv(CWNS_TABLE_CSV, dtype=str)
    ciwqs = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str).rename(columns={'CWNS_ID': 'CWNS_NUM'})
    llm = load_llm_facility_table()
    in_cwns = ciwqs['CWNS_NUM'].isin(cwns_table['CWNS_ID'])
    common_pids = set(ciwqs.loc[in_cwns, 'Place ID']) & set(llm['Place ID'])
    print(f'  {len(common_pids)} common Place IDs')

    # Place ID → CWNS_NUM → 2022 flow; a CWNS facility counts for one Place ID only
    tt_published = pd.read_csv(fetch_ghg_file('GHG_accounting/input_data/tt_assignments_2022.csv'),
                               dtype={'CWNS_NUM': str})
    link = (ciwqs.loc[ciwqs['Place ID'].isin(common_pids), ['Place ID', 'CWNS_NUM']]
            .merge(tt_published[['CWNS_NUM', 'FLOW_2022_MGD_FINAL']], on='CWNS_NUM')
            .drop_duplicates(subset=['CWNS_NUM']))

    # El Abbadi & Feng's published trains (CWNS + WEF/DOE + EPA + manual checks); all have one
    tt_published['LAGOON_UNCATEGORIZED'] = tt_published[['LAGOON_OTHER', 'STBL_POND']].max(axis=1)
    published = link.merge(tt_published[['CWNS_NUM'] + ALL_TT], on='CWNS_NUM')
    print(f'Source manual (published): {len(published)} CWNS rows')

    cwns_rows, _ = build_cwns_facility_processes(cwns_table, target_facilities=common_pids)
    llm_rows = llm[llm['Place ID'].isin(common_pids)]
    external = (load_epa_nitrification_set(), load_biogas_cwns_set())
    werf_curated = load_cwns_col_to_werf()
    raw = {
        'cwns_external': build_source('Source 1', cwns_rows, link, werf_curated, external),
        'cwns_only': build_source('Source 2', cwns_rows, link, werf_curated),
        'published': published,
        'permits': build_source('Source 3', llm_rows, link, werf_curated, cogen_proxy=True),
        'permits_external': build_source('Source 4', llm_rows, link, werf_curated, external,
                                         cogen_proxy=True),
    }

    # 1:1 matching: every source keeps the CWNS facilities Source 3 assigned a train
    assigned_cwns = set(raw['permits'].loc[raw['permits']['TT_IDENTIFIED'] >= 1, 'CWNS_NUM'])
    print(f'\nFiltering every source to the {len(assigned_cwns)} CWNS facilities '
          f'where LLM assigned a TT...')
    print('\n── GHG calculation ──')
    ghg = {key: calc_ghg(df[df['CWNS_NUM'].isin(assigned_cwns)], ef, grid_carbon, CALC_LABELS[key])
           for key, df in raw.items()}

    # Enforce intersection across all sources
    cwns_sets = {key: set(df['CWNS_NUM']) for key, df in ghg.items()}
    common_cwns = set.intersection(*cwns_sets.values())
    print('\n── intersection ──')
    print('  ' + ' | '.join(f'{CALC_LABELS[key]}: {len(s)}' for key, s in cwns_sets.items()))
    print(f'  Final intersection: {len(common_cwns)} CWNS facilities')
    ghg = {key: df[df['CWNS_NUM'].isin(common_cwns)] for key, df in ghg.items()}
    results = {key: totals_from_df(df) for key, df in ghg.items()}

    print('GHG Emissions Source Comparison (for matched facilities)')
    rows = []
    for s, name in SOURCE_NAMES.items():
        r = results[s]
        rows.append({
            'Source': name,
            # only the published source imports El Abbadi & Feng's trains; the rest re-derive them
            'TT assignment': 'published' if s == 'published' else 're-derived',
            'N CWNS': r['n_facilities'],
            'Flow (MGD)': f"{r['total_flow_MGD']:.0f}",
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
            'per MGD': f"{r['total_ktyr'] / r['total_flow_MGD']:.3f}",
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

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
            print(f'  {name}: N2O {pct_change(r, base, "N2O_ktyr"):+.1f}%,  '
                  f'FNG {pct_change(r, base, "NG_comb_ktyr"):+.1f}%,  '
                  f'Total {pct_change(r, base, "total_ktyr"):+.1f}%')

    print('\n── EXPERIMENT: automatic WERF mapping (CSV-derived) vs curated JSON werf_codes ──')
    print('  (CWNS-based sources only, both compared against Source manual = El Abbadi & Feng published)')
    werf_automatic = load_cwns_col_to_werf(automatic=True)
    manual = results['published']
    for label, curated, source_external in [('Source 1', results['cwns_external'], external),
                                            ('Source 2', results['cwns_only'], None)]:
        auto_df = build_source(f'{label} (auto WERF)', cwns_rows, link, werf_automatic, source_external)
        auto_ghg = calc_ghg(auto_df[auto_df['CWNS_NUM'].isin(assigned_cwns)], ef, grid_carbon,
                            f'{label} (auto WERF)')
        automatic = totals_from_df(auto_ghg[auto_ghg['CWNS_NUM'].isin(common_cwns)])
        for mapping, r in [('curated werf_codes', curated), ('automatic WERF csv', automatic)]:
            print(f'  {label}, {mapping}: N2O {pct_change(r, manual, "N2O_ktyr"):+.1f}% vs manual,  '
                  f'Total {pct_change(r, manual, "total_ktyr"):+.1f}% vs manual  '
                  f'(Total={r["total_ktyr"]:.1f}, N2O={r["N2O_ktyr"]:.1f})')

    first = results[PLOT_ORDER[0]]
    n_label = f'n={first["n_facilities"]} facilities, {first["total_flow_MGD"]:.0f} MGD'
    plot_comparison(results, n_label)
    plot_n2o_breakdown(ghg, ef, n_label)


if __name__ == '__main__':
    main()
