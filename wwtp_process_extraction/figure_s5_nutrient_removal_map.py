import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from helpers.plotting import save_and_close
from helpers.utils import CIWQS_TO_CWNS_CSV
from figure_4_ca_ghg_comparison import (
    LLM_CSV, GHG_ROOT, TT_SECONDARY, PRESENT_STATUSES,
    build_werf_pivot, assign_treatment_trains, _fetch,
)

OUTPUT_DIR = SCRIPT_DIR / 'output'
CA_COUNTIES_URL = 'https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson'
US_STATES_URL = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
NON_CONTIGUOUS = {'Alaska', 'Hawaii', 'Puerto Rico'}  # excluded from the US map for framing; CONUS + DC only
NON_CONTIGUOUS_CODES = {'AK', 'HI', 'PR', 'GU', 'VI', 'AS', 'MP'}

# Nitrification-only (AS + Nitrification, i.e. E2/E2P) is excluded on purpose.
NR_LABELS = {'AS + Nitrif./Denitrif.', 'AS + BNR', 'BNR-MBR'}
NR_TT_CODES = [tt for tt, label in TT_SECONDARY.items() if label in NR_LABELS]


def add_has_nr(df):
    cols = [c for c in NR_TT_CODES if c in df.columns]
    df = df.copy()
    df['has_nr'] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1) > 0
    return df


def to_points_gdf(df, lat_col, lon_col):
    d = df.copy()
    d[lat_col] = pd.to_numeric(d[lat_col], errors='coerce')
    d[lon_col] = pd.to_numeric(d[lon_col], errors='coerce')
    d = d.dropna(subset=[lat_col, lon_col])
    return gpd.GeoDataFrame(
        d, geometry=[Point(lon, lat) for lon, lat in zip(d[lon_col], d[lat_col])],
        crs='EPSG:4326',
    )


def plot_nr_map(boundary, points, out_path, figsize=(7, 9)):
    fig, ax = plt.subplots(figsize=figsize)
    boundary.plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.4)

    nr_true = points[points['has_nr']]
    nr_false = points[~points['has_nr']]
    nr_false.plot(ax=ax, color='#444444', markersize=20, alpha=0.6, zorder=3)
    nr_true.plot(ax=ax, color='#2a9d8f', markersize=20, alpha=0.8, zorder=4)
    ax.set_axis_off()

    legend_patches = [
        mpatches.Patch(color='#2a9d8f', label=f'Nutrient removal ({len(nr_true)})'),
        mpatches.Patch(color='#444444', label=f'No nutrient removal ({len(nr_false)})'),
    ]
    ax.legend(handles=legend_patches, loc='lower left', frameon=False, fontsize=14)
    save_and_close(fig, out_path)
    print(f'Saved {out_path}: NR={len(nr_true)}, no NR={len(nr_false)}, total={len(points)}')


def build_llm_ca():
    """All CA LLM facilities (not restricted to the CWNS-intersection subset)."""
    llm = pd.read_csv(LLM_CSV, dtype=str)
    pivot, _ = build_werf_pivot(llm, {'PRESENT', 'FUTURE'})
    df = assign_treatment_trains(pivot)
    df = add_has_nr(df)

    ciwqs = pd.read_csv(CIWQS_TO_CWNS_CSV, dtype=str).drop_duplicates(subset='Place ID')
    df = df.merge(ciwqs[['Place ID', 'Latitude_CIWQS', 'Longitude_CIWQS']], on='Place ID', how='left')
    return to_points_gdf(df, 'Latitude_CIWQS', 'Longitude_CIWQS')


def build_el_abbadi(phys):
    """El Abbadi & Feng published TT assignments (national), with coordinates + state code."""
    tt = pd.read_csv(_fetch('GHG_accounting/input_data/tt_assignments_2022.csv'), dtype={'CWNS_NUM': str})
    tt = add_has_nr(tt)
    df = tt.merge(phys, left_on='CWNS_NUM', right_on='CWNS_ID', how='inner')
    return to_points_gdf(df, 'LATITUDE', 'LONGITUDE')


if __name__ == '__main__':
    ca_counties = gpd.read_file(CA_COUNTIES_URL).to_crs('EPSG:4326')
    us_states = gpd.read_file(US_STATES_URL).to_crs('EPSG:4326')
    us_states = us_states[~us_states['name'].isin(NON_CONTIGUOUS)]

    phys = pd.read_csv(SCRIPT_DIR / 'data/cwns/2022/PHYSICAL_LOCATION.csv', dtype=str) \
        .drop_duplicates(subset='CWNS_ID')

    gdf_llm = build_llm_ca()
    plot_nr_map(ca_counties, gdf_llm, OUTPUT_DIR / 'final' / 'figure_s5_nr_map_llm_ca')

    gdf_el_abbadi = build_el_abbadi(phys)
    gdf_el_abbadi_ca = gdf_el_abbadi[gdf_el_abbadi['STATE_CODE'] == 'CA']
    plot_nr_map(ca_counties, gdf_el_abbadi_ca, OUTPUT_DIR / 'final' / 'figure_s5_nr_map_el_abbadi_ca')

    gdf_el_abbadi_us = gdf_el_abbadi[~gdf_el_abbadi['STATE_CODE'].isin(NON_CONTIGUOUS_CODES)]
    plot_nr_map(us_states, gdf_el_abbadi_us, OUTPUT_DIR / 'final' / 'figure_s5_nr_map_el_abbadi_us', figsize=(12, 7))
