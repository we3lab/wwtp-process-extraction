import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
from shapely.geometry import Point
from helpers.plotting import save_and_close

DATE = "2026-5-25"
OUT_DIR = f"wwtp_process_extraction/output/{DATE}"

# nutrient removal leaf columns (from unitprocess_keywords.json top-level "Nutrient Removal")
with open("wwtp_process_extraction/data/unitprocess_keywords.json") as f:
    kw = json.load(f)

def get_leaves(d):
    leaves = []
    for k, v in d.items():
        if isinstance(v, dict):
            if "alt_names" in v:
                leaves.append(k)
            else:
                leaves.extend(get_leaves(v))
    return leaves

nr_cols = get_leaves(kw["Nutrient Removal"])

# load llm output and coords
llm = pd.read_csv(f"{OUT_DIR}/llm_unit_processes_by_facility.csv")
coords = pd.read_csv("wwtp_process_extraction/data/ciwqs_to_cwns.csv")

df = llm.merge(coords[["Place ID", "Latitude_CIWQS", "Longitude_CIWQS"]], on="Place ID", how="inner")
df = df.dropna(subset=["Latitude_CIWQS", "Longitude_CIWQS"])

# flag: any nutrient removal process present
present_cols = [c for c in nr_cols if c in df.columns]
df["has_nr"] = df[present_cols].isin(["PRESENT", "FUTURE", "PAST"]).any(axis=1)

gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(lon, lat) for lon, lat in zip(df["Longitude_CIWQS"], df["Latitude_CIWQS"])],
    crs="EPSG:4326",
)

# california boundary
ca = gpd.read_file("https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/california-counties.geojson")
ca = ca.to_crs("EPSG:4326")

fig, ax = plt.subplots(figsize=(7, 9))
ca.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.4)

nr_true = gdf[gdf["has_nr"]]
nr_false = gdf[~gdf["has_nr"]]

nr_false.plot(ax=ax, color="#444444", markersize=20, alpha=0.6, zorder=3)
nr_true.plot(ax=ax, color="#2a9d8f", markersize=28, alpha=0.8, zorder=4)

ax.set_axis_off()

legend_patches = [
    mpatches.Patch(color="#2a9d8f", label=f"Nutrient removal ({nr_true.shape[0]})"),
    mpatches.Patch(color="#444444", label=f"No nutrient removal ({nr_false.shape[0]})"),
]
ax.legend(handles=legend_patches, loc="lower left", frameon=False, fontsize=14)

save_and_close(fig, f"{OUT_DIR}/final/si_map_nutrient_removal.png")
print(f"Saved to {OUT_DIR}/final/si_map_nutrient_removal.png")
print(f"  NR: {nr_true.shape[0]}, no NR: {nr_false.shape[0]}, total: {gdf.shape[0]}")
