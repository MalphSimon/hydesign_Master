"""
SitePlotter.py

Plot study-case sites on a simplified Europe map.

Requirements:
- Natural Earth 10m Admin 0 Countries shapefile stored at:
  ../../ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp

Output:
- study_cases_map.png
"""

import os

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# GDAL_DATA guard (helps on some Windows/conda installs)
# ---------------------------------------------------------------------
if "GDAL_DATA" not in os.environ:
    gdal_guess = os.path.join(
        os.environ.get("CONDA_PREFIX", ""), "Library", "share", "gdal"
    )
    if os.path.isdir(gdal_guess):
        os.environ["GDAL_DATA"] = gdal_guess


# ---------------------------------------------------------------------
# Load countries shapefile (Natural Earth 10m)
# ---------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
countries_path = os.path.join(
    base_dir, "ne_10m_admin_0_countries", "ne_10m_admin_0_countries.shp"
)

if not os.path.isfile(countries_path):
    raise FileNotFoundError(
        "Countries shapefile not found.\n"
        "Expected: ../../ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp\n"
        f"Resolved path: {countries_path}"
    )

countries = gpd.read_file(countries_path)
countries = countries[countries.geometry.notna() & ~countries.geometry.is_empty].copy()

# Pick the most likely "country name" column across Natural Earth variants
name_col_candidates = ["ADMIN", "NAME_EN", "NAME", "SOVEREIGNT", "NAME_LONG"]
name_col = next((c for c in name_col_candidates if c in countries.columns), None)
if name_col is None:
    raise ValueError(
        f"Could not find a country name column. Available columns: {list(countries.columns)}"
    )

# ---------------------------------------------------------------------
# Country whitelist: “within the specified border”
# (Western/Central/Southern Europe up to PL–SK–HU–RO–BG, excluding Nordics)
# ---------------------------------------------------------------------
keep_countries = {
    # Iberia + Islands (add/remove as you prefer)
    "Portugal",
    "Spain",

    # France + Benelux
    "France",
    "Belgium",
    "Netherlands",
    "Luxembourg",

    # DACH
    "Germany",
    "Switzerland",
    "Austria",

    # Italy + microstates (optional)
    "Italy",
    "San Marino",
    "Vatican",

    # UK + Ireland + Denmark
    "United Kingdom",
    "Ireland",
    "Denmark",
    "Norway",
    "Sweden",
    "Finland",
    "Latvia",
    "Lithuania",
    "Estonia",
    "Belarus",
    "Ukraine",
    "Moldova",

    # “eastern cutoff line” countries you mentioned
    "Poland",
    "Slovakia",
    "Hungary",
    "Romania",
    "Bulgaria",

    # Countries south/west of that line that are typically inside the same “block”
    "Czechia",          # Natural Earth uses "Czechia" (not "Czech Republic") in many versions
    "Slovenia",
    "Croatia",
    "Greece",

    # Optional (include if you want them in the map)
    # "Malta",
    # "Cyprus",
}

# Explicit exclusions (you asked for these)
exclude_countries = {"Iceland"}

selected = countries[countries[name_col].isin(keep_countries)].copy()
selected = selected[~selected[name_col].isin(exclude_countries)].copy()

if selected.empty:
    raise ValueError(
        "After filtering, no countries remain. "
        f"Check the country names in your shapefile column '{name_col}'."
    )

# ---------------------------------------------------------------------
# Map extent (same intent as before)
# ---------------------------------------------------------------------
xmin, xmax = -12, 40
ymin, ymax = 35, 72

selected = selected.cx[xmin:xmax, ymin:ymax].copy()

# ---------------------------------------------------------------------
# Sites (replace with final coordinates)
# ---------------------------------------------------------------------
sites = pd.DataFrame(
    {
        "site": [
            "Nordsøen Midt (DK)",
            "Thetys (NL)",
            "Sud-Atlantique I (FR)",
            "Golfe du Lion Est (FR)",
            "Vestavind D (NO)",
            "Silicy South (IT)",
        ],
        "lat": [56.0, 52.54, 45.89, 43.01, 60.33, 37.15],
        "lon": [7.3, 4.19, -1.78, 4.29, 4.38, 13.2],
    }
)

sites_gdf = gpd.GeoDataFrame(
    sites,
    geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
    crs="EPSG:4326",
)

# ---------------------------------------------------------------------
# Plot (everything else remains white because we only plot "selected")
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor("white")

selected.plot(
    ax=ax,
    color="#2f6b4f",      # green
    edgecolor="white",
    linewidth=0.8,
    zorder=2,
)

sites_gdf.plot(
    ax=ax,
    color="#d62828",
    markersize=35,
    edgecolor="white",
    linewidth=0.6,
    zorder=3,
)

ax.set_axis_off()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig("study_cases_map.png", dpi=300)
plt.show()