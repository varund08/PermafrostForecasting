"""Feature construction for active-layer-thickness (ALT) forecasting.

ALT for summer of year Y is paired with station conditions from the preceding
cold season (Sep of Y-1 through May of Y), which are known before the thaw
season starts. The station file has no Jun-Aug data, so this is the only
temporally valid pairing available.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

FEATURES = {
    "Snow depth": "snow_depth_m",
    "air temperature [°C]": "air_temp_c",
    "Ground surface temperature [°C]": "gst_c",
    "Ground temperature at 25cm [°C]": "gt25_c",
    "Ground temperature at 50cm [°C]": "gt50_c",
    "Ground temperature at 75cm [°C]": "gt75_c",
    "Ground temperature at 100cm [°C]": "gt100_c",
}
FEATURE_COLS = list(FEATURES.values())
FALL_MONTHS = {"Sep", "Oct", "Nov", "Dec"}
MIN_MONTHS = 6  # of the 9 possible Sep-May months


def load_alt() -> pd.DataFrame:
    a = pd.read_csv(DATA / "alt_cm.csv")
    a["site"] = a.groupby(["lat", "lon"]).ngroup()
    return a.rename(columns={"ALT": "alt_cm"})


def load_station_winters() -> pd.DataFrame:
    """One row per station x winter, averaged over Sep(Y-1)..May(Y)."""
    c = pd.read_csv(DATA / "conditions.csv")
    c.columns = [" ".join(col.split()) for col in c.columns]  # fix double spaces
    c = c.rename(columns=FEATURES)
    c["winter"] = c["Year"] + c["Month"].isin(FALL_MONTHS).astype(int)
    g = c.groupby(["Name", "winter"])
    w = g[FEATURE_COLS].mean()
    w["n_months"] = g.size()
    w[["st_lat", "st_lon"]] = g[["Latitude", "Longitude"]].first()
    return w[w.n_months >= MIN_MONTHS].reset_index()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(h))


def build_panel(power: float = 2.0) -> pd.DataFrame:
    """ALT site-year panel with inverse-distance-weighted winter features.

    Weights are computed per year over the stations that reported that
    winter, so features vary by both location and year.
    """
    alt = load_alt()
    win = load_station_winters()
    m = alt.merge(win, left_on="year", right_on="winter")
    m["dist_km"] = haversine_km(m.lat, m.lon, m.st_lat, m.st_lon)
    m["w"] = 1.0 / np.maximum(m.dist_km, 1.0) ** power
    m[FEATURE_COLS] = m[FEATURE_COLS].mul(m.w, axis=0)
    g = m.groupby(["site", "year"])
    out = g[FEATURE_COLS].sum().div(g.w.sum(), axis=0)
    out["n_stations"] = g.size()
    out["nearest_station_km"] = g.dist_km.min()
    panel = alt.merge(out.reset_index(), on=["site", "year"], how="left")
    return panel.sort_values(["site", "year"]).reset_index(drop=True)
