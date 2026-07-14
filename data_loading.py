"""
Data loading for the HPWH vs Gas WH cost calculator.

Reads the actual project data files in ./data:

  1. electricity_rates_weekdays_202607.xlsx
       sheet 'TOU', columns: state, tariff, year, month, h0..h23  ($/kWh)
       - first row is a comment/legend row -> dropped
       - state/tariff/year use merged cells -> forward-filled
       - tariffs: TOU (Time of Day 3-7pm), TOD (Time of Day 11am-7pm),
                  OS (Overnight Savers), each x 12 months

  2. gas_rates_weekdays_converted_to_kwh_202607.xlsx
       one row, columns h0..h23 in $/kWh (flat rate, converted from $/therm)

  3. MI_upgrade0_downsize.xlsx  (extracted from ResStock MI_upgrade0, 76MB -> 557KB)
       columns: bldg_id, in.water_heater_efficiency,
                out.electricity.hot_water.energy_consumption..kwh,
                out.natural_gas.hot_water.energy_consumption..kwh
       - the first column header carries a leaked tar-header string from the
         original file; it is renamed to 'bldg_id' on load

NOTE: the water-heater type filter uses 'in.water_heater_efficiency'
(values like 'Electric Heat Pump, 50 gal, 3.45 UEF', 'Natural Gas Standard'),
because 'in.water_heater_fuel' only contains the fuel name.
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ELEC_RATES_FILE = os.path.join(DATA_DIR, "electricity_rates_weekdays_202607.xlsx")
GAS_RATES_FILE = os.path.join(DATA_DIR, "gas_rates_weekdays_converted_to_kwh_202607.xlsx")
CONSUMPTION_FILE = os.path.join(DATA_DIR, "MI_upgrade0_downsize.xlsx")

HOUR_COLS = [f"h{i}" for i in range(24)]

# Display names for the tariff codes in the rates file
TARIFF_LABELS = {
    "TOU": "Time of Day 3 p.m. - 7 p.m. (TOU)",
    "TOD": "Time of Day 11 a.m. - 7 p.m. (TOD)",
    "OS": "Overnight Savers (OS)",
}

# Water-heater type filters (substring match on in.water_heater_efficiency)
WH_TYPE_COL = "in.water_heater_efficiency"
HPWH_PATTERN = "Electric Heat Pump"
GAS_STD_PATTERN = "Natural Gas Standard"

HPWH_ENERGY_COL = "out.electricity.hot_water.energy_consumption..kwh"
GAS_ENERGY_COL = "out.natural_gas.hot_water.energy_consumption..kwh"

# ---------------------------------------------------------------------------
# Hourly usage fraction f_h (h = 0..23), base-schedules-simple.xml,
# 'schedule by water fixtures' (weekday == weekend). Normalized to sum to 1.
# ---------------------------------------------------------------------------
_RAW_FRACTIONS = [
    0.012, 0.006, 0.004, 0.005, 0.010, 0.034, 0.078, 0.087,
    0.080, 0.067, 0.056, 0.047, 0.040, 0.035, 0.033, 0.031,
    0.039, 0.051, 0.060, 0.060, 0.055, 0.048, 0.038, 0.026,
]
_S = sum(_RAW_FRACTIONS)
USAGE_FRACTIONS = [f / _S for f in _RAW_FRACTIONS]


def load_electricity_rates() -> pd.DataFrame:
    """Return tidy rates: columns [tariff, year, month, h0..h23] in $/kWh."""
    df = pd.read_excel(ELEC_RATES_FILE, sheet_name=0)
    # drop the legend row (h0 is text like '24(0)')
    df = df[pd.to_numeric(df["h0"], errors="coerce").notna()].copy()
    # merged cells -> forward-fill block labels
    df[["state", "tariff", "year"]] = df[["state", "tariff", "year"]].ffill()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df[HOUR_COLS] = df[HOUR_COLS].astype(float)
    return df[["state", "tariff", "year", "month"] + HOUR_COLS].reset_index(drop=True)


def load_gas_rate() -> float:
    """Return the flat gas rate r_g in $/kWh."""
    df = pd.read_excel(GAS_RATES_FILE, sheet_name=0)
    row = df.iloc[0]
    vals = row[HOUR_COLS].astype(float)
    if vals.nunique() != 1:
        raise ValueError("Gas rate file is expected to be flat across all 24 hours.")
    return float(vals.iloc[0])


def load_daily_consumption() -> dict:
    """
    Compute daily consumption from the slim ResStock extract.

    Procedure:
      #1 filter rows by water heater type (in.water_heater_efficiency)
      #2 average the annual consumption column (kWh/year)
      #3 divide by 365 -> daily consumption (kWh/day)
    """
    df = pd.read_excel(CONSUMPTION_FILE)
    df = df.rename(columns={df.columns[0]: "bldg_id"})  # fix leaked tar-header name
    out = {}
    for key, pattern, col in [
        ("hpwh", HPWH_PATTERN, HPWH_ENERGY_COL),
        ("gas", GAS_STD_PATTERN, GAS_ENERGY_COL),
    ]:
        mask = df[WH_TYPE_COL].astype(str).str.contains(pattern, case=False, na=False)
        sub = df.loc[mask, col].dropna()
        annual = float(sub.mean())
        out[key] = {
            "n_rows": int(sub.shape[0]),
            "annual_avg_kwh": annual,
            "daily_kwh": annual / 365.0,
        }
    return out


def get_hourly_rates(rates: pd.DataFrame, tariff: str, year: int, month: int) -> list[float]:
    """Return the 24-hour rate vector r_{p,m,h} in $/kWh for one tariff/year/month."""
    row = rates[(rates.tariff == tariff) & (rates.year == year) & (rates.month == month)]
    if row.empty:
        raise ValueError(f"No rates for tariff={tariff}, year={year}, month={month}")
    return row.iloc[0][HOUR_COLS].astype(float).tolist()
