from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

ELECTRICITY_USAGE_FILE = "MI_housesample_elec_hourly_average_kwh.xlsx"
GAS_USAGE_FILE = "MI_housesample_gas_hourly_average_kwh.xlsx"
PROVIDER_FILE = "MI_provider_county_with_utility_providers.xlsx"
ELECTRICITY_RATE_FILE = "electricity_rates_weekdays_202607.xlsx"
GAS_RATE_FILE = "gas_rates_weekdays_converted_to_kwh_202607.xlsx"

USAGE_HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(24)]
RATE_HOUR_COLUMNS = [f"h{hour}" for hour in range(24)]

TARIFF_LABELS = {
    "TOU": "Time of Use",
    "TOD": "Time of Day",
    "OS": "Overnight Savings",
}

PROFILE_LABELS = {
    "1": "January hourly average",
    "8": "August hourly average",
    "year": "Annual hourly average",
}


@dataclass(frozen=True)
class CalculatorData:
    electricity_usage: pd.DataFrame
    gas_usage: pd.DataFrame
    provider_map: pd.DataFrame
    electricity_rates: pd.DataFrame
    gas_rates: pd.DataFrame
    provider_mapping_warning: str | None


def _require_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}. "
            "Place the five Excel files inside the repository's data/ folder."
        )
    return path


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _normalize_building_id(series: pd.Series) -> pd.Series:
    # Excel sometimes reads identifiers such as 4644 as 4644.0.
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _normalize_profile(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"year", "annual", "annual average"}:
        return "year"

    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _validate_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    source_name: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )


def load_hourly_usage(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"bldg_id": "string"})
    df.columns = [str(column).strip() for column in df.columns]

    required = ["bldg_id", "season", *USAGE_HOUR_COLUMNS]
    _validate_columns(df, required, source_name)

    df = df[required].copy()
    df["bldg_id"] = _normalize_building_id(df["bldg_id"])
    df["season"] = df["season"].map(_normalize_profile)

    for column in USAGE_HOUR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[USAGE_HOUR_COLUMNS].isna().any().any():
        bad_rows = df.index[df[USAGE_HOUR_COLUMNS].isna().any(axis=1)].tolist()[:10]
        raise ValueError(
            f"{source_name} contains missing or nonnumeric hourly usage values. "
            f"Example Excel data-row indices: {bad_rows}"
        )

    duplicated = df.duplicated(["bldg_id", "season"], keep=False)
    if duplicated.any():
        examples = (
            df.loc[duplicated, ["bldg_id", "season"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{source_name} has duplicate bldg_id/season combinations: {examples}"
        )

    return df.reset_index(drop=True)


def load_electricity_rates(path: Path) -> pd.DataFrame:
    # Read every sheet so that future sheets can be added without code changes.
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []

    required = [
        "state", "elec_provd", "tariff", "year", "month",
        *RATE_HOUR_COLUMNS,
    ]

    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

        # This removes title/annotation rows such as the row containing
        # "24(0), 1, 2, ... 23" underneath the main header.
        df = df[df["year"].notna() & df["month"].notna()].copy()

        df["state"] = _clean_text(df["state"]).str.upper()
        df["elec_provd"] = _clean_text(df["elec_provd"])
        df["tariff"] = _clean_text(df["tariff"]).str.upper()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        for column in RATE_HOUR_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(
            subset=["state", "elec_provd", "tariff", *RATE_HOUR_COLUMNS]
        )
        df["source_sheet"] = sheet_name
        frames.append(df)

    if not frames:
        raise ValueError(
            "No valid electricity-rate rows were found. "
            "The workbook must contain state, elec_provd, tariff, year, month, "
            "and h0-h23 columns."
        )

    rates = pd.concat(frames, ignore_index=True)

    duplicate_keys = ["state", "elec_provd", "tariff", "year", "month"]
    duplicated = rates.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = (
            rates.loc[duplicated, duplicate_keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Electricity rates contain duplicate provider/tariff/year/month rows: "
            f"{examples}"
        )

    return rates.sort_values(duplicate_keys).reset_index(drop=True)


def load_gas_rates(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []
    required = ["state", "tariff", "year", "gas_provd", *RATE_HOUR_COLUMNS]

    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()

        # The supplied workbook uses vertically merged cells, which pandas
        # reads as blanks below the first provider. Forward-fill only the
        # shared grouping fields.
        df[["state", "tariff"]] = df[["state", "tariff"]].ffill()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna() & df["gas_provd"].notna()].copy()

        df["state"] = _clean_text(df["state"]).str.upper()
        df["tariff"] = _clean_text(df["tariff"]).str.upper()
        df["gas_provd"] = _clean_text(df["gas_provd"])
        df["year"] = df["year"].astype(int)

        for column in RATE_HOUR_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(
            subset=["state", "gas_provd", *RATE_HOUR_COLUMNS]
        )
        df["source_sheet"] = sheet_name
        frames.append(df)

    if not frames:
        raise ValueError(
            "No valid gas-rate rows were found. "
            "The workbook must contain state, tariff, year, gas_provd, "
            "and h0-h23 columns."
        )

    rates = pd.concat(frames, ignore_index=True)
    duplicate_keys = ["state", "gas_provd", "year"]
    duplicated = rates.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = (
            rates.loc[duplicated, duplicate_keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Gas rates contain duplicate state/provider/year rows: "
            f"{examples}"
        )

    return rates.sort_values(duplicate_keys).reset_index(drop=True)


def load_provider_map(
    path: Path,
    building_ids_in_usage_order: list[str],
) -> tuple[pd.DataFrame, str | None]:
    df = pd.read_excel(path)
    df.columns = [str(column).strip() for column in df.columns]

    # Accept either the standardized ID name or the current source-file name.
    id_aliases = ["bldg_id", "MI_bldg_id"]
    id_column = next((name for name in id_aliases if name in df.columns), None)
    if id_column is None:
        raise ValueError(
            f"{path.name} must contain one of these ID columns: {id_aliases}. "
            "Provider records are not matched to buildings by row order."
        )

    if id_column != "bldg_id":
        df = df.rename(columns={id_column: "bldg_id"})

    required = ["bldg_id", "in.county_name", "elec_provd", "gas_provd"]
    _validate_columns(df, required, path.name)
    df = df[required].copy()

    df["bldg_id"] = _normalize_building_id(df["bldg_id"])
    df["in.county_name"] = _clean_text(df["in.county_name"])
    df["elec_provd"] = _clean_text(df["elec_provd"])
    df["gas_provd"] = _clean_text(df["gas_provd"])

    if df[required].isna().any().any():
        null_counts = df[required].isna().sum()
        null_counts = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"{path.name} contains blank required values: {null_counts}"
        )

    duplicated = df.duplicated("bldg_id", keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "bldg_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Provider file has duplicate bldg_id values: {examples}")

    usage_ids = set(map(str, building_ids_in_usage_order))
    provider_ids = set(df["bldg_id"])

    extra_ids = sorted(provider_ids - usage_ids)
    if extra_ids:
        raise ValueError(
            "The provider file contains building IDs that do not occur in the "
            f"usage files. Examples: {extra_ids[:10]}"
        )

    missing_ids = sorted(usage_ids - provider_ids)
    warning = None
    if missing_ids:
        warning = (
            f"{len(missing_ids)} building(s) in the hourly-usage files have no "
            "provider mapping and are excluded from the building selector: "
            + ", ".join(missing_ids[:10])
        )

    return df.sort_values("bldg_id").reset_index(drop=True), warning

def load_calculator_data(data_dir: Path = DATA_DIR) -> CalculatorData:
    electricity_usage = load_hourly_usage(
        _require_file(data_dir, ELECTRICITY_USAGE_FILE),
        ELECTRICITY_USAGE_FILE,
    )
    gas_usage = load_hourly_usage(
        _require_file(data_dir, GAS_USAGE_FILE),
        GAS_USAGE_FILE,
    )

    electric_keys = set(
        map(tuple, electricity_usage[["bldg_id", "season"]].to_numpy())
    )
    gas_keys = set(map(tuple, gas_usage[["bldg_id", "season"]].to_numpy()))
    if electric_keys != gas_keys:
        only_electric = sorted(electric_keys - gas_keys)[:10]
        only_gas = sorted(gas_keys - electric_keys)[:10]
        raise ValueError(
            "Electric and gas usage files do not contain identical "
            "bldg_id/season combinations. "
            f"Only in electricity: {only_electric}; only in gas: {only_gas}"
        )

    building_ids = electricity_usage["bldg_id"].drop_duplicates().tolist()
    provider_map, provider_warning = load_provider_map(
        _require_file(data_dir, PROVIDER_FILE),
        building_ids,
    )

    electricity_rates = load_electricity_rates(
        _require_file(data_dir, ELECTRICITY_RATE_FILE)
    )
    gas_rates = load_gas_rates(
        _require_file(data_dir, GAS_RATE_FILE)
    )

    return CalculatorData(
        electricity_usage=electricity_usage,
        gas_usage=gas_usage,
        provider_map=provider_map,
        electricity_rates=electricity_rates,
        gas_rates=gas_rates,
        provider_mapping_warning=provider_warning,
    )


def get_usage_vector(
    usage: pd.DataFrame,
    building_id: str,
    profile: str,
) -> list[float]:
    matched = usage[
        (usage["bldg_id"] == str(building_id))
        & (usage["season"] == str(profile))
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one usage row for bldg_id={building_id}, "
            f"profile={profile}; found {len(matched)}."
        )
    return matched.iloc[0][USAGE_HOUR_COLUMNS].astype(float).tolist()


def get_electricity_rate_vector(
    rates: pd.DataFrame,
    state: str,
    provider: str,
    tariff: str,
    year: int,
    month: int,
) -> list[float]:
    matched = rates[
        (rates["state"] == state)
        & (rates["elec_provd"] == provider)
        & (rates["tariff"] == tariff)
        & (rates["year"] == int(year))
        & (rates["month"] == int(month))
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one electricity-rate row for "
            f"{state} / {provider} / {tariff} / {year}-{month:02d}; "
            f"found {len(matched)}."
        )
    return matched.iloc[0][RATE_HOUR_COLUMNS].astype(float).tolist()


def get_gas_rate_vector(
    rates: pd.DataFrame,
    state: str,
    provider: str,
    year: int,
) -> list[float]:
    matched = rates[
        (rates["state"] == state)
        & (rates["gas_provd"] == provider)
        & (rates["year"] == int(year))
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected exactly one gas-rate row for "
            f"{state} / {provider} / {year}; found {len(matched)}."
        )
    return matched.iloc[0][RATE_HOUR_COLUMNS].astype(float).tolist()
