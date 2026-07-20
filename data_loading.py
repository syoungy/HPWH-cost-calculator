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
    "1": "January",
    "8": "August",
    "year": "Annual average",
}

# These aliases reconcile names already present in the supplied workbooks.
# Add more aliases here only when a source file uses a different spelling for
# the same legal utility.
PROVIDER_ALIASES = {
    "Consumers Energy": "Consumers Energy Company",
    "Cherryland Electric Co-op": "Cherryland Electric Cooperative",
}

TARIFF_ALIASES = {
    "SummerR": "Summer Rate",
}


@dataclass(frozen=True)
class CalculatorData:
    electricity_usage: pd.DataFrame
    gas_usage: pd.DataFrame
    provider_map: pd.DataFrame
    electricity_rates: pd.DataFrame
    gas_rates: pd.DataFrame
    excluded_building_ids: tuple[str, ...]


def _require_file(data_dir: Path, filename: str) -> Path:
    repository_root = Path(__file__).resolve().parent
    candidates = [data_dir / filename, repository_root / filename]

    for path in candidates:
        if path.exists():
            return path

    checked = " | ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Required file not found: {filename}. Checked: {checked}. "
        "Upload it either to data/ or beside app.py."
    )


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _normalize_building_id(series: pd.Series) -> pd.Series:
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


def _normalize_provider(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    return cleaned.replace(PROVIDER_ALIASES)


def _normalize_tariff(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    return cleaned.replace(TARIFF_ALIASES)


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
            f"Example row indices: {bad_rows}"
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


def load_provider_map(
    path: Path,
    building_ids_in_usage: list[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    df = pd.read_excel(path)
    df.columns = [str(column).strip() for column in df.columns]

    id_aliases = ["bldg_id", "MI_bldg_id"]
    id_column = next((name for name in id_aliases if name in df.columns), None)
    if id_column is None:
        raise ValueError(
            f"{path.name} must contain one of these ID columns: {id_aliases}."
        )
    if id_column != "bldg_id":
        df = df.rename(columns={id_column: "bldg_id"})

    required = ["bldg_id", "in.county_name", "elec_provd", "gas_provd"]
    _validate_columns(df, required, path.name)
    df = df[required].copy()

    df["bldg_id"] = _normalize_building_id(df["bldg_id"])
    df["in.county_name"] = _clean_text(df["in.county_name"])
    df["elec_provd"] = _normalize_provider(df["elec_provd"])
    df["gas_provd"] = _normalize_provider(df["gas_provd"])

    if df[required].isna().any().any():
        null_counts = df[required].isna().sum()
        null_counts = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"{path.name} contains blank required values: {null_counts}")

    duplicated = df.duplicated("bldg_id", keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, "bldg_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Provider file has duplicate bldg_id values: {examples}")

    usage_ids = set(map(str, building_ids_in_usage))
    provider_ids = set(df["bldg_id"])
    extra_ids = sorted(provider_ids - usage_ids)
    if extra_ids:
        raise ValueError(
            "Provider file contains building IDs absent from usage files. "
            f"Examples: {extra_ids[:10]}"
        )

    excluded_ids = tuple(sorted(usage_ids - provider_ids))
    return df.reset_index(drop=True), excluded_ids


def load_electricity_rates(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []
    required = [
        "state", "elec_provd", "tariff", "year", "month",
        *RATE_HOUR_COLUMNS,
    ]

    source_order = 0
    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()

        # Workbooks may use vertically merged cells for state/provider/tariff.
        df["state"] = df["state"].ffill()
        df["elec_provd"] = df["elec_provd"].ffill()

        df["elec_provd"] = _normalize_provider(df["elec_provd"])
        df["tariff"] = _normalize_tariff(df["tariff"])

        # Fill tariff only within a provider. Remaining blanks represent a
        # provider's single unnamed/basic plan.
        df["tariff"] = (
            df.groupby("elec_provd", dropna=False)["tariff"]
            .ffill()
            .fillna("Standard Rate")
        )

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df = df[df["year"].notna() & df["month"].notna()].copy()

        df["state"] = _clean_text(df["state"]).str.upper()
        df["year"] = df["year"].astype(int)
        df["month"] = df["month"].astype(int)

        for column in RATE_HOUR_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Drop incomplete placeholder rows, but keep every complete rate row.
        df = df.dropna(
            subset=["state", "elec_provd", "tariff", *RATE_HOUR_COLUMNS]
        )
        df["source_sheet"] = sheet_name
        df["source_order"] = range(source_order, source_order + len(df))
        source_order += len(df)
        frames.append(df)

    if not frames:
        raise ValueError(
            "No complete electricity-rate rows were found. Required columns are "
            "state, elec_provd, tariff, year, month, and h0-h23."
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

    return rates.reset_index(drop=True)


def load_gas_rates(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    frames: list[pd.DataFrame] = []
    required = ["state", "tariff", "year", "gas_provd", *RATE_HOUR_COLUMNS]

    source_order = 0
    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()
        df[["state", "tariff"]] = df[["state", "tariff"]].ffill()
        df["gas_provd"] = _normalize_provider(df["gas_provd"])
        df["tariff"] = _normalize_tariff(df["tariff"]).fillna("Standard Rate")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna() & df["gas_provd"].notna()].copy()

        df["state"] = _clean_text(df["state"]).str.upper()
        df["year"] = df["year"].astype(int)

        for column in RATE_HOUR_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["state", "gas_provd", *RATE_HOUR_COLUMNS])
        df["source_sheet"] = sheet_name
        df["source_order"] = range(source_order, source_order + len(df))
        source_order += len(df)
        frames.append(df)

    if not frames:
        raise ValueError(
            "No complete gas-rate rows were found. Required columns are "
            "state, tariff, year, gas_provd, and h0-h23."
        )

    rates = pd.concat(frames, ignore_index=True)
    duplicate_keys = ["state", "gas_provd", "tariff", "year"]
    duplicated = rates.duplicated(duplicate_keys, keep=False)
    if duplicated.any():
        examples = (
            rates.loc[duplicated, duplicate_keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Gas rates contain duplicate provider/tariff/year rows: "
            f"{examples}"
        )

    return rates.reset_index(drop=True)


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
    gas_keys = set(
        map(tuple, gas_usage[["bldg_id", "season"]].to_numpy())
    )
    if electric_keys != gas_keys:
        raise ValueError(
            "Electric and gas usage files do not contain identical "
            "bldg_id/season combinations."
        )

    building_ids = electricity_usage["bldg_id"].drop_duplicates().tolist()
    provider_map, excluded_ids = load_provider_map(
        _require_file(data_dir, PROVIDER_FILE),
        building_ids,
    )

    return CalculatorData(
        electricity_usage=electricity_usage,
        gas_usage=gas_usage,
        provider_map=provider_map,
        electricity_rates=load_electricity_rates(
            _require_file(data_dir, ELECTRICITY_RATE_FILE)
        ),
        gas_rates=load_gas_rates(
            _require_file(data_dir, GAS_RATE_FILE)
        ),
        excluded_building_ids=excluded_ids,
    )
