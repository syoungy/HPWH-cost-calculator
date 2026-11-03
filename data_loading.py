from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import calendar


DATA_DIR = Path(__file__).resolve().parent / "data"

ELECTRICITY_USAGE_FILE = "MI_water_heating_scenario9_monthly_average_kwh.xlsx"
GAS_USAGE_FILE = "MI_water_heating_scenario0_monthly_average_kwh.xlsx"
GT_ELECTRICITY_USAGE_FILE = "GT_water_heating_scenario9_monthly_average_kwh.xlsx"
GT_GAS_USAGE_FILE = "GT_water_heating_scenario0_monthly_average_kwh.xlsx"
GT_SPACE_ELECTRICITY_USAGE_FILE = (
    "GT_space_heating_scenario4_monthly_average_kwh.xlsx"
)
GT_SPACE_GAS_USAGE_FILE = (
    "GT_space_heating_scenario0_monthly_average_kwh.xlsx"
)

GT_SPACE_UPGRADE4_COMPONENTS = (
    "main_heat_pump",
    "main_fan_pump",
    "auxiliary_electric_heating",
    "auxiliary_fan_pump",
    "auxiliary_gas_heating",
)

GT_SPACE_UPGRADE0_COMPONENTS = (
    "main_gas_heating",
    "main_fan_pump",
)
PROVIDER_FILE = "MI_provider_county_with_utility_providers.xlsx"
ELECTRICITY_RATE_FILE = "electricity_rates_weekdays_202607.xlsx"
GAS_RATE_FILE = "gas_rates_weekdays_converted_to_kwh_202607.xlsx"

USAGE_HOUR_COLUMNS = [f"hour_{hour:02d}" for hour in range(24)]
RATE_HOUR_COLUMNS = [f"h{hour}" for hour in range(24)]

PROFILE_LABELS = {
    "year": "Annual average",
    "1": "January",
    "8": "August",
}

TARIFF_LABELS = {
    "TOU": "Time of Use",
    "TOD": "Time of Day",
    "OS": "Overnight Savings",
}

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
    gt_electricity_usage: pd.DataFrame
    gt_gas_usage: pd.DataFrame
    gt_space_electricity_usage: pd.DataFrame
    gt_space_gas_usage: pd.DataFrame
    provider_map: pd.DataFrame
    electricity_rates: pd.DataFrame
    gas_rates: pd.DataFrame
    excluded_building_ids: tuple[str, ...]


def _find_file(data_dir: Path, filename: str) -> Path:
    root = Path(__file__).resolve().parent
    candidates = [data_dir / filename, root / filename]
    for path in candidates:
        if path.exists():
            return path
    checked = " | ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Required file not found: {filename}. Checked: {checked}"
    )


def _clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _normalize_id(series: pd.Series) -> pd.Series:
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
    aliases = {key.casefold(): value for key, value in PROVIDER_ALIASES.items()}
    aliases.update({
        "holland board of public works": "Holland Board of Public Works",
        "traverse city light & power": "Traverse City Light & Power",
        "cloveland electric cooperative": "Cloverland Electric Cooperative",
        "cloverland electric cooperative": "Cloverland Electric Cooperative",
    })
    return _clean_text(series).map(lambda x: aliases.get(x.casefold(), x) if pd.notna(x) else x).astype("string")


def _normalize_tariff(series: pd.Series) -> pd.Series:
    return _clean_text(series).replace(TARIFF_ALIASES)


def _require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    source: str,
) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def _load_monthly_usage(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Monthly_Averages", dtype={"bldg_id": "string"})
    df.columns = [str(c).strip() for c in df.columns]
    required = ["bldg_id", "year", "month", "component", "energy_source", "n_days", *USAGE_HOUR_COLUMNS]
    _require_columns(df, required, source)
    df = df[required].copy()
    df["bldg_id"] = _normalize_id(df["bldg_id"])
    for col in ["component", "energy_source"]:
        df[col] = _clean_text(df[col])
    if df.empty or df[required].isna().any().any() or df[["bldg_id", "component", "energy_source"]].eq("").any().any():
        raise ValueError(f"{source}: empty data or missing required values.")
    for col in ["year", "month", "n_days", *USAGE_HOUR_COLUMNS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"{source}: invalid or negative values in {col}.")
    for col in ["year", "month", "n_days"]:
        if (df[col] != df[col].astype(int)).any():
            raise ValueError(f"{source}: {col} must contain integers.")
        df[col] = df[col].astype(int)
    if df.year.nunique() != 1 or not df.month.between(1, 12).all():
        raise ValueError(f"{source}: expected one usage year and months 1–12.")
    expected_days = [calendar.monthrange(y, m)[1] for y, m in zip(df.year, df.month)]
    if not np.array_equal(df.n_days, expected_days):
        raise ValueError(f"{source}: n_days does not match the source-year calendar.")
    keys = ["bldg_id", "month", "component"]
    if df.duplicated(keys).any():
        raise ValueError(f"{source}: duplicate house/month/component rows.")
    groups = df.groupby(["bldg_id", "component"])["month"].agg(set)
    if not groups.map(lambda x: x == set(range(1, 13))).all():
        raise ValueError(f"{source}: each house/component must have all 12 months.")
    return df.reset_index(drop=True)


def load_hourly_usage(path: Path, source: str) -> pd.DataFrame:
    df = _load_monthly_usage(path, source)
    expected_fuel = "electricity" if "scenario9" in path.name else "natural_gas"
    if set(df.component) != {"water_heating"} or set(df.energy_source) != {expected_fuel}:
        raise ValueError(f"{source}: expected water_heating / {expected_fuel}.")
    return df


def load_space_component_usage(path: Path, source: str) -> pd.DataFrame:
    df = _load_monthly_usage(path, source)
    expected = GT_SPACE_UPGRADE4_COMPONENTS if "scenario4" in path.name else GT_SPACE_UPGRADE0_COMPONENTS
    actual = df.groupby(["bldg_id", "month"])["component"].agg(set)
    if not actual.map(lambda x: x == set(expected)).all():
        raise ValueError(f"{source}: missing or unexpected heating components.")
    for component in expected:
        fuel = "natural_gas" if component in {"main_gas_heating", "auxiliary_gas_heating"} else "electricity"
        if set(df.loc[df.component == component, "energy_source"]) != {fuel}:
            raise ValueError(f"{source}: wrong fuel for {component}.")
    return df


def load_provider_map(
    path: Path,
    usage_ids: list[str],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    df = pd.read_excel(path)
    df.columns = [str(column).strip() for column in df.columns]

    id_column = next(
        (name for name in ["bldg_id", "MI_bldg_id"] if name in df.columns),
        None,
    )
    if id_column is None:
        raise ValueError(
            f"{path.name} must contain bldg_id or MI_bldg_id."
        )
    if id_column != "bldg_id":
        df = df.rename(columns={id_column: "bldg_id"})

    required = ["bldg_id", "in.county_name", "elec_provd", "gas_provd"]
    _require_columns(df, required, path.name)
    df = df[required].copy()

    df["bldg_id"] = _normalize_id(df["bldg_id"])
    df["in.county_name"] = _clean_text(df["in.county_name"])
    df["elec_provd"] = _normalize_provider(df["elec_provd"])
    df["gas_provd"] = _normalize_provider(df["gas_provd"])

    if df[required].isna().any().any():
        counts = df[required].isna().sum()
        raise ValueError(
            f"{path.name} contains blank required values: "
            f"{counts[counts > 0].to_dict()}"
        )

    duplicate = df.duplicated("bldg_id", keep=False)
    if duplicate.any():
        examples = df.loc[duplicate, "bldg_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Duplicate provider-map building IDs: {examples}")

    usage_id_set = set(map(str, usage_ids))
    provider_id_set = set(df["bldg_id"])

    extra = sorted(provider_id_set - usage_id_set)
    if extra:
        raise ValueError(
            "Provider map contains IDs absent from usage files. "
            f"Examples: {extra[:10]}"
        )

    excluded = tuple(sorted(usage_id_set - provider_id_set))
    return df.reset_index(drop=True), excluded


def load_electricity_rates(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    required = [
        "state", "elec_provd", "tariff", "year", "month",
        *RATE_HOUR_COLUMNS,
    ]
    frames: list[pd.DataFrame] = []
    source_order = 0

    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()

        # The supplied workbook uses merged cells and blank repeated labels.
        df["state"] = df["state"].ffill()
        df["elec_provd"] = df["elec_provd"].ffill()
        df["elec_provd"] = _normalize_provider(df["elec_provd"])
        df["tariff"] = _normalize_tariff(df["tariff"])

        # Fill tariff labels only within the same provider. A provider whose
        # entire tariff block is blank is treated as a single Standard Rate.
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

        if not np.isfinite(df[RATE_HOUR_COLUMNS].to_numpy(dtype=float)).all() or (df[RATE_HOUR_COLUMNS] < 0).any().any():
            raise ValueError(f"{path.name} / {sheet_name}: invalid hourly electricity rates.")
        if not df["month"].between(1, 12).all():
            raise ValueError(f"{path.name}: rate month must be between 1 and 12.")
        # A usable rate row must have all 24 hourly values.
        df = df.dropna(
            subset=[
                "state", "elec_provd", "tariff",
                *RATE_HOUR_COLUMNS,
            ]
        )

        df["source_sheet"] = sheet_name
        df["source_order"] = range(source_order, source_order + len(df))
        source_order += len(df)
        frames.append(df)

    if not frames:
        raise ValueError("No complete electricity-rate rows were found.")

    rates = pd.concat(frames, ignore_index=True)

    keys = ["state", "elec_provd", "tariff", "year", "month"]
    duplicate = rates.duplicated(keys, keep=False)
    if duplicate.any():
        examples = (
            rates.loc[duplicate, keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate electricity-rate rows: {examples}")

    return rates.reset_index(drop=True)


def load_gas_rates(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    required = ["state", "tariff", "year", "gas_provd", *RATE_HOUR_COLUMNS]
    frames: list[pd.DataFrame] = []
    source_order = 0

    for sheet_name, raw in sheets.items():
        raw.columns = [str(column).strip() for column in raw.columns]
        if not set(required).issubset(raw.columns):
            continue

        df = raw[required].copy()
        df["state"] = df["state"].ffill()
        df["tariff"] = df["tariff"].ffill()
        df["gas_provd"] = _normalize_provider(df["gas_provd"])
        df["tariff"] = _normalize_tariff(df["tariff"]).fillna("Standard Rate")

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"].notna() & df["gas_provd"].notna()].copy()

        df["state"] = _clean_text(df["state"]).str.upper()
        df["year"] = df["year"].astype(int)

        for column in RATE_HOUR_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        if not np.isfinite(df[RATE_HOUR_COLUMNS].to_numpy(dtype=float)).all() or (df[RATE_HOUR_COLUMNS] < 0).any().any():
            raise ValueError(f"{path.name} / {sheet_name}: invalid hourly gas rates.")
        df = df.dropna(
            subset=["state", "gas_provd", "tariff", *RATE_HOUR_COLUMNS]
        )
        df["source_sheet"] = sheet_name
        df["source_order"] = range(source_order, source_order + len(df))
        source_order += len(df)
        frames.append(df)

    if not frames:
        raise ValueError("No complete gas-rate rows were found.")

    rates = pd.concat(frames, ignore_index=True)

    keys = ["state", "gas_provd", "tariff", "year"]
    duplicate = rates.duplicated(keys, keep=False)
    if duplicate.any():
        examples = (
            rates.loc[duplicate, keys]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"Duplicate gas-rate rows: {examples}")

    return rates.reset_index(drop=True)



def _validate_dedicated_usage_pair(electricity_usage, gas_usage, sample_label):
    if set(electricity_usage.bldg_id) != set(gas_usage.bldg_id) or electricity_usage.empty:
        raise ValueError(f"{sample_label}: scenarios must have identical nonempty household sets.")
    if set(electricity_usage.year) != set(gas_usage.year):
        raise ValueError(f"{sample_label}: scenarios must use the same source year.")


def _validate_space_heating_component_pair(heat_pump_usage, gas_usage, sample_label):
    _validate_dedicated_usage_pair(heat_pump_usage, gas_usage, sample_label)
    excluded = {"191709", "254728", "29770", "340081", "505125"}
    ids = set(heat_pump_usage.bldg_id)
    if len(ids) != 70 or ids & excluded:
        raise ValueError(f"{sample_label}: use the matched 70-household workbooks with the five exclusions applied.")
    main_hp = heat_pump_usage.loc[heat_pump_usage.component == "main_heat_pump"]
    if (main_hp.groupby("bldg_id")[USAGE_HOUR_COLUMNS].sum().sum(axis=1) <= 0).any():
        raise ValueError(f"{sample_label}: a retained household has no main heat-pump use.")
    return heat_pump_usage, gas_usage, tuple(sorted(excluded))


def _empty_frame() -> pd.DataFrame:
    """Return an empty placeholder for data not needed by the active mode."""
    return pd.DataFrame()


def _load_mapped_water_heating_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Load and validate only the statewide mapped water-heating inputs."""
    electric_usage = load_hourly_usage(
        _find_file(data_dir, ELECTRICITY_USAGE_FILE),
        ELECTRICITY_USAGE_FILE,
    )
    gas_usage = load_hourly_usage(
        _find_file(data_dir, GAS_USAGE_FILE),
        GAS_USAGE_FILE,
    )

    electric_usage_ids = electric_usage["bldg_id"].drop_duplicates().tolist()
    provider_map, excluded = load_provider_map(
        _find_file(data_dir, PROVIDER_FILE),
        electric_usage_ids,
    )

    _validate_dedicated_usage_pair(electric_usage, gas_usage, "Michigan water heating")

    return electric_usage, gas_usage, provider_map, excluded


def _load_rate_inputs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the electricity and natural-gas tariff workbooks."""
    electricity_rates = load_electricity_rates(
        _find_file(data_dir, ELECTRICITY_RATE_FILE)
    )
    gas_rates = load_gas_rates(
        _find_file(data_dir, GAS_RATE_FILE)
    )
    return electricity_rates, gas_rates


def load_statewide_data(data_dir: Path = DATA_DIR) -> CalculatorData:
    """Load only inputs used by the All Michigan sample mode."""
    (
        electric_usage,
        gas_usage,
        provider_map,
        excluded,
    ) = _load_mapped_water_heating_inputs(data_dir)
    electricity_rates, gas_rates = _load_rate_inputs(data_dir)

    return CalculatorData(
        electricity_usage=electric_usage,
        gas_usage=gas_usage,
        gt_electricity_usage=_empty_frame(),
        gt_gas_usage=_empty_frame(),
        gt_space_electricity_usage=_empty_frame(),
        gt_space_gas_usage=_empty_frame(),
        provider_map=provider_map,
        electricity_rates=electricity_rates,
        gas_rates=gas_rates,
        excluded_building_ids=excluded,
    )


def load_county_data(data_dir: Path = DATA_DIR) -> CalculatorData:
    """Load water-heating inputs used by County / area scenario mode."""
    (
        electric_usage,
        gas_usage,
        provider_map,
        excluded,
    ) = _load_mapped_water_heating_inputs(data_dir)

    gt_electricity_usage = load_hourly_usage(
        _find_file(data_dir, GT_ELECTRICITY_USAGE_FILE),
        GT_ELECTRICITY_USAGE_FILE,
    )
    gt_gas_usage = load_hourly_usage(
        _find_file(data_dir, GT_GAS_USAGE_FILE),
        GT_GAS_USAGE_FILE,
    )
    _validate_dedicated_usage_pair(
        gt_electricity_usage,
        gt_gas_usage,
        "Grand Traverse dedicated sample",
    )

    electricity_rates, gas_rates = _load_rate_inputs(data_dir)

    return CalculatorData(
        electricity_usage=electric_usage,
        gas_usage=gas_usage,
        gt_electricity_usage=gt_electricity_usage,
        gt_gas_usage=gt_gas_usage,
        gt_space_electricity_usage=_empty_frame(),
        gt_space_gas_usage=_empty_frame(),
        provider_map=provider_map,
        electricity_rates=electricity_rates,
        gas_rates=gas_rates,
        excluded_building_ids=excluded,
    )


def load_space_heating_data(data_dir: Path = DATA_DIR) -> CalculatorData:
    """Load only the dedicated Traverse City space-heating inputs and rates."""
    gt_space_electricity_usage = load_space_component_usage(
        _find_file(data_dir, GT_SPACE_ELECTRICITY_USAGE_FILE),
        GT_SPACE_ELECTRICITY_USAGE_FILE,
    )
    gt_space_gas_usage = load_space_component_usage(
        _find_file(data_dir, GT_SPACE_GAS_USAGE_FILE),
        GT_SPACE_GAS_USAGE_FILE,
    )
    (
        gt_space_electricity_usage,
        gt_space_gas_usage,
        excluded_space_ids,
    ) = _validate_space_heating_component_pair(
        gt_space_electricity_usage,
        gt_space_gas_usage,
        "Grand Traverse space-heating sample",
    )

    electricity_rates, gas_rates = _load_rate_inputs(data_dir)

    return CalculatorData(
        electricity_usage=_empty_frame(),
        gas_usage=_empty_frame(),
        gt_electricity_usage=_empty_frame(),
        gt_gas_usage=_empty_frame(),
        gt_space_electricity_usage=gt_space_electricity_usage,
        gt_space_gas_usage=gt_space_gas_usage,
        provider_map=_empty_frame(),
        electricity_rates=electricity_rates,
        gas_rates=gas_rates,
        excluded_building_ids=excluded_space_ids,
    )


def load_calculator_data(data_dir: Path = DATA_DIR) -> CalculatorData:
    """Backward-compatible full loader for offline validation only.

    The Streamlit application uses the three mode-specific loaders above so
    that inactive analysis modes do not delay the current page.
    """
    county_data = load_county_data(data_dir)

    gt_space_electricity_usage = load_space_component_usage(
        _find_file(data_dir, GT_SPACE_ELECTRICITY_USAGE_FILE),
        GT_SPACE_ELECTRICITY_USAGE_FILE,
    )
    gt_space_gas_usage = load_space_component_usage(
        _find_file(data_dir, GT_SPACE_GAS_USAGE_FILE),
        GT_SPACE_GAS_USAGE_FILE,
    )
    (
        gt_space_electricity_usage,
        gt_space_gas_usage,
        excluded_space_ids,
    ) = _validate_space_heating_component_pair(
        gt_space_electricity_usage,
        gt_space_gas_usage,
        "Grand Traverse space-heating sample",
    )

    return CalculatorData(
        electricity_usage=county_data.electricity_usage,
        gas_usage=county_data.gas_usage,
        gt_electricity_usage=county_data.gt_electricity_usage,
        gt_gas_usage=county_data.gt_gas_usage,
        gt_space_electricity_usage=gt_space_electricity_usage,
        gt_space_gas_usage=gt_space_gas_usage,
        provider_map=county_data.provider_map,
        electricity_rates=county_data.electricity_rates,
        gas_rates=county_data.gas_rates,
        excluded_building_ids=tuple(
            sorted(
                set(county_data.excluded_building_ids)
                | set(excluded_space_ids)
            )
        ),
    )
