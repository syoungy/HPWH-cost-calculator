from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"

ELECTRICITY_USAGE_FILE = "MI_housesample_elec_hourly_average_kwh.xlsx"
GAS_USAGE_FILE = "MI_housesample_gas_hourly_average_kwh.xlsx"
GT_ELECTRICITY_USAGE_FILE = "MI_housesample_gt_elec_hourly_average_kwh.xlsx"
GT_GAS_USAGE_FILE = "MI_housesample_gt_gas_hourly_average_kwh.xlsx"
GT_SPACE_ELECTRICITY_USAGE_FILE = (
    "Grand Traverse_Space heating_upgrade4_hourly_average_kwh.xlsx"
)
GT_SPACE_GAS_USAGE_FILE = (
    "Grand Traverse_Space heating_upgrade0_hourly_average_kwh.xlsx"
)

GT_SPACE_UPGRADE4_COMPONENTS = (
    "(Main) HP heating",
    "(Main) additional_fan_pump",
    "(Aux_electricity) back up",
    "(Aux_electricity) additional_fan_pump",
    "(Aux_natural gas) back up",
)

GT_SPACE_UPGRADE0_COMPONENTS = (
    "(Main) Natural Gas heating",
    "(additional) blower_fan",
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
    return _clean_text(series).replace(PROVIDER_ALIASES)


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


def load_hourly_usage(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype={"bldg_id": "string"})
    df.columns = [str(column).strip() for column in df.columns]

    required = ["bldg_id", "season", *USAGE_HOUR_COLUMNS]
    _require_columns(df, required, source)
    df = df[required].copy()

    df["bldg_id"] = _normalize_id(df["bldg_id"])
    df["season"] = df["season"].map(_normalize_profile)

    for column in USAGE_HOUR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[USAGE_HOUR_COLUMNS].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{source} contains blank or nonnumeric hourly usage values. "
            f"Example row indices: {df.index[invalid].tolist()[:10]}"
        )

    duplicate = df.duplicated(["bldg_id", "season"], keep=False)
    if duplicate.any():
        examples = (
            df.loc[duplicate, ["bldg_id", "season"]]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{source} contains duplicate building/profile rows: {examples}"
        )

    return df.reset_index(drop=True)


def load_space_component_usage(path: Path, source: str) -> pd.DataFrame:
    """Load a space-heating component workbook.

    The source workbooks use one row per building / period / component and
    contain either ``season`` or ``season(month)`` as the period column.
    """
    df = pd.read_excel(path, dtype={"bldg_id": "string"})
    df.columns = [str(column).strip() for column in df.columns]

    if "season" not in df.columns and "season(month)" in df.columns:
        df = df.rename(columns={"season(month)": "season"})

    required = [
        "bldg_id",
        "season",
        "component",
        *USAGE_HOUR_COLUMNS,
    ]
    _require_columns(df, required, source)
    df = df[required].copy()

    df["bldg_id"] = _normalize_id(df["bldg_id"])
    df["season"] = df["season"].map(_normalize_profile)
    df["component"] = _clean_text(df["component"])

    for column in USAGE_HOUR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[USAGE_HOUR_COLUMNS].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{source} contains blank or nonnumeric hourly usage values. "
            f"Example row indices: {df.index[invalid].tolist()[:10]}"
        )

    duplicate = df.duplicated(
        ["bldg_id", "season", "component"],
        keep=False,
    )
    if duplicate.any():
        examples = (
            df.loc[
                duplicate,
                ["bldg_id", "season", "component"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            f"{source} contains duplicate building/profile/component rows: "
            f"{examples}"
        )

    return df.reset_index(drop=True)

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



def _validate_dedicated_usage_pair(
    electricity_usage: pd.DataFrame,
    gas_usage: pd.DataFrame,
    sample_label: str,
) -> None:
    required_profiles = {"1", "8", "year"}
    electricity_ids = set(electricity_usage["bldg_id"].astype(str))
    gas_ids = set(gas_usage["bldg_id"].astype(str))

    if not electricity_ids:
        raise ValueError(f"{sample_label} contains no households.")
    if electricity_ids != gas_ids:
        missing_from_electricity = sorted(gas_ids - electricity_ids)
        missing_from_gas = sorted(electricity_ids - gas_ids)
        raise ValueError(
            f"{sample_label} HPWH and gas files do not contain identical "
            "building IDs. "
            f"Missing from HPWH: {missing_from_electricity[:10]}; "
            f"missing from gas: {missing_from_gas[:10]}."
        )

    for usage_name, usage_df in [
        ("HPWH electricity", electricity_usage),
        ("natural-gas input", gas_usage),
    ]:
        profile_sets = (
            usage_df.groupby("bldg_id")["season"]
            .agg(lambda values: set(values.astype(str)))
        )
        incomplete = profile_sets[
            profile_sets.map(
                lambda profiles: not required_profiles.issubset(profiles)
            )
        ]
        if not incomplete.empty:
            examples = {
                str(building_id): sorted(list(profiles))
                for building_id, profiles in incomplete.head(10).items()
            }
            raise ValueError(
                f"{sample_label} {usage_name} file is missing one or more "
                f"required profiles (1, 8, year): {examples}"
            )

        daily_total = usage_df[USAGE_HOUR_COLUMNS].astype(float).sum(axis=1)
        nonpositive = daily_total <= 0
        if nonpositive.any():
            examples = (
                usage_df.loc[nonpositive, ["bldg_id", "season"]]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                f"{sample_label} {usage_name} contains zero or negative "
                f"daily profiles. Examples: {examples}"
            )

def _validate_space_heating_component_pair(
    heat_pump_usage: pd.DataFrame,
    gas_usage: pd.DataFrame,
    sample_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Validate the paired Upgrade 4 / Upgrade 0 component workbooks.

    Every household must have January, August, and annual-average profiles.
    Upgrade 4 must contain five named components and Upgrade 0 must contain
    two named components for every building/period. Individual component rows
    may be all zero. A household is excluded only if an entire scenario has no
    positive space-heating use across all components and stored periods.
    """
    required_profiles = {"1", "8", "year"}

    heat_pump_ids = set(heat_pump_usage["bldg_id"].astype(str))
    gas_ids = set(gas_usage["bldg_id"].astype(str))

    if not heat_pump_ids:
        raise ValueError(f"{sample_label} contains no households.")
    if heat_pump_ids != gas_ids:
        missing_from_heat_pump = sorted(gas_ids - heat_pump_ids)
        missing_from_gas = sorted(heat_pump_ids - gas_ids)
        raise ValueError(
            f"{sample_label} Upgrade 4 and Upgrade 0 files do not contain "
            "identical building IDs. "
            f"Missing from Upgrade 4: {missing_from_heat_pump[:10]}; "
            f"missing from Upgrade 0: {missing_from_gas[:10]}."
        )

    positive_ids_by_scenario: list[set[str]] = []

    for usage_name, usage_df, expected_components in [
        ("Upgrade 4 heat-pump scenario", heat_pump_usage, set(GT_SPACE_UPGRADE4_COMPONENTS)),
        ("Upgrade 0 natural-gas scenario", gas_usage, set(GT_SPACE_UPGRADE0_COMPONENTS)),
    ]:
        profile_sets = (
            usage_df.groupby("bldg_id")["season"]
            .agg(lambda values: set(values.astype(str)))
        )
        incomplete_profiles = profile_sets[
            profile_sets.map(
                lambda profiles: not required_profiles.issubset(profiles)
            )
        ]
        if not incomplete_profiles.empty:
            examples = {
                str(building_id): sorted(list(profiles))
                for building_id, profiles in incomplete_profiles.head(10).items()
            }
            raise ValueError(
                f"{sample_label} {usage_name} is missing one or more "
                f"required profiles (1, 8, year): {examples}"
            )

        combo_components = (
            usage_df.groupby(["bldg_id", "season"])["component"]
            .agg(lambda values: set(values.astype(str)))
        )
        invalid_components = combo_components[
            combo_components.map(lambda components: components != expected_components)
        ]
        if not invalid_components.empty:
            examples = {
                f"{building_id}/{season}": sorted(list(components))
                for (building_id, season), components
                in invalid_components.head(10).items()
            }
            raise ValueError(
                f"{sample_label} {usage_name} does not contain exactly the "
                f"expected components. Examples: {examples}. "
                f"Expected: {sorted(expected_components)}"
            )

        hourly = usage_df[USAGE_HOUR_COLUMNS].astype(float)
        negative = hourly.lt(0).any(axis=1)
        if negative.any():
            examples = (
                usage_df.loc[
                    negative,
                    ["bldg_id", "season", "component"],
                ]
                .head(10)
                .to_dict("records")
            )
            raise ValueError(
                f"{sample_label} {usage_name} contains negative hourly "
                f"profiles. Examples: {examples}"
            )

        row_totals = hourly.sum(axis=1)
        household_totals = pd.DataFrame(
            {
                "bldg_id": usage_df["bldg_id"].astype(str).to_numpy(),
                "profile_total": row_totals.to_numpy(),
            }
        ).groupby("bldg_id", sort=False)["profile_total"].sum()

        positive_ids_by_scenario.append(
            set(household_totals[household_totals > 0].index.astype(str))
        )

    eligible_ids = positive_ids_by_scenario[0] & positive_ids_by_scenario[1]
    excluded_ids = tuple(sorted(heat_pump_ids - eligible_ids))

    if not eligible_ids:
        raise ValueError(
            f"{sample_label} contains no eligible paired households with "
            "positive space-heating use."
        )

    filtered_heat_pump = heat_pump_usage.loc[
        heat_pump_usage["bldg_id"].astype(str).isin(eligible_ids)
    ].reset_index(drop=True)
    filtered_gas = gas_usage.loc[
        gas_usage["bldg_id"].astype(str).isin(eligible_ids)
    ].reset_index(drop=True)

    return filtered_heat_pump, filtered_gas, excluded_ids

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

    mapped_ids = set(provider_map["bldg_id"].astype(str))
    required_profiles = {"1", "8", "year"}

    for usage_name, usage_df in [
        ("electricity usage", electric_usage),
        ("gas usage", gas_usage),
    ]:
        mapped_usage = usage_df[
            usage_df["bldg_id"].isin(mapped_ids)
        ].copy()

        missing_ids = sorted(
            mapped_ids - set(mapped_usage["bldg_id"].astype(str))
        )
        if missing_ids:
            raise ValueError(
                f"{usage_name} is missing mapped households: "
                f"{missing_ids[:10]}"
            )

        profile_sets = (
            mapped_usage.groupby("bldg_id")["season"]
            .agg(lambda values: set(values.astype(str)))
        )
        incomplete = profile_sets[
            profile_sets.map(
                lambda profiles: not required_profiles.issubset(profiles)
            )
        ]
        if not incomplete.empty:
            examples = {
                str(building_id): sorted(list(profiles))
                for building_id, profiles in incomplete.head(10).items()
            }
            raise ValueError(
                f"{usage_name} is missing one or more required profiles "
                f"(1, 8, year): {examples}"
            )

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
