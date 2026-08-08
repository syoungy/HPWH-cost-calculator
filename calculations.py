from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loading import (
    CalculatorData,
    GT_SPACE_UPGRADE0_COMPONENTS,
    GT_SPACE_UPGRADE4_COMPONENTS,
    RATE_HOUR_COLUMNS,
    USAGE_HOUR_COLUMNS,
)


PERIOD_MONTH = {"1": 1, "8": 8}
SUPPORTED_PERIODS = {"1", "8", "year"}

GAS_WH_EFFICIENCY = 0.62
PROPANE_WH_EFFICIENCY = 0.62
RESISTANCE_WH_EFFICIENCY = 0.95
PROPANE_KWH_PER_GALLON = 91_452.0 / 3_412.0
DEFAULT_PROPANE_PRICE_PER_GALLON = 2.370


@dataclass(frozen=True)
class StatewideResult:
    household_summary: pd.DataFrame
    hpwh_tariff_costs: pd.DataFrame
    resistance_tariff_costs: pd.DataFrame
    gas_households: pd.DataFrame
    paired_households: pd.DataFrame
    hpwh_summary: dict[str, float | int]
    gas_summary: dict[str, float | int]
    propane_summary: dict[str, float | int]
    resistance_summary: dict[str, float | int]
    hpwh_minus_gas_summary: dict[str, float | int]
    interval_95: dict[str, dict[str, float | int]]
    hourly_average_rates: pd.DataFrame
    hourly_average_usage: pd.DataFrame
    missing_electric: pd.DataFrame
    missing_gas: pd.DataFrame


@dataclass(frozen=True)
class CountyScenarioResult:
    households: pd.DataFrame
    hpwh_summary: dict[str, float | int]
    gas_summary: dict[str, float | int]
    propane_summary: dict[str, float | int]
    resistance_summary: dict[str, float | int]
    hpwh_minus_gas_summary: dict[str, float | int]
    geography_label: str
    sample_counties: tuple[str, ...]
    electric_provider: str
    gas_provider: str
    electric_tariff: str
    gas_tariff: str
    electric_year: int
    gas_year: int
    usage_sample: str


@dataclass(frozen=True)
class SpaceHeatingResult:
    households: pd.DataFrame
    heat_pump_summary: dict[str, float | int]
    gas_summary: dict[str, float | int]
    propane_summary: dict[str, float | int]
    heat_pump_minus_gas_summary: dict[str, float | int]
    interval_95: dict[str, dict[str, float | int]]
    geography_label: str
    sample_counties: tuple[str, ...]
    electric_provider: str
    gas_provider: str
    electric_tariff: str
    gas_tariff: str
    electric_year: int
    gas_year: int


def _summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "n": 0}
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "n": int(clean.size),
    }


def _household_tariff_summary(
    frame: pd.DataFrame,
    cost_column: str,
    prefix: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for building_id, group in frame.groupby("bldg_id", sort=False):
        min_index = group[cost_column].idxmin()
        max_index = group[cost_column].idxmax()
        first = group.iloc[0]
        records.append(
            {
                "bldg_id": str(building_id),
                "county": first["county"],
                "electric_provider": first["electric_provider"],
                f"{prefix}_tariff_count": int(len(group)),
                f"{prefix}_min": float(group[cost_column].min()),
                f"{prefix}_min_tariff": group.loc[min_index, "electric_tariff"],
                f"{prefix}_max": float(group[cost_column].max()),
                f"{prefix}_max_tariff": group.loc[max_index, "electric_tariff"],
                f"{prefix}_average": float(group[cost_column].mean()),
            }
        )
    return pd.DataFrame(records)


def _electric_summary(
    household_frame: pd.DataFrame,
    prefix: str,
) -> dict[str, float | int]:
    if household_frame.empty:
        return _summary(pd.Series(dtype=float))
    return {
        "min": float(household_frame[f"{prefix}_min"].min()),
        "max": float(household_frame[f"{prefix}_max"].max()),
        "mean": float(household_frame[f"{prefix}_average"].mean()),
        "n": int(len(household_frame)),
    }


def _period_multiplier(year: int, period: str) -> float:
    if period in PERIOD_MONTH:
        return float(calendar.monthrange(year, PERIOD_MONTH[period])[1])
    return float(
        sum(calendar.monthrange(year, month)[1] for month in range(1, 13))
        / 12.0
    )


def _usage_profile(
    usage: pd.DataFrame,
    period: str,
    prefix: str,
) -> pd.DataFrame:
    selected = usage.loc[
        usage["season"].astype(str) == str(period),
        ["bldg_id", *USAGE_HOUR_COLUMNS],
    ].copy()
    selected["bldg_id"] = selected["bldg_id"].astype(str)
    selected = selected.rename(
        columns={column: f"{prefix}_{hour:02d}" for hour, column in enumerate(USAGE_HOUR_COLUMNS)}
    )
    return selected


def _space_component_profile(
    usage: pd.DataFrame,
    period: str,
    component: str,
    prefix: str,
) -> pd.DataFrame:
    """Select one named component row per house for a space-heating period."""
    selected = usage.loc[
        (usage["season"].astype(str) == str(period))
        & (usage["component"].astype(str) == str(component)),
        ["bldg_id", *USAGE_HOUR_COLUMNS],
    ].copy()
    selected["bldg_id"] = selected["bldg_id"].astype(str)
    selected = selected.rename(
        columns={
            column: f"{prefix}_{hour:02d}"
            for hour, column in enumerate(USAGE_HOUR_COLUMNS)
        }
    )
    return selected

def _houses_with_profiles(data: CalculatorData, period: str) -> pd.DataFrame:
    electric = _usage_profile(data.electricity_usage, period, "hpwh")
    gas = _usage_profile(data.gas_usage, period, "gas")
    houses = (
        data.provider_map.copy()
        .assign(bldg_id=lambda frame: frame["bldg_id"].astype(str))
        .merge(electric, on="bldg_id", how="left", validate="one_to_one")
        .merge(gas, on="bldg_id", how="left", validate="one_to_one")
    )
    return houses


def _dedicated_houses_with_profiles(
    electricity_usage: pd.DataFrame,
    gas_usage: pd.DataFrame,
    period: str,
    county_label: str,
) -> pd.DataFrame:
    electric = _usage_profile(electricity_usage, period, "hpwh")
    gas = _usage_profile(gas_usage, period, "gas")
    houses = electric.merge(
        gas,
        on="bldg_id",
        how="inner",
        validate="one_to_one",
    )
    houses.insert(1, "in.county_name", str(county_label))
    return houses


def _latest_complete_electric_scenarios(
    rates: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """Create one reusable 24-hour cost vector per provider/tariff."""
    records: list[dict[str, object]] = []

    for (provider, tariff), group in rates.groupby(
        ["elec_provd", "tariff"], sort=False
    ):
        chosen_year: int | None = None
        chosen_group: pd.DataFrame | None = None

        if period in PERIOD_MONTH:
            month = PERIOD_MONTH[period]
            eligible = group[group["month"].astype(int) == month]
            if not eligible.empty:
                chosen_year = int(eligible["year"].max())
                chosen_group = eligible[eligible["year"].astype(int) == chosen_year]
                if len(chosen_group) != 1:
                    chosen_group = None
        else:
            valid_years = [
                int(year)
                for year, year_group in group.groupby("year")
                if set(year_group["month"].astype(int)) >= set(range(1, 13))
            ]
            if valid_years:
                chosen_year = max(valid_years)
                chosen_group = group[
                    (group["year"].astype(int) == chosen_year)
                    & (group["month"].astype(int).isin(range(1, 13)))
                ].copy()
                if len(chosen_group) != 12:
                    chosen_group = None

        if chosen_group is None or chosen_year is None:
            continue

        if period in PERIOD_MONTH:
            month = PERIOD_MONTH[period]
            rate_vector = chosen_group.iloc[0][RATE_HOUR_COLUMNS].astype(float).to_numpy()
            days = calendar.monthrange(chosen_year, month)[1]
            cost_factor = days * rate_vector
            display_rate = rate_vector
        else:
            cost_factor = np.zeros(24, dtype=float)
            weighted_rate = np.zeros(24, dtype=float)
            annual_days = 0
            for month in range(1, 13):
                row = chosen_group[chosen_group["month"].astype(int) == month]
                if len(row) != 1:
                    raise ValueError(
                        f"Incomplete annual tariff: {provider} / {tariff} / {chosen_year}."
                    )
                days = calendar.monthrange(chosen_year, month)[1]
                rate_vector = row.iloc[0][RATE_HOUR_COLUMNS].astype(float).to_numpy()
                cost_factor += days * rate_vector / 12.0
                weighted_rate += days * rate_vector
                annual_days += days
            display_rate = weighted_rate / annual_days

        record: dict[str, object] = {
            "electric_provider": str(provider),
            "electric_tariff": str(tariff),
            "electric_year": int(chosen_year),
        }
        record.update({f"factor_{hour:02d}": float(cost_factor[hour]) for hour in range(24)})
        record.update({f"rate_{hour:02d}": float(display_rate[hour]) for hour in range(24)})
        records.append(record)

    return pd.DataFrame(records)


def _latest_gas_scenarios(
    rates: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (provider, tariff), group in rates.groupby(["gas_provd", "tariff"], sort=False):
        year = int(group["year"].max())
        chosen = group[group["year"].astype(int) == year].copy()
        if "source_order" in chosen.columns:
            chosen = chosen.sort_values("source_order", kind="stable")
        row = chosen.iloc[0]
        rate_vector = row[RATE_HOUR_COLUMNS].astype(float).to_numpy()
        multiplier = _period_multiplier(year, period)
        record: dict[str, object] = {
            "gas_provider": str(provider),
            "gas_tariff": str(tariff),
            "gas_year": year,
        }
        record.update({f"factor_{hour:02d}": float(multiplier * rate_vector[hour]) for hour in range(24)})
        record.update({f"rate_{hour:02d}": float(rate_vector[hour]) for hour in range(24)})
        records.append(record)
    return pd.DataFrame(records)


def available_electric_tariffs(
    data: CalculatorData,
    provider: str,
    period: str,
) -> list[str]:
    scenarios = _latest_complete_electric_scenarios(data.electricity_rates, period)
    return sorted(
        scenarios.loc[
            scenarios["electric_provider"].astype(str) == str(provider),
            "electric_tariff",
        ].astype(str).unique().tolist()
    )


def available_gas_tariffs(data: CalculatorData, provider: str) -> list[str]:
    return sorted(
        data.gas_rates.loc[
            data.gas_rates["gas_provd"].astype(str) == str(provider),
            "tariff",
        ].astype(str).unique().tolist()
    )


def _technology_tariff_costs(
    houses: pd.DataFrame,
    scenarios: pd.DataFrame,
    usage_columns: list[str],
    cost_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate all mapped electricity tariffs with vectorized matrix products."""
    long_frames: list[pd.DataFrame] = []
    missing: list[dict[str, object]] = []
    house_rate_vectors: list[np.ndarray] = []

    factor_columns = [f"factor_{hour:02d}" for hour in range(24)]
    rate_columns = [f"rate_{hour:02d}" for hour in range(24)]

    for provider, provider_houses in houses.groupby("elec_provd", sort=False):
        provider_scenarios = scenarios[
            scenarios["electric_provider"].astype(str) == str(provider)
        ].copy()

        if provider_scenarios.empty:
            for _, house in provider_houses.iterrows():
                missing.append(
                    {
                        "bldg_id": str(house["bldg_id"]),
                        "county": house["in.county_name"],
                        "electric_provider": provider,
                        "reason": "No complete applicable electricity tariff",
                    }
                )
            continue

        usage_matrix = provider_houses[usage_columns].astype(float).to_numpy()
        factor_matrix = provider_scenarios[factor_columns].astype(float).to_numpy()
        cost_matrix = usage_matrix @ factor_matrix.T

        n_houses = len(provider_houses)
        n_scenarios = len(provider_scenarios)
        frame = pd.DataFrame(
            {
                "bldg_id": np.repeat(provider_houses["bldg_id"].astype(str).to_numpy(), n_scenarios),
                "county": np.repeat(provider_houses["in.county_name"].astype(str).to_numpy(), n_scenarios),
                "electric_provider": np.repeat(str(provider), n_houses * n_scenarios),
                "electric_tariff": np.tile(provider_scenarios["electric_tariff"].astype(str).to_numpy(), n_houses),
                "electric_year": np.tile(provider_scenarios["electric_year"].astype(int).to_numpy(), n_houses),
                cost_column: cost_matrix.reshape(-1),
            }
        )
        long_frames.append(frame)

        mean_rate = provider_scenarios[rate_columns].astype(float).to_numpy().mean(axis=0)
        house_rate_vectors.extend([mean_rate] * n_houses)

    long_frame = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()
    missing_frame = pd.DataFrame(missing)
    rate_frame = pd.DataFrame(house_rate_vectors, columns=[f"hour_{hour:02d}" for hour in range(24)])
    return long_frame, missing_frame, rate_frame


def _gas_household_costs(
    houses: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    missing: list[dict[str, object]] = []
    house_rate_vectors: list[np.ndarray] = []
    factor_columns = [f"factor_{hour:02d}" for hour in range(24)]
    rate_columns = [f"rate_{hour:02d}" for hour in range(24)]
    gas_columns = [f"gas_{hour:02d}" for hour in range(24)]

    for provider, provider_houses in houses.groupby("gas_provd", sort=False):
        provider_scenarios = scenarios[
            scenarios["gas_provider"].astype(str) == str(provider)
        ].copy()
        if provider_scenarios.empty:
            for _, house in provider_houses.iterrows():
                missing.append(
                    {
                        "bldg_id": str(house["bldg_id"]),
                        "county": house["in.county_name"],
                        "gas_provider": provider,
                        "reason": "No gas tariff",
                    }
                )
            continue

        # Statewide logic retains one latest mapped gas tariff per provider.
        scenario = provider_scenarios.sort_values(
            ["gas_year", "gas_tariff"], kind="stable"
        ).iloc[-1]
        usage_matrix = provider_houses[gas_columns].astype(float).to_numpy()
        factor_vector = scenario[factor_columns].astype(float).to_numpy()
        costs = usage_matrix @ factor_vector

        frame = provider_houses[["bldg_id", "in.county_name"]].copy()
        frame = frame.rename(columns={"in.county_name": "county"})
        frame["bldg_id"] = frame["bldg_id"].astype(str)
        frame["gas_provider"] = str(provider)
        frame["gas_tariff"] = str(scenario["gas_tariff"])
        frame["gas_year"] = int(scenario["gas_year"])
        frame["gas_monthly_cost"] = costs
        records.append(frame)

        rate_vector = scenario[rate_columns].astype(float).to_numpy()
        house_rate_vectors.extend([rate_vector] * len(provider_houses))

    cost_frame = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    missing_frame = pd.DataFrame(missing)
    rate_frame = pd.DataFrame(house_rate_vectors, columns=[f"hour_{hour:02d}" for hour in range(24)])
    return cost_frame, missing_frame, rate_frame


def _central_95_ids(frame: pd.DataFrame, usage_column: str) -> set[str]:
    clean = frame[["bldg_id", usage_column]].dropna().copy()
    if clean.empty:
        return set()
    lower = float(clean[usage_column].quantile(0.025))
    upper = float(clean[usage_column].quantile(0.975))
    return set(
        clean.loc[
            clean[usage_column].between(lower, upper, inclusive="both"),
            "bldg_id",
        ].astype(str)
    )


def _interval_mean(
    frame: pd.DataFrame,
    ids: set[str],
    value_column: str,
) -> dict[str, float | int]:
    selected = frame[frame["bldg_id"].astype(str).isin(ids)]
    values = pd.to_numeric(selected[value_column], errors="coerce").dropna()
    return {
        "mean": float(values.mean()) if not values.empty else np.nan,
        "n": int(values.size),
    }


def calculate_statewide(
    data: CalculatorData,
    period: str,
    propane_price_per_gallon: float = DEFAULT_PROPANE_PRICE_PER_GALLON,
) -> StatewideResult:
    if period not in SUPPORTED_PERIODS:
        raise ValueError(f"Unsupported period: {period}")
    if propane_price_per_gallon < 0:
        raise ValueError("Propane price cannot be negative.")

    houses = _houses_with_profiles(data, period)
    hpwh_columns = [f"hpwh_{hour:02d}" for hour in range(24)]
    gas_columns = [f"gas_{hour:02d}" for hour in range(24)]

    houses["hpwh_daily_use_kwh"] = houses[hpwh_columns].sum(axis=1)
    houses["gas_input_daily_use_kwh"] = houses[gas_columns].sum(axis=1)
    houses["resistance_daily_use_kwh"] = (
        houses["gas_input_daily_use_kwh"]
        * GAS_WH_EFFICIENCY
        / RESISTANCE_WH_EFFICIENCY
    )
    houses["propane_input_daily_kwh"] = (
        houses["gas_input_daily_use_kwh"]
        * GAS_WH_EFFICIENCY
        / PROPANE_WH_EFFICIENCY
    )

    electric_scenarios = _latest_complete_electric_scenarios(
        data.electricity_rates, period
    )
    gas_scenarios = _latest_gas_scenarios(data.gas_rates, period)

    hpwh_costs, missing_electric, hpwh_rate_rows = _technology_tariff_costs(
        houses,
        electric_scenarios,
        hpwh_columns,
        "hpwh_monthly_cost",
    )

    resistance_houses = houses.copy()
    resistance_columns = [f"resistance_{hour:02d}" for hour in range(24)]
    resistance_ratio = GAS_WH_EFFICIENCY / RESISTANCE_WH_EFFICIENCY
    for hour, column in enumerate(gas_columns):
        resistance_houses[resistance_columns[hour]] = (
            resistance_houses[column].astype(float) * resistance_ratio
        )

    resistance_costs, _, resistance_rate_rows = _technology_tariff_costs(
        resistance_houses,
        electric_scenarios,
        resistance_columns,
        "resistance_monthly_cost",
    )

    gas_costs, missing_gas, gas_rate_rows = _gas_household_costs(
        houses, gas_scenarios
    )

    hpwh_house = _household_tariff_summary(
        hpwh_costs, "hpwh_monthly_cost", "hpwh"
    )
    resistance_house = _household_tariff_summary(
        resistance_costs, "resistance_monthly_cost", "resistance"
    )

    propane_year = 2026
    propane_multiplier = _period_multiplier(propane_year, period)
    propane_frame = houses[
        ["bldg_id", "in.county_name", "propane_input_daily_kwh"]
    ].copy()
    propane_frame = propane_frame.rename(columns={"in.county_name": "county"})
    propane_frame["bldg_id"] = propane_frame["bldg_id"].astype(str)
    propane_frame["propane_price_per_gallon"] = float(propane_price_per_gallon)
    propane_frame["propane_daily_gallons"] = (
        propane_frame["propane_input_daily_kwh"] / PROPANE_KWH_PER_GALLON
    )
    propane_frame["propane_monthly_cost"] = (
        propane_frame["propane_daily_gallons"]
        * propane_multiplier
        * float(propane_price_per_gallon)
    )

    base = houses[
        [
            "bldg_id",
            "in.county_name",
            "elec_provd",
            "gas_provd",
            "hpwh_daily_use_kwh",
            "gas_input_daily_use_kwh",
            "resistance_daily_use_kwh",
            "propane_input_daily_kwh",
        ]
    ].copy()
    base = base.rename(
        columns={
            "in.county_name": "county",
            "elec_provd": "electric_provider",
            "gas_provd": "gas_provider",
        }
    )
    base["bldg_id"] = base["bldg_id"].astype(str)

    household_summary = (
        base.merge(hpwh_house, on=["bldg_id", "county", "electric_provider"], how="left")
        .merge(
            resistance_house,
            on=["bldg_id", "county", "electric_provider"],
            how="left",
        )
        .merge(
            gas_costs[
                [
                    "bldg_id",
                    "county",
                    "gas_tariff",
                    "gas_year",
                    "gas_monthly_cost",
                ]
            ],
            on=["bldg_id", "county"],
            how="left",
        )
        .merge(
            propane_frame[
                [
                    "bldg_id",
                    "county",
                    "propane_price_per_gallon",
                    "propane_daily_gallons",
                    "propane_monthly_cost",
                ]
            ],
            on=["bldg_id", "county"],
            how="left",
        )
    )

    paired = household_summary.dropna(
        subset=["hpwh_average", "gas_monthly_cost"]
    ).copy()
    paired["hpwh_minus_gas_min"] = paired["hpwh_min"] - paired["gas_monthly_cost"]
    paired["hpwh_minus_gas_max"] = paired["hpwh_max"] - paired["gas_monthly_cost"]
    paired["hpwh_minus_gas_average"] = (
        paired["hpwh_average"] - paired["gas_monthly_cost"]
    )
    paired["resistance_minus_gas_average"] = (
        paired["resistance_average"] - paired["gas_monthly_cost"]
    )
    paired["propane_minus_gas"] = (
        paired["propane_monthly_cost"] - paired["gas_monthly_cost"]
    )

    hpwh_summary = _electric_summary(hpwh_house, "hpwh")
    resistance_summary = _electric_summary(resistance_house, "resistance")
    gas_summary = _summary(gas_costs["gas_monthly_cost"])
    propane_summary = _summary(propane_frame["propane_monthly_cost"])
    hpwh_minus_gas_summary = {
        "min": float(paired["hpwh_minus_gas_min"].min()),
        "max": float(paired["hpwh_minus_gas_max"].max()),
        "mean": float(paired["hpwh_minus_gas_average"].mean()),
        "n": int(len(paired)),
    }

    hpwh_95_ids = _central_95_ids(base, "hpwh_daily_use_kwh")
    gas_95_ids = _central_95_ids(base, "gas_input_daily_use_kwh")
    paired_95_ids = hpwh_95_ids & gas_95_ids

    interval_95 = {
        "hpwh": _interval_mean(hpwh_house, hpwh_95_ids, "hpwh_average"),
        "gas": _interval_mean(gas_costs, gas_95_ids, "gas_monthly_cost"),
        "propane": _interval_mean(
            propane_frame, gas_95_ids, "propane_monthly_cost"
        ),
        "resistance": _interval_mean(
            resistance_house, gas_95_ids, "resistance_average"
        ),
        "hpwh_minus_gas": _interval_mean(
            paired, paired_95_ids, "hpwh_minus_gas_average"
        ),
    }

    # Equal-house weighting for rate charts. The technology shares the same
    # electricity tariff set, so HPWH and resistance rate profiles are equal.
    electric_rate_average = (
        hpwh_rate_rows.mean(axis=0).to_numpy()
        if not hpwh_rate_rows.empty
        else np.repeat(np.nan, 24)
    )
    gas_rate_average = (
        gas_rate_rows.mean(axis=0).to_numpy()
        if not gas_rate_rows.empty
        else np.repeat(np.nan, 24)
    )
    hourly_average_rates = pd.DataFrame(
        {
            "hour": range(24),
            "electricity_rate": electric_rate_average,
            "gas_rate": gas_rate_average,
        }
    )

    hourly_average_usage = pd.DataFrame(
        {
            "hour": range(24),
            "hpwh_kwh": houses[hpwh_columns].mean(axis=0).to_numpy(),
            "gas_input_kwh": houses[gas_columns].mean(axis=0).to_numpy(),
            "resistance_kwh": resistance_houses[resistance_columns]
            .mean(axis=0)
            .to_numpy(),
        }
    )

    return StatewideResult(
        household_summary=household_summary,
        hpwh_tariff_costs=hpwh_costs,
        resistance_tariff_costs=resistance_costs,
        gas_households=gas_costs,
        paired_households=paired,
        hpwh_summary=hpwh_summary,
        gas_summary=gas_summary,
        propane_summary=propane_summary,
        resistance_summary=resistance_summary,
        hpwh_minus_gas_summary=hpwh_minus_gas_summary,
        interval_95=interval_95,
        hourly_average_rates=hourly_average_rates,
        hourly_average_usage=hourly_average_usage,
        missing_electric=missing_electric,
        missing_gas=missing_gas,
    )


def _selected_electric_scenario(
    data: CalculatorData,
    provider: str,
    tariff: str,
    period: str,
) -> pd.Series:
    scenarios = _latest_complete_electric_scenarios(data.electricity_rates, period)
    selected = scenarios[
        (scenarios["electric_provider"].astype(str) == str(provider))
        & (scenarios["electric_tariff"].astype(str) == str(tariff))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one electricity scenario for {provider} / {tariff} / {period}; "
            f"found {len(selected)}."
        )
    return selected.iloc[0]


def _selected_gas_scenario(
    data: CalculatorData,
    provider: str,
    tariff: str,
    period: str,
) -> pd.Series:
    scenarios = _latest_gas_scenarios(data.gas_rates, period)
    selected = scenarios[
        (scenarios["gas_provider"].astype(str) == str(provider))
        & (scenarios["gas_tariff"].astype(str) == str(tariff))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one gas scenario for {provider} / {tariff}; found {len(selected)}."
        )
    return selected.iloc[0]


def calculate_county_scenario(
    data: CalculatorData,
    county: str | tuple[str, ...],
    electric_provider: str,
    electric_tariff: str,
    gas_provider: str,
    gas_tariff: str,
    period: str,
    propane_price_per_gallon: float = DEFAULT_PROPANE_PRICE_PER_GALLON,
    geography_label: str | None = None,
    usage_sample: str = "mapped",
) -> CountyScenarioResult:
    if period not in SUPPORTED_PERIODS:
        raise ValueError(f"Unsupported period: {period}")
    if propane_price_per_gallon < 0:
        raise ValueError("Propane price cannot be negative.")

    sample_counties = (county,) if isinstance(county, str) else tuple(county)
    sample_counties = tuple(str(value) for value in sample_counties)
    if not sample_counties:
        raise ValueError("At least one sample county is required.")

    label = geography_label or (
        sample_counties[0]
        if len(sample_counties) == 1
        else " + ".join(sample_counties)
    )

    if usage_sample == "mapped":
        houses = _houses_with_profiles(data, period)
        houses = houses[
            houses["in.county_name"].astype(str).isin(sample_counties)
        ].copy()
        empty_message = "No mapped sample households"
    elif usage_sample == "grand_traverse_dedicated":
        if sample_counties != ("Grand Traverse County",):
            raise ValueError(
                "The Grand Traverse dedicated sample can only be used with "
                "Grand Traverse County."
            )
        houses = _dedicated_houses_with_profiles(
            data.gt_electricity_usage,
            data.gt_gas_usage,
            period,
            "Grand Traverse County",
        )
        empty_message = "No dedicated Grand Traverse sample households"
    else:
        raise ValueError(f"Unsupported usage sample: {usage_sample}")

    if houses.empty:
        raise ValueError(
            f"{empty_message} were found for {label}: "
            f"{', '.join(sample_counties)}."
        )

    # The selected utilities are applied to every sampled household in the
    # chosen county or multi-county area. This intentionally permits a tariff
    # scenario that differs from the provider mapping in the source workbook.
    electric_scenario = _selected_electric_scenario(
        data, electric_provider, electric_tariff, period
    )
    gas_scenario = _selected_gas_scenario(
        data, gas_provider, gas_tariff, period
    )

    hpwh_columns = [f"hpwh_{hour:02d}" for hour in range(24)]
    gas_columns = [f"gas_{hour:02d}" for hour in range(24)]
    factor_columns = [f"factor_{hour:02d}" for hour in range(24)]

    hpwh_matrix = houses[hpwh_columns].astype(float).to_numpy()
    gas_matrix = houses[gas_columns].astype(float).to_numpy()
    resistance_matrix = gas_matrix * GAS_WH_EFFICIENCY / RESISTANCE_WH_EFFICIENCY

    electric_factor = electric_scenario[factor_columns].astype(float).to_numpy()
    gas_factor = gas_scenario[factor_columns].astype(float).to_numpy()

    hpwh_cost = hpwh_matrix @ electric_factor
    resistance_cost = resistance_matrix @ electric_factor
    gas_cost = gas_matrix @ gas_factor

    propane_daily_input = gas_matrix.sum(axis=1) * GAS_WH_EFFICIENCY / PROPANE_WH_EFFICIENCY
    propane_daily_gallons = propane_daily_input / PROPANE_KWH_PER_GALLON
    propane_multiplier = _period_multiplier(2026, period)
    propane_cost = (
        propane_daily_gallons
        * propane_multiplier
        * float(propane_price_per_gallon)
    )

    result = pd.DataFrame(
        {
            "bldg_id": houses["bldg_id"].astype(str).to_numpy(),
            "scenario_geography": str(label),
            "county": houses["in.county_name"].astype(str).to_numpy(),
            "electric_provider": str(electric_provider),
            "electric_tariff": str(electric_tariff),
            "electric_year": int(electric_scenario["electric_year"]),
            "gas_provider": str(gas_provider),
            "gas_tariff": str(gas_tariff),
            "gas_year": int(gas_scenario["gas_year"]),
            "hpwh_daily_use_kwh": hpwh_matrix.sum(axis=1),
            "gas_input_daily_use_kwh": gas_matrix.sum(axis=1),
            "resistance_daily_use_kwh": resistance_matrix.sum(axis=1),
            "propane_daily_gallons": propane_daily_gallons,
            "hpwh_monthly_cost": hpwh_cost,
            "gas_monthly_cost": gas_cost,
            "propane_monthly_cost": propane_cost,
            "resistance_monthly_cost": resistance_cost,
        }
    )
    result["hpwh_minus_gas"] = result["hpwh_monthly_cost"] - result["gas_monthly_cost"]
    result["propane_minus_gas"] = result["propane_monthly_cost"] - result["gas_monthly_cost"]
    result["resistance_minus_gas"] = result["resistance_monthly_cost"] - result["gas_monthly_cost"]

    return CountyScenarioResult(
        households=result,
        hpwh_summary=_summary(result["hpwh_monthly_cost"]),
        gas_summary=_summary(result["gas_monthly_cost"]),
        propane_summary=_summary(result["propane_monthly_cost"]),
        resistance_summary=_summary(result["resistance_monthly_cost"]),
        hpwh_minus_gas_summary=_summary(result["hpwh_minus_gas"]),
        geography_label=str(label),
        sample_counties=sample_counties,
        electric_provider=str(electric_provider),
        gas_provider=str(gas_provider),
        electric_tariff=str(electric_tariff),
        gas_tariff=str(gas_tariff),
        electric_year=int(electric_scenario["electric_year"]),
        gas_year=int(gas_scenario["gas_year"]),
        usage_sample=str(usage_sample),
    )


def calculate_space_heating_scenario(
    data: CalculatorData,
    county: str | tuple[str, ...],
    electric_provider: str,
    electric_tariff: str,
    gas_provider: str,
    gas_tariff: str,
    period: str,
    propane_price_per_gallon: float = DEFAULT_PROPANE_PRICE_PER_GALLON,
    geography_label: str | None = None,
) -> SpaceHeatingResult:
    """Compare component-level operating costs for Grand Traverse space heating.

    Upgrade 4:
      - main heat-pump heating -> electricity tariff
      - main fan/pump -> electricity tariff
      - electric auxiliary heat -> electricity tariff
      - auxiliary fan/pump -> electricity tariff
      - natural-gas auxiliary heat -> gas tariff

    Upgrade 0:
      - main natural-gas heating -> gas tariff
      - blower/fan electricity -> electricity tariff

    The scenario totals are calculated household-by-household from the component
    costs, so the mean component costs sum to the displayed mean total before
    display rounding.
    """
    if period not in SUPPORTED_PERIODS:
        raise ValueError(f"Unsupported period: {period}")
    if propane_price_per_gallon < 0:
        raise ValueError("Propane price cannot be negative.")

    sample_counties = (county,) if isinstance(county, str) else tuple(county)
    sample_counties = tuple(str(value) for value in sample_counties)
    if sample_counties != ("Grand Traverse County",):
        raise ValueError(
            "The current space-heating sample is available only for "
            "Grand Traverse County / Traverse City."
        )

    label = geography_label or "Traverse City"

    hp_main_component, hp_main_fan_component, hp_aux_electric_component,         hp_aux_fan_component, hp_aux_gas_component = GT_SPACE_UPGRADE4_COMPONENTS
    gas_main_component, gas_blower_component = GT_SPACE_UPGRADE0_COMPONENTS

    hp_main = _space_component_profile(
        data.gt_space_electricity_usage,
        period,
        hp_main_component,
        "hp_main",
    )
    hp_main_fan = _space_component_profile(
        data.gt_space_electricity_usage,
        period,
        hp_main_fan_component,
        "hp_main_fan",
    )
    hp_aux_electric = _space_component_profile(
        data.gt_space_electricity_usage,
        period,
        hp_aux_electric_component,
        "hp_aux_electric",
    )
    hp_aux_fan = _space_component_profile(
        data.gt_space_electricity_usage,
        period,
        hp_aux_fan_component,
        "hp_aux_fan",
    )
    hp_aux_gas = _space_component_profile(
        data.gt_space_electricity_usage,
        period,
        hp_aux_gas_component,
        "hp_aux_gas",
    )

    gas_main = _space_component_profile(
        data.gt_space_gas_usage,
        period,
        gas_main_component,
        "gas_main",
    )
    gas_blower = _space_component_profile(
        data.gt_space_gas_usage,
        period,
        gas_blower_component,
        "gas_blower",
    )

    houses = (
        hp_main
        .merge(hp_main_fan, on="bldg_id", how="inner", validate="one_to_one")
        .merge(hp_aux_electric, on="bldg_id", how="inner", validate="one_to_one")
        .merge(hp_aux_fan, on="bldg_id", how="inner", validate="one_to_one")
        .merge(hp_aux_gas, on="bldg_id", how="inner", validate="one_to_one")
        .merge(gas_main, on="bldg_id", how="inner", validate="one_to_one")
        .merge(gas_blower, on="bldg_id", how="inner", validate="one_to_one")
    )
    houses.insert(1, "in.county_name", "Grand Traverse County")

    if houses.empty:
        raise ValueError(
            "No paired Grand Traverse space-heating households were found."
        )

    electric_scenario = _selected_electric_scenario(
        data,
        electric_provider,
        electric_tariff,
        period,
    )
    gas_scenario = _selected_gas_scenario(
        data,
        gas_provider,
        gas_tariff,
        period,
    )

    factor_columns = [f"factor_{hour:02d}" for hour in range(24)]
    electric_factor = electric_scenario[factor_columns].astype(float).to_numpy()
    gas_factor = gas_scenario[factor_columns].astype(float).to_numpy()

    def matrix(prefix: str) -> np.ndarray:
        columns = [f"{prefix}_{hour:02d}" for hour in range(24)]
        return houses[columns].astype(float).to_numpy()

    hp_main_matrix = matrix("hp_main")
    hp_main_fan_matrix = matrix("hp_main_fan")
    hp_aux_electric_matrix = matrix("hp_aux_electric")
    hp_aux_fan_matrix = matrix("hp_aux_fan")
    hp_aux_gas_matrix = matrix("hp_aux_gas")
    gas_main_matrix = matrix("gas_main")
    gas_blower_matrix = matrix("gas_blower")

    # Upgrade 4 component costs.
    hp_main_cost = hp_main_matrix @ electric_factor
    hp_main_fan_cost = hp_main_fan_matrix @ electric_factor
    hp_aux_electric_cost = hp_aux_electric_matrix @ electric_factor
    hp_aux_fan_cost = hp_aux_fan_matrix @ electric_factor
    hp_aux_gas_cost = hp_aux_gas_matrix @ gas_factor

    heat_pump_electricity_cost = (
        hp_main_cost
        + hp_main_fan_cost
        + hp_aux_electric_cost
        + hp_aux_fan_cost
    )
    heat_pump_cost = heat_pump_electricity_cost + hp_aux_gas_cost

    # Upgrade 0 component costs.
    gas_main_cost = gas_main_matrix @ gas_factor
    gas_blower_cost = gas_blower_matrix @ electric_factor
    gas_cost = gas_main_cost + gas_blower_cost

    # Propane keeps the Upgrade-0 main heating input and blower/fan electricity.
    propane_daily_gallons = gas_main_matrix.sum(axis=1) / PROPANE_KWH_PER_GALLON
    propane_multiplier = _period_multiplier(
        int(electric_scenario["electric_year"]),
        period,
    )
    propane_fuel_cost = (
        propane_daily_gallons
        * propane_multiplier
        * float(propane_price_per_gallon)
    )
    propane_cost = propane_fuel_cost + gas_blower_cost

    hp_main_daily = hp_main_matrix.sum(axis=1)
    hp_main_fan_daily = hp_main_fan_matrix.sum(axis=1)
    hp_aux_electric_daily = hp_aux_electric_matrix.sum(axis=1)
    hp_aux_fan_daily = hp_aux_fan_matrix.sum(axis=1)
    hp_aux_gas_daily = hp_aux_gas_matrix.sum(axis=1)
    heat_pump_electric_daily = (
        hp_main_daily
        + hp_main_fan_daily
        + hp_aux_electric_daily
        + hp_aux_fan_daily
    )

    gas_main_daily = gas_main_matrix.sum(axis=1)
    gas_blower_daily = gas_blower_matrix.sum(axis=1)

    result = pd.DataFrame(
        {
            "bldg_id": houses["bldg_id"].astype(str).to_numpy(),
            "scenario_geography": str(label),
            "county": houses["in.county_name"].astype(str).to_numpy(),
            "electric_provider": str(electric_provider),
            "electric_tariff": str(electric_tariff),
            "electric_year": int(electric_scenario["electric_year"]),
            "gas_provider": str(gas_provider),
            "gas_tariff": str(gas_tariff),
            "gas_year": int(gas_scenario["gas_year"]),

            # Upgrade 4 daily component use.
            "hp_main_heating_daily_use_kwh": hp_main_daily,
            "hp_main_fan_pump_daily_use_kwh": hp_main_fan_daily,
            "hp_aux_electric_daily_use_kwh": hp_aux_electric_daily,
            "hp_aux_fan_pump_daily_use_kwh": hp_aux_fan_daily,
            "hp_aux_natural_gas_daily_use_kwh": hp_aux_gas_daily,
            "heat_pump_electric_daily_use_kwh": heat_pump_electric_daily,
            "heat_pump_gas_aux_daily_use_kwh": hp_aux_gas_daily,
            "heat_pump_total_daily_use_kwh": (
                heat_pump_electric_daily + hp_aux_gas_daily
            ),

            # Upgrade 0 daily component use.
            "gas_main_daily_use_kwh": gas_main_daily,
            "gas_fan_pump_daily_use_kwh": gas_blower_daily,
            "gas_total_daily_use_kwh": gas_main_daily + gas_blower_daily,
            "propane_daily_gallons": propane_daily_gallons,

            # Upgrade 4 component costs.
            "hp_main_heating_electricity_cost": hp_main_cost,
            "hp_main_fan_pump_electricity_cost": hp_main_fan_cost,
            "hp_aux_electricity_cost": hp_aux_electric_cost,
            "hp_aux_fan_pump_electricity_cost": hp_aux_fan_cost,
            "hp_aux_natural_gas_cost": hp_aux_gas_cost,

            # Backward-compatible aggregate cost columns.
            "heat_pump_electricity_cost": heat_pump_electricity_cost,
            "heat_pump_gas_aux_cost": hp_aux_gas_cost,
            "heat_pump_monthly_cost": heat_pump_cost,

            # Upgrade 0 costs.
            "gas_fuel_cost": gas_main_cost,
            "gas_fan_pump_electricity_cost": gas_blower_cost,
            "gas_monthly_cost": gas_cost,

            # Propane scenario.
            "propane_fuel_cost": propane_fuel_cost,
            "propane_fan_pump_electricity_cost": gas_blower_cost,
            "propane_monthly_cost": propane_cost,
        }
    )

    result["heat_pump_minus_gas"] = (
        result["heat_pump_monthly_cost"] - result["gas_monthly_cost"]
    )
    result["propane_minus_gas"] = (
        result["propane_monthly_cost"] - result["gas_monthly_cost"]
    )

    # Preserve the v5.3 central-95% trimming basis:
    # Upgrade 4 -> total electricity profile; Upgrade 0 / propane -> main gas.
    heat_pump_95_ids = _central_95_ids(
        result,
        "heat_pump_electric_daily_use_kwh",
    )
    gas_95_ids = _central_95_ids(
        result,
        "gas_main_daily_use_kwh",
    )
    paired_95_ids = heat_pump_95_ids & gas_95_ids

    interval_95 = {
        "heat_pump": _interval_mean(
            result,
            heat_pump_95_ids,
            "heat_pump_monthly_cost",
        ),
        "gas": _interval_mean(
            result,
            gas_95_ids,
            "gas_monthly_cost",
        ),
        "propane": _interval_mean(
            result,
            gas_95_ids,
            "propane_monthly_cost",
        ),
        "heat_pump_minus_gas": _interval_mean(
            result,
            paired_95_ids,
            "heat_pump_minus_gas",
        ),
    }

    return SpaceHeatingResult(
        households=result,
        heat_pump_summary=_summary(result["heat_pump_monthly_cost"]),
        gas_summary=_summary(result["gas_monthly_cost"]),
        propane_summary=_summary(result["propane_monthly_cost"]),
        heat_pump_minus_gas_summary=_summary(result["heat_pump_minus_gas"]),
        interval_95=interval_95,
        geography_label=str(label),
        sample_counties=sample_counties,
        electric_provider=str(electric_provider),
        gas_provider=str(gas_provider),
        electric_tariff=str(electric_tariff),
        gas_tariff=str(gas_tariff),
        electric_year=int(electric_scenario["electric_year"]),
        gas_year=int(gas_scenario["gas_year"]),
    )

