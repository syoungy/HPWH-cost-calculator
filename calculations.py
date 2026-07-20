from __future__ import annotations

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_loading import (
    CalculatorData,
    RATE_HOUR_COLUMNS,
    USAGE_HOUR_COLUMNS,
)


PERIOD_MONTH = {
    "1": 1,
    "8": 8,
}


@dataclass(frozen=True)
class CalculationResult:
    electric_tariff_costs: pd.DataFrame
    electric_households: pd.DataFrame
    gas_households: pd.DataFrame
    all_households: pd.DataFrame
    paired_households: pd.DataFrame
    missing_electric: pd.DataFrame
    missing_gas: pd.DataFrame
    electric_summary: dict[str, float | int]
    gas_summary: dict[str, float | int]
    paired_gas_summary: dict[str, float | int]
    difference_summary: dict[str, float | int]
    hourly_average_rates: pd.DataFrame


def _profile_table(
    usage: pd.DataFrame,
    period: str,
    prefix: str,
) -> pd.DataFrame:
    selected = usage.loc[
        usage["season"] == period,
        ["bldg_id", *USAGE_HOUR_COLUMNS],
    ].copy()
    selected = selected.rename(
        columns={
            column: f"{prefix}_{hour}"
            for hour, column in enumerate(USAGE_HOUR_COLUMNS)
        }
    )
    return selected


def _valid_electric_years(
    rates: pd.DataFrame,
    provider: str,
    tariff: str,
    period: str,
) -> list[int]:
    subset = rates[
        (rates["elec_provd"] == provider)
        & (rates["tariff"] == tariff)
    ]

    if period in PERIOD_MONTH:
        month = PERIOD_MONTH[period]
        return sorted(
            subset.loc[subset["month"] == month, "year"]
            .astype(int)
            .unique()
            .tolist()
        )

    valid: list[int] = []
    for year, group in subset.groupby("year"):
        if set(group["month"].astype(int)) >= set(range(1, 13)):
            valid.append(int(year))
    return sorted(valid)


def _electric_monthly_cost(
    usage_vector: np.ndarray,
    rates: pd.DataFrame,
    provider: str,
    tariff: str,
    year: int,
    period: str,
) -> float:
    subset = rates[
        (rates["elec_provd"] == provider)
        & (rates["tariff"] == tariff)
        & (rates["year"] == int(year))
    ]

    if period in PERIOD_MONTH:
        month = PERIOD_MONTH[period]
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Expected one electricity-rate row for "
                f"{provider} / {tariff} / {year}-{month:02d}; "
                f"found {len(row)}."
            )
        rate_vector = (
            row.iloc[0][RATE_HOUR_COLUMNS]
            .astype(float)
            .to_numpy()
        )
        days = calendar.monthrange(year, month)[1]
        return float(days * np.dot(usage_vector, rate_vector))

    # The "year" usage row is the average hourly profile for one typical day.
    # Apply it to every calendar month's tariff, total the year, then divide by
    # 12 to report one average month.
    annual_cost = 0.0
    for month in range(1, 13):
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Annual average requires 12 complete monthly rate rows. "
                f"Problem at {provider} / {tariff} / {year}-{month:02d}."
            )
        rate_vector = (
            row.iloc[0][RATE_HOUR_COLUMNS]
            .astype(float)
            .to_numpy()
        )
        days = calendar.monthrange(year, month)[1]
        annual_cost += days * float(np.dot(usage_vector, rate_vector))

    return annual_cost / 12.0



def _electric_period_rate_vector(
    rates: pd.DataFrame,
    provider: str,
    tariff: str,
    year: int,
    period: str,
) -> np.ndarray:
    """
    Return the 24-hour electricity-rate vector used for the selected period.

    January/August:
        The selected month's hourly tariff.

    Annual average:
        A calendar-day-weighted mean of the 12 monthly hourly tariffs.
    """
    subset = rates[
        (rates["elec_provd"] == provider)
        & (rates["tariff"] == tariff)
        & (rates["year"] == int(year))
    ]

    if period in PERIOD_MONTH:
        month = PERIOD_MONTH[period]
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Expected one electricity-rate row for "
                f"{provider} / {tariff} / {year}-{month:02d}; "
                f"found {len(row)}."
            )
        return (
            row.iloc[0][RATE_HOUR_COLUMNS]
            .astype(float)
            .to_numpy()
        )

    weighted_total = np.zeros(24, dtype=float)
    annual_days = 0

    for month in range(1, 13):
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Annual average requires 12 complete monthly rate rows. "
                f"Problem at {provider} / {tariff} / {year}-{month:02d}."
            )

        days = calendar.monthrange(year, month)[1]
        rate_vector = (
            row.iloc[0][RATE_HOUR_COLUMNS]
            .astype(float)
            .to_numpy()
        )
        weighted_total += days * rate_vector
        annual_days += days

    return weighted_total / annual_days

def _latest_gas_row(
    rates: pd.DataFrame,
    provider: str,
) -> pd.Series | None:
    subset = rates[rates["gas_provd"] == provider]
    if subset.empty:
        return None

    latest_year = int(subset["year"].max())
    latest = subset[subset["year"] == latest_year].copy()
    if "source_order" in latest.columns:
        latest = latest.sort_values("source_order", kind="stable")
    return latest.iloc[0]


def _gas_monthly_cost(
    usage_vector: np.ndarray,
    rate_vector: np.ndarray,
    year: int,
    period: str,
) -> float:
    daily_cost = float(np.dot(usage_vector, rate_vector))

    if period in PERIOD_MONTH:
        month = PERIOD_MONTH[period]
        return calendar.monthrange(year, month)[1] * daily_cost

    annual_days = sum(
        calendar.monthrange(year, month)[1]
        for month in range(1, 13)
    )
    return annual_days * daily_cost / 12.0


def _summary(
    values: pd.Series,
) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "n": 0,
        }
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "n": int(clean.size),
    }


def calculate_period(
    data: CalculatorData,
    period: str,
) -> CalculationResult:
    if period not in {"1", "8", "year"}:
        raise ValueError(f"Unsupported period: {period}")

    electric_usage = _profile_table(
        data.electricity_usage,
        period,
        "electric",
    )
    gas_usage = _profile_table(
        data.gas_usage,
        period,
        "gas",
    )

    houses = (
        data.provider_map
        .merge(
            electric_usage,
            on="bldg_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            gas_usage,
            on="bldg_id",
            how="left",
            validate="one_to_one",
        )
    )

    electric_scenarios: list[dict] = []
    electric_houses: list[dict] = []
    gas_houses: list[dict] = []
    missing_electric: list[dict] = []
    missing_gas: list[dict] = []

    # Rate-profile averaging follows the same equal-household logic as the
    # displayed cost average. Electricity tariffs are averaged within each
    # household first; household rate profiles are then averaged across homes.
    electric_house_rate_vectors: list[np.ndarray] = []
    gas_house_rate_vectors: list[np.ndarray] = []

    for _, house in houses.iterrows():
        building_id = str(house["bldg_id"])
        county = house["in.county_name"]
        electric_provider = house["elec_provd"]
        gas_provider = house["gas_provd"]

        electric_vector = house[
            [f"electric_{hour}" for hour in range(24)]
        ].astype(float).to_numpy()
        gas_vector = house[
            [f"gas_{hour}" for hour in range(24)]
        ].astype(float).to_numpy()

        # --------------------------------------------------------------
        # Gas: one mapped provider rate per household.
        # --------------------------------------------------------------
        gas_row = _latest_gas_row(data.gas_rates, gas_provider)
        if gas_row is None:
            missing_gas.append({
                "bldg_id": building_id,
                "county": county,
                "gas_provider": gas_provider,
                "reason": "No complete gas-rate row",
            })
        else:
            gas_year = int(gas_row["year"])
            gas_rate_vector = (
                gas_row[RATE_HOUR_COLUMNS]
                .astype(float)
                .to_numpy()
            )
            gas_cost = _gas_monthly_cost(
                gas_vector,
                gas_rate_vector,
                gas_year,
                period,
            )
            gas_house_rate_vectors.append(gas_rate_vector)
            gas_houses.append({
                "bldg_id": building_id,
                "county": county,
                "gas_provider": gas_provider,
                "gas_tariff": gas_row["tariff"],
                "gas_year": gas_year,
                "gas_monthly_cost": gas_cost,
            })

        # --------------------------------------------------------------
        # Electricity: calculate every complete tariff applicable to the
        # household's mapped electricity provider.
        # --------------------------------------------------------------
        provider_rates = data.electricity_rates[
            data.electricity_rates["elec_provd"] == electric_provider
        ]

        tariff_costs: list[dict] = []
        for tariff in provider_rates["tariff"].drop_duplicates().tolist():
            years = _valid_electric_years(
                data.electricity_rates,
                electric_provider,
                tariff,
                period,
            )
            if not years:
                continue

            year = years[-1]
            cost = _electric_monthly_cost(
                electric_vector,
                data.electricity_rates,
                electric_provider,
                tariff,
                year,
                period,
            )
            rate_vector = _electric_period_rate_vector(
                data.electricity_rates,
                electric_provider,
                tariff,
                year,
                period,
            )
            record = {
                "bldg_id": building_id,
                "county": county,
                "electric_provider": electric_provider,
                "electric_tariff": tariff,
                "electric_year": year,
                "electric_monthly_cost": cost,
                "_rate_vector": rate_vector,
            }
            tariff_costs.append(record)
            electric_scenarios.append({
                key: value
                for key, value in record.items()
                if key != "_rate_vector"
            })

        if not tariff_costs:
            missing_electric.append({
                "bldg_id": building_id,
                "county": county,
                "electric_provider": electric_provider,
                "reason": "No complete applicable electricity tariff",
            })
            continue

        tariff_df = pd.DataFrame(tariff_costs)

        household_rate_vector = np.mean(
            np.stack(tariff_df["_rate_vector"].to_list()),
            axis=0,
        )
        electric_house_rate_vectors.append(household_rate_vector)

        min_index = tariff_df["electric_monthly_cost"].idxmin()
        max_index = tariff_df["electric_monthly_cost"].idxmax()

        electric_houses.append({
            "bldg_id": building_id,
            "county": county,
            "electric_provider": electric_provider,
            "tariff_count": len(tariff_df),
            "electric_min": float(
                tariff_df["electric_monthly_cost"].min()
            ),
            "electric_min_tariff": tariff_df.loc[
                min_index, "electric_tariff"
            ],
            "electric_max": float(
                tariff_df["electric_monthly_cost"].max()
            ),
            "electric_max_tariff": tariff_df.loc[
                max_index, "electric_tariff"
            ],
            # Equal household weighting: first average tariffs within a house.
            "electric_average": float(
                tariff_df["electric_monthly_cost"].mean()
            ),
        })

    electric_tariff_df = pd.DataFrame(electric_scenarios)
    electric_house_df = pd.DataFrame(electric_houses)
    gas_house_df = pd.DataFrame(gas_houses)
    missing_electric_df = pd.DataFrame(missing_electric)
    missing_gas_df = pd.DataFrame(missing_gas)

    if electric_house_df.empty:
        electric_summary = _summary(pd.Series(dtype=float))
    else:
        # These are the exact summary definitions requested:
        # - global minimum of each household's minimum
        # - global maximum of each household's maximum
        # - mean of each household's tariff-average cost
        electric_summary = {
            "min": float(electric_house_df["electric_min"].min()),
            "max": float(electric_house_df["electric_max"].max()),
            "mean": float(electric_house_df["electric_average"].mean()),
            "n": int(len(electric_house_df)),
        }

    gas_summary = _summary(
        gas_house_df["gas_monthly_cost"]
        if "gas_monthly_cost" in gas_house_df
        else pd.Series(dtype=float)
    )

    paired = electric_house_df.merge(
        gas_house_df,
        on=["bldg_id", "county"],
        how="inner",
        validate="one_to_one",
    )

    paired_gas_summary = _summary(
        paired["gas_monthly_cost"]
        if "gas_monthly_cost" in paired
        else pd.Series(dtype=float)
    )

    # Household-by-household 1:1 comparison:
    # each household's HPWH minimum/maximum/average is compared with that
    # same household's single gas-water-heater monthly cost.
    if not paired.empty:
        paired["difference_min"] = (
            paired["electric_min"] - paired["gas_monthly_cost"]
        )
        paired["difference_max"] = (
            paired["electric_max"] - paired["gas_monthly_cost"]
        )
        paired["difference_average"] = (
            paired["electric_average"] - paired["gas_monthly_cost"]
        )
        difference_summary = {
            "min": float(paired["difference_min"].min()),
            "max": float(paired["difference_max"].max()),
            "mean": float(paired["difference_average"].mean()),
            "n": int(len(paired)),
        }
    else:
        difference_summary = {
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "n": 0,
        }

    # Always construct the household display table from the 149-house
    # provider map first, then left-join calculations. Thus every mapped
    # household appears even if a future workbook has a missing rate.
    all_households = (
        data.provider_map[
            ["bldg_id", "in.county_name", "elec_provd", "gas_provd"]
        ]
        .rename(columns={
            "in.county_name": "county",
            "elec_provd": "electric_provider",
            "gas_provd": "gas_provider",
        })
        .merge(
            electric_house_df[[
                "bldg_id",
                "tariff_count",
                "electric_min",
                "electric_min_tariff",
                "electric_max",
                "electric_max_tariff",
                "electric_average",
            ]] if not electric_house_df.empty else pd.DataFrame(
                columns=[
                    "bldg_id", "tariff_count", "electric_min",
                    "electric_min_tariff", "electric_max",
                    "electric_max_tariff", "electric_average",
                ]
            ),
            on="bldg_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            gas_house_df[[
                "bldg_id",
                "gas_tariff",
                "gas_year",
                "gas_monthly_cost",
            ]] if not gas_house_df.empty else pd.DataFrame(
                columns=[
                    "bldg_id", "gas_tariff", "gas_year",
                    "gas_monthly_cost",
                ]
            ),
            on="bldg_id",
            how="left",
            validate="one_to_one",
        )
    )

    all_households["difference_min"] = (
        all_households["electric_min"]
        - all_households["gas_monthly_cost"]
    )
    all_households["difference_max"] = (
        all_households["electric_max"]
        - all_households["gas_monthly_cost"]
    )
    all_households["difference_average"] = (
        all_households["electric_average"]
        - all_households["gas_monthly_cost"]
    )
    all_households["electric_status"] = np.where(
        all_households["electric_average"].notna(),
        "Calculated",
        "Missing applicable electricity rate",
    )
    all_households["gas_status"] = np.where(
        all_households["gas_monthly_cost"].notna(),
        "Calculated",
        "Missing gas rate",
    )

    if electric_house_rate_vectors:
        average_electric_rate = np.mean(
            np.stack(electric_house_rate_vectors),
            axis=0,
        )
    else:
        average_electric_rate = np.full(24, np.nan)

    if gas_house_rate_vectors:
        average_gas_rate = np.mean(
            np.stack(gas_house_rate_vectors),
            axis=0,
        )
    else:
        average_gas_rate = np.full(24, np.nan)

    hourly_average_rates = pd.DataFrame({
        "hour": range(24),
        "average_electricity_rate": average_electric_rate,
        "average_gas_rate": average_gas_rate,
    })

    return CalculationResult(
        electric_tariff_costs=electric_tariff_df,
        electric_households=electric_house_df,
        gas_households=gas_house_df,
        all_households=all_households,
        paired_households=paired,
        missing_electric=missing_electric_df,
        missing_gas=missing_gas_df,
        electric_summary=electric_summary,
        gas_summary=gas_summary,
        paired_gas_summary=paired_gas_summary,
        difference_summary=difference_summary,
        hourly_average_rates=hourly_average_rates,
    )
