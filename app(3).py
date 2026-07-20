"""
HPWH vs Gas Water Heater — Household Range Calculator (v2.4)

Only the consumption period is selectable. Each household is automatically
matched to its electricity and gas providers. Every complete electricity
tariff available for that household's provider is evaluated.

Displayed range:
- Minimum and maximum across all valid household–tariff scenarios.
- Average gives every household equal weight: tariff costs are first averaged
  within each household, then household means are averaged.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from data_loading import (
    PROFILE_LABELS,
    RATE_HOUR_COLUMNS,
    TARIFF_LABELS,
    USAGE_HOUR_COLUMNS,
    load_calculator_data,
)


PERIOD_OPTIONS = {
    "year": "Annual average",
    "1": "January",
    "8": "August",
}


@st.cache_data(show_spinner=False)
def load_all_data():
    return load_calculator_data()


def tariff_display(code: str) -> str:
    label = TARIFF_LABELS.get(str(code), str(code))
    return f"{label} ({code})" if label != code else label


def profile_usage_table(
    usage: pd.DataFrame,
    profile: str,
    prefix: str,
) -> pd.DataFrame:
    selected = usage[usage["season"] == profile][
        ["bldg_id", *USAGE_HOUR_COLUMNS]
    ].copy()
    rename = {
        column: f"{prefix}_{hour}"
        for hour, column in enumerate(USAGE_HOUR_COLUMNS)
    }
    return selected.rename(columns=rename)


def valid_electric_years(
    rates: pd.DataFrame,
    provider: str,
    tariff: str,
    period: str,
) -> set[int]:
    subset = rates[
        (rates["elec_provd"] == provider)
        & (rates["tariff"] == tariff)
    ]
    if period in {"1", "8"}:
        month = int(period)
        return set(
            subset.loc[subset["month"] == month, "year"].astype(int)
        )

    years: set[int] = set()
    for year, group in subset.groupby("year"):
        if set(group["month"].astype(int)) >= set(range(1, 13)):
            years.add(int(year))
    return years


def electric_cost(
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
        & (rates["year"] == year)
    ]

    if period in {"1", "8"}:
        month = int(period)
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Expected one electricity row for {provider} / {tariff} / "
                f"{year}-{month:02d}; found {len(row)}."
            )
        rate_vector = row.iloc[0][RATE_HOUR_COLUMNS].astype(float).to_numpy()
        days = calendar.monthrange(year, month)[1]
        return float(days * np.dot(usage_vector, rate_vector))

    annual_cost = 0.0
    for month in range(1, 13):
        row = subset[subset["month"] == month]
        if len(row) != 1:
            raise ValueError(
                f"Annual average requires one rate row for each month. "
                f"Missing/duplicate: {provider} / {tariff} / {year}-{month:02d}."
            )
        rate_vector = row.iloc[0][RATE_HOUR_COLUMNS].astype(float).to_numpy()
        days = calendar.monthrange(year, month)[1]
        annual_cost += days * float(np.dot(usage_vector, rate_vector))
    return annual_cost / 12.0


def gas_cost(
    usage_vector: np.ndarray,
    rate_vector: np.ndarray,
    year: int,
    period: str,
) -> float:
    daily_cost = float(np.dot(usage_vector, rate_vector))
    if period in {"1", "8"}:
        month = int(period)
        return calendar.monthrange(year, month)[1] * daily_cost
    return sum(calendar.monthrange(year, month)[1] for month in range(1, 13)) * daily_cost / 12.0


def range_summary(
    scenario_df: pd.DataFrame,
    value_column: str,
) -> dict[str, float | int]:
    if scenario_df.empty:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "n": 0}

    # Each household receives equal weight in the displayed average even when
    # providers offer different numbers of tariffs.
    household_mean = scenario_df.groupby("bldg_id")[value_column].mean()
    return {
        "min": float(scenario_df[value_column].min()),
        "max": float(scenario_df[value_column].max()),
        "mean": float(household_mean.mean()),
        "n": int(household_mean.size),
    }


def simple_house_summary(
    house_df: pd.DataFrame,
    value_column: str,
) -> dict[str, float | int]:
    clean = house_df[value_column].dropna()
    return {
        "min": float(clean.min()) if not clean.empty else np.nan,
        "max": float(clean.max()) if not clean.empty else np.nan,
        "mean": float(clean.mean()) if not clean.empty else np.nan,
        "n": int(clean.size),
    }


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def metric_card(
    column,
    label: str,
    summary: dict[str, float | int],
    denominator: int,
) -> None:
    column.metric(
        label,
        f"{money(summary['min'])} – {money(summary['max'])}",
    )
    column.caption(
        f"Average: **{money(summary['mean'])}** · "
        f"Households: **{summary['n']}/{denominator}**"
    )


def latest_gas_row(
    gas_rates: pd.DataFrame,
    provider: str,
) -> pd.Series | None:
    subset = gas_rates[gas_rates["gas_provd"] == provider]
    if subset.empty:
        return None
    latest_year = int(subset["year"].max())
    latest = subset[subset["year"] == latest_year].sort_values("source_order")
    return latest.iloc[0]


def selected_electric_rate_rows(
    electricity_rates: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for (provider, tariff), group in electricity_rates.groupby(
        ["elec_provd", "tariff"],
        sort=False,
    ):
        years = valid_electric_years(
            electricity_rates,
            provider,
            tariff,
            period,
        )
        if not years:
            continue
        year = max(years)
        chosen = group[group["year"] == year]
        if period in {"1", "8"}:
            chosen = chosen[chosen["month"] == int(period)]
        else:
            chosen = chosen[chosen["month"].isin(range(1, 13))]
        records.append(chosen)

    if not records:
        return electricity_rates.iloc[0:0].copy()
    return pd.concat(records, ignore_index=True)


st.set_page_config(
    page_title="HPWH vs Gas WH Range",
    page_icon="💧",
    layout="wide",
)
st.title("HPWH vs Gas Water Heater — Household Cost Range")
st.caption(
    "Mapped utility providers × all available electricity tariffs × "
    "household-level hourly energy use"
)

try:
    data = load_all_data()
except Exception as exc:
    st.error(f"Failed to load or validate the data files: {exc}")
    st.stop()

with st.sidebar:
    st.header("Input")
    period = st.selectbox(
        "Consumption period",
        list(PERIOD_OPTIONS),
        index=0,
        format_func=lambda code: PERIOD_OPTIONS[code],
    )

profile_label = PROFILE_LABELS[period]
mapped = data.provider_map.copy()
total_houses = len(mapped)

electric_usage = profile_usage_table(
    data.electricity_usage,
    period,
    "electric_usage",
)
gas_usage = profile_usage_table(
    data.gas_usage,
    period,
    "gas_usage",
)

houses = (
    mapped
    .merge(electric_usage, on="bldg_id", how="left", validate="one_to_one")
    .merge(gas_usage, on="bldg_id", how="left", validate="one_to_one")
)

electric_scenarios: list[dict] = []
gas_house_records: list[dict] = []
missing_electric: list[dict] = []
missing_gas: list[dict] = []

# Calculate one gas result per household using the provider's latest rate year.
for _, house in houses.iterrows():
    gas_row = latest_gas_row(data.gas_rates, house["gas_provd"])
    if gas_row is None:
        missing_gas.append({
            "bldg_id": house["bldg_id"],
            "county": house["in.county_name"],
            "gas_provider": house["gas_provd"],
            "reason": "No gas rate",
        })
        continue

    gas_vector = house[
        [f"gas_usage_{hour}" for hour in range(24)]
    ].astype(float).to_numpy()
    gas_rate_vector = gas_row[RATE_HOUR_COLUMNS].astype(float).to_numpy()
    year = int(gas_row["year"])
    cost = gas_cost(gas_vector, gas_rate_vector, year, period)

    gas_house_records.append({
        "bldg_id": house["bldg_id"],
        "county": house["in.county_name"],
        "gas_provider": house["gas_provd"],
        "gas_tariff": gas_row["tariff"],
        "gas_year": year,
        "gas_cost": cost,
    })

gas_house_df = pd.DataFrame(gas_house_records)

# Calculate every complete tariff for each household's mapped electric provider.
for _, house in houses.iterrows():
    provider = house["elec_provd"]
    gas_provider = house["gas_provd"]
    provider_rates = data.electricity_rates[
        data.electricity_rates["elec_provd"] == provider
    ]

    if provider_rates.empty:
        missing_electric.append({
            "bldg_id": house["bldg_id"],
            "county": house["in.county_name"],
            "electric_provider": provider,
            "reason": "No electricity-rate rows",
        })
        continue

    electric_vector = house[
        [f"electric_usage_{hour}" for hour in range(24)]
    ].astype(float).to_numpy()
    gas_vector = house[
        [f"gas_usage_{hour}" for hour in range(24)]
    ].astype(float).to_numpy()

    valid_tariff_found = False
    for tariff in provider_rates["tariff"].drop_duplicates():
        electric_year_set = valid_electric_years(
            data.electricity_rates,
            provider,
            tariff,
            period,
        )
        gas_year_set = set(
            data.gas_rates.loc[
                data.gas_rates["gas_provd"] == gas_provider,
                "year",
            ].astype(int)
        )
        common_years = sorted(electric_year_set & gas_year_set)
        if not common_years:
            continue

        year = common_years[-1]
        gas_candidates = data.gas_rates[
            (data.gas_rates["gas_provd"] == gas_provider)
            & (data.gas_rates["year"] == year)
        ].sort_values("source_order")
        if gas_candidates.empty:
            continue

        gas_row = gas_candidates.iloc[0]
        gas_rate_vector = gas_row[RATE_HOUR_COLUMNS].astype(float).to_numpy()

        hpwh_cost = electric_cost(
            electric_vector,
            data.electricity_rates,
            provider,
            tariff,
            year,
            period,
        )
        gas_same_year_cost = gas_cost(
            gas_vector,
            gas_rate_vector,
            year,
            period,
        )

        electric_scenarios.append({
            "bldg_id": house["bldg_id"],
            "county": house["in.county_name"],
            "electric_provider": provider,
            "electric_tariff": tariff,
            "gas_provider": gas_provider,
            "gas_tariff": gas_row["tariff"],
            "year": year,
            "hpwh_cost": hpwh_cost,
            "gas_cost": gas_same_year_cost,
            "savings": gas_same_year_cost - hpwh_cost,
        })
        valid_tariff_found = True

    if not valid_tariff_found:
        missing_electric.append({
            "bldg_id": house["bldg_id"],
            "county": house["in.county_name"],
            "electric_provider": provider,
            "reason": "No complete tariff with a common gas-rate year",
        })

scenario_df = pd.DataFrame(electric_scenarios)
missing_electric_df = pd.DataFrame(missing_electric).drop_duplicates(
    subset=["bldg_id"]
) if missing_electric else pd.DataFrame()
missing_gas_df = pd.DataFrame(missing_gas)

hpwh_summary = range_summary(scenario_df, "hpwh_cost")
gas_summary = simple_house_summary(gas_house_df, "gas_cost")
savings_summary = range_summary(scenario_df, "savings")

st.subheader(f"Results — {profile_label}")
result_1, result_2, result_3 = st.columns(3)
metric_card(result_1, "HPWH", hpwh_summary, total_houses)
metric_card(result_2, "Gas water heater", gas_summary, total_houses)
metric_card(result_3, "HPWH savings vs Gas WH", savings_summary, total_houses)

st.caption(
    "Ranges use all valid household–tariff scenarios. The displayed average "
    "is household-weighted: applicable tariff results are averaged within each "
    "household first, and then averaged across households."
)

if hpwh_summary["n"] < total_houses or gas_summary["n"] < total_houses:
    st.warning(
        "The current rate workbooks do not yet cover every mapped utility. "
        f"HPWH/savings coverage: {hpwh_summary['n']}/{total_houses}; "
        f"gas coverage: {gas_summary['n']}/{total_houses}. "
        "The app will automatically reach 149/149 as complete provider-rate "
        "rows are added without changing the workbook structure."
    )

st.divider()

# ------------------------------------------------------------------
# (1) House sampling
# ------------------------------------------------------------------
st.subheader("(1) House sampling")

sample_summary = pd.DataFrame({
    "Item": [
        "Buildings in hourly-usage files",
        "Buildings with county/provider mapping",
        "Buildings excluded because provider mapping is missing",
        "HPWH/savings calculation coverage",
        "Gas-WH calculation coverage",
        "Selected consumption profile",
    ],
    "Value": [
        data.electricity_usage["bldg_id"].nunique(),
        total_houses,
        len(data.excluded_building_ids),
        f"{hpwh_summary['n']} / {total_houses}",
        f"{gas_summary['n']} / {total_houses}",
        profile_label,
    ],
})
st.dataframe(sample_summary, use_container_width=True, hide_index=True)

county_sample = (
    mapped.groupby("in.county_name", as_index=False)
    .agg(households=("bldg_id", "nunique"))
    .sort_values(["households", "in.county_name"], ascending=[False, True])
)
county_sample["share (%)"] = county_sample["households"] / total_houses * 100

with st.expander("County distribution of sampled households", expanded=False):
    st.dataframe(
        county_sample.style.format({"share (%)": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------
# (2) County-level utility providers
# ------------------------------------------------------------------
st.subheader("(2) Electricity and gas providers by county")

county_provider = (
    mapped.groupby(
        ["in.county_name", "elec_provd", "gas_provd"],
        as_index=False,
    )
    .agg(households=("bldg_id", "nunique"))
    .sort_values(
        ["in.county_name", "households", "elec_provd", "gas_provd"],
        ascending=[True, False, True, True],
    )
)
st.dataframe(county_provider, use_container_width=True, hide_index=True)

provider_coverage = (
    mapped.groupby(["elec_provd"], as_index=False)
    .agg(mapped_households=("bldg_id", "nunique"))
)
available_electric_providers = set(data.electricity_rates["elec_provd"])
provider_coverage["complete rate data loaded"] = provider_coverage[
    "elec_provd"
].isin(available_electric_providers)
provider_coverage = provider_coverage.sort_values(
    ["complete rate data loaded", "mapped_households"],
    ascending=[False, False],
)

with st.expander("Electricity-provider rate coverage", expanded=False):
    st.dataframe(provider_coverage, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# (3) Utility rates, including tariff types
# ------------------------------------------------------------------
st.subheader("(3) Electricity and gas rates, including tariff type")

selected_electric = selected_electric_rate_rows(
    data.electricity_rates,
    period,
)

mapped_electric_counts = (
    mapped.groupby("elec_provd")["bldg_id"].nunique().to_dict()
)
electric_rate_summary_records = []
for (provider, tariff, year), group in selected_electric.groupby(
    ["elec_provd", "tariff", "year"],
    sort=False,
):
    values = group[RATE_HOUR_COLUMNS].astype(float).to_numpy().ravel()
    months = sorted(group["month"].astype(int).unique())
    electric_rate_summary_records.append({
        "electricity provider": provider,
        "tariff type": tariff_display(tariff),
        "year": int(year),
        "months represented": (
            ", ".join(map(str, months)) if period == "year" else months[0]
        ),
        "mapped households": mapped_electric_counts.get(provider, 0),
        "minimum rate ($/kWh)": float(np.min(values)),
        "average rate ($/kWh)": float(np.mean(values)),
        "maximum rate ($/kWh)": float(np.max(values)),
    })

electric_rate_summary = pd.DataFrame(electric_rate_summary_records)
st.markdown("**Electricity rates**")
if electric_rate_summary.empty:
    st.info("No complete electricity-rate rows are available for this period.")
else:
    st.dataframe(
        electric_rate_summary.style.format({
            "minimum rate ($/kWh)": "{:.6f}",
            "average rate ($/kWh)": "{:.6f}",
            "maximum rate ($/kWh)": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

electric_detail_columns = [
    "elec_provd", "tariff", "year", "month", *RATE_HOUR_COLUMNS
]
electric_detail = selected_electric[electric_detail_columns].copy()
electric_detail = electric_detail.rename(columns={
    "elec_provd": "electricity provider",
    "tariff": "tariff type",
})
electric_detail["tariff type"] = electric_detail["tariff type"].map(
    tariff_display
)

with st.expander("Electricity hourly-rate detail", expanded=False):
    st.dataframe(
        electric_detail.style.format({
            column: "{:.6f}" for column in RATE_HOUR_COLUMNS
        }),
        use_container_width=True,
        hide_index=True,
    )

mapped_gas_counts = mapped.groupby("gas_provd")["bldg_id"].nunique().to_dict()
gas_summary_records = []
gas_detail_records = []

for provider, group in data.gas_rates.groupby("gas_provd", sort=False):
    latest_year = int(group["year"].max())
    rows = group[group["year"] == latest_year].sort_values("source_order")
    for _, row in rows.iterrows():
        values = row[RATE_HOUR_COLUMNS].astype(float).to_numpy()
        gas_summary_records.append({
            "gas provider": provider,
            "tariff type": tariff_display(row["tariff"]),
            "year": latest_year,
            "mapped households": mapped_gas_counts.get(provider, 0),
            "minimum rate ($/kWh)": float(np.min(values)),
            "average rate ($/kWh)": float(np.mean(values)),
            "maximum rate ($/kWh)": float(np.max(values)),
        })
        detail = {
            "gas provider": provider,
            "tariff type": tariff_display(row["tariff"]),
            "year": latest_year,
        }
        detail.update({
            column: float(row[column]) for column in RATE_HOUR_COLUMNS
        })
        gas_detail_records.append(detail)

gas_rate_summary = pd.DataFrame(gas_summary_records)
gas_rate_detail = pd.DataFrame(gas_detail_records)

st.markdown("**Gas rates**")
st.dataframe(
    gas_rate_summary.style.format({
        "minimum rate ($/kWh)": "{:.6f}",
        "average rate ($/kWh)": "{:.6f}",
        "maximum rate ($/kWh)": "{:.6f}",
    }),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Gas hourly-rate detail", expanded=False):
    st.dataframe(
        gas_rate_detail.style.format({
            column: "{:.6f}" for column in RATE_HOUR_COLUMNS
        }),
        use_container_width=True,
        hide_index=True,
    )

# Calculation transparency
with st.expander("Household–tariff calculation detail", expanded=False):
    if scenario_df.empty:
        st.info("No complete household–tariff scenarios are available.")
    else:
        display_scenarios = scenario_df.copy()
        display_scenarios["electric_tariff"] = display_scenarios[
            "electric_tariff"
        ].map(tariff_display)
        st.dataframe(
            display_scenarios.style.format({
                "hpwh_cost": "${:,.2f}",
                "gas_cost": "${:,.2f}",
                "savings": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

if not missing_electric_df.empty or not missing_gas_df.empty:
    with st.expander("Missing-rate records", expanded=False):
        if not missing_electric_df.empty:
            st.markdown("**Missing electricity-rate coverage**")
            st.dataframe(
                missing_electric_df,
                use_container_width=True,
                hide_index=True,
            )
        if not missing_gas_df.empty:
            st.markdown("**Missing gas-rate coverage**")
            st.dataframe(
                missing_gas_df,
                use_container_width=True,
                hide_index=True,
            )

st.divider()
st.markdown(
    r"""
### Cost definitions

For January or August:

\[
C_{i,p,m}
=
D_m
\sum_{h=0}^{23}
E_{i,m,h}R_{p,m,h}
\]

For annual average, the annual cost uses all 12 monthly electricity-rate
profiles and actual calendar days, then is divided by 12.

Savings are calculated for the same household, tariff scenario, and rate year:

\[
S_{i,p}=C^{Gas}_{i}-C^{HPWH}_{i,p}
\]
"""
)
