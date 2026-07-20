"""
HPWH vs Gas Water Heater — Average Household Cost Calculator (v2.3)

The calculator uses the hour-by-hour arithmetic mean across all buildings
that have valid provider mappings. Individual buildings are not selectable.

January:
    31-day January cost using January mean usage and January rates.

August:
    31-day August cost using August mean usage and August rates.

Annual average:
    Annual cost using the annual mean hourly usage profile and all 12 monthly
    electricity-rate profiles, divided by 12 to report average monthly cost.
"""

from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from data_loading import (
    PROFILE_LABELS,
    TARIFF_LABELS,
    get_average_usage_vector,
    get_electricity_rate_vector,
    get_gas_rate_vector,
    load_calculator_data,
)


PERIOD_OPTIONS = {
    "1": "January",
    "8": "August",
    "year": "Annual average",
}


@st.cache_data(show_spinner=False)
def load_all_data():
    return load_calculator_data()


def ordered_unique(series: pd.Series) -> list:
    return series.dropna().drop_duplicates().tolist()


def latest_common_year(
    electricity_rates: pd.DataFrame,
    gas_rates: pd.DataFrame,
    state: str,
    electric_provider: str,
    tariff: str,
    gas_provider: str,
) -> int:
    electric_years = set(
        electricity_rates.loc[
            (electricity_rates["state"] == state)
            & (electricity_rates["elec_provd"] == electric_provider)
            & (electricity_rates["tariff"] == tariff),
            "year",
        ].astype(int)
    )
    gas_years = set(
        gas_rates.loc[
            (gas_rates["state"] == state)
            & (gas_rates["gas_provd"] == gas_provider),
            "year",
        ].astype(int)
    )
    common = sorted(electric_years & gas_years)
    if not common:
        raise ValueError(
            "The selected electricity and gas providers have no common rate year."
        )
    return common[-1]


st.set_page_config(
    page_title="HPWH vs Gas WH Cost",
    page_icon="💧",
    layout="wide",
)

st.title("HPWH vs Gas Water Heater — Average Monthly Cost")
st.caption(
    "Mean hourly energy use across mapped homes × selected utility rates"
)

try:
    data = load_all_data()
except Exception as exc:
    st.error(f"Failed to load or validate the data files: {exc}")
    st.stop()

electricity_rates = data.electricity_rates
gas_rates = data.gas_rates
mapped_building_ids = data.provider_map["bldg_id"].astype(str).tolist()
sample_size = len(mapped_building_ids)

shared_states = [
    state
    for state in ordered_unique(electricity_rates["state"])
    if state in set(gas_rates["state"].dropna())
]
if not shared_states:
    st.error("No state is shared by the electricity and gas rate files.")
    st.stop()

# This version is designed for the Michigan dataset. When more states are added,
# the first state in the workbook is used automatically.
state = shared_states[0]

with st.sidebar:
    st.header("Inputs")

    electric_providers = ordered_unique(
        electricity_rates.loc[
            electricity_rates["state"] == state,
            "elec_provd",
        ]
    )
    if not electric_providers:
        st.error(f"No electricity providers are available for {state}.")
        st.stop()

    electric_provider = st.selectbox(
        "Electricity provider",
        electric_providers,
        index=0,
    )

    provider_rate_rows = electricity_rates[
        (electricity_rates["state"] == state)
        & (electricity_rates["elec_provd"] == electric_provider)
    ]
    tariff_codes = ordered_unique(provider_rate_rows["tariff"])
    if not tariff_codes:
        st.error(f"No tariffs are available for {electric_provider}.")
        st.stop()

    tariff = st.selectbox(
        "Electricity rate plan",
        tariff_codes,
        index=0,
        format_func=lambda code: TARIFF_LABELS.get(code, code),
    )

    gas_providers = ordered_unique(
        gas_rates.loc[
            gas_rates["state"] == state,
            "gas_provd",
        ]
    )
    if not gas_providers:
        st.error(f"No gas providers are available for {state}.")
        st.stop()

    gas_provider = st.selectbox(
        "Gas provider",
        gas_providers,
        index=0,
    )

    period = st.selectbox(
        "Period",
        list(PERIOD_OPTIONS),
        index=0,
        format_func=lambda code: PERIOD_OPTIONS[code],
    )

try:
    year = latest_common_year(
        electricity_rates=electricity_rates,
        gas_rates=gas_rates,
        state=state,
        electric_provider=electric_provider,
        tariff=tariff,
        gas_provider=gas_provider,
    )

    hpwh_usage, hpwh_n = get_average_usage_vector(
        data.electricity_usage,
        mapped_building_ids,
        period,
    )
    gas_usage, gas_n = get_average_usage_vector(
        data.gas_usage,
        mapped_building_ids,
        period,
    )
    if hpwh_n != gas_n:
        raise ValueError(
            f"Electric and gas averages use different sample sizes: "
            f"{hpwh_n} vs {gas_n}."
        )

    gas_rate = get_gas_rate_vector(
        gas_rates,
        state,
        gas_provider,
        year,
    )
except Exception as exc:
    st.error(f"Could not prepare the selected calculation: {exc}")
    st.stop()

if period in {"1", "8"}:
    month = int(period)
    days = calendar.monthrange(year, month)[1]

    try:
        electric_rate = get_electricity_rate_vector(
            electricity_rates,
            state,
            electric_provider,
            tariff,
            year,
            month,
        )
    except Exception as exc:
        st.error(f"Could not load the selected monthly electricity rate: {exc}")
        st.stop()

    hpwh_daily_cost_by_hour = [
        usage * rate for usage, rate in zip(hpwh_usage, electric_rate)
    ]
    gas_daily_cost_by_hour = [
        usage * rate for usage, rate in zip(gas_usage, gas_rate)
    ]

    hpwh_result_cost = days * sum(hpwh_daily_cost_by_hour)
    gas_result_cost = days * sum(gas_daily_cost_by_hour)
    result_label = f"{PERIOD_OPTIONS[period]} monthly cost"
    period_detail = f"{PERIOD_OPTIONS[period]} {year} · {days} calendar days"

else:
    year_rows = electricity_rates[
        (electricity_rates["state"] == state)
        & (electricity_rates["elec_provd"] == electric_provider)
        & (electricity_rates["tariff"] == tariff)
        & (electricity_rates["year"] == year)
    ]
    available_months = set(year_rows["month"].astype(int))
    required_months = set(range(1, 13))
    missing_months = sorted(required_months - available_months)
    if missing_months:
        st.error(
            "Annual average requires all 12 monthly electricity-rate rows. "
            f"Missing months: {missing_months}"
        )
        st.stop()

    annual_days = 0
    hpwh_annual_cost = 0.0
    weighted_electric_rate = [0.0] * 24

    for month in range(1, 13):
        days = calendar.monthrange(year, month)[1]
        month_rate = get_electricity_rate_vector(
            electricity_rates,
            state,
            electric_provider,
            tariff,
            year,
            month,
        )

        annual_days += days
        hpwh_annual_cost += days * sum(
            usage * rate for usage, rate in zip(hpwh_usage, month_rate)
        )
        weighted_electric_rate = [
            current + days * rate
            for current, rate in zip(weighted_electric_rate, month_rate)
        ]

    electric_rate = [
        total / annual_days for total in weighted_electric_rate
    ]
    hpwh_daily_cost_by_hour = [
        usage * rate for usage, rate in zip(hpwh_usage, electric_rate)
    ]
    gas_daily_cost_by_hour = [
        usage * rate for usage, rate in zip(gas_usage, gas_rate)
    ]

    gas_annual_cost = annual_days * sum(gas_daily_cost_by_hour)

    hpwh_result_cost = hpwh_annual_cost / 12
    gas_result_cost = gas_annual_cost / 12
    result_label = "Average monthly cost"
    period_detail = (
        f"Annual profile and all 12 monthly rates for {year} · "
        f"annual total divided by 12"
    )

savings = gas_result_cost - hpwh_result_cost

hourly = pd.DataFrame({
    "hour": range(24),
    "Average HPWH use (kWh/hour)": hpwh_usage,
    "Electricity rate ($/kWh)": electric_rate,
    "Average gas-WH use (kWh/hour)": gas_usage,
    "Gas rate ($/kWh)": gas_rate,
    "HPWH cost ($/average day by hour)": hpwh_daily_cost_by_hour,
    "Gas-WH cost ($/average day by hour)": gas_daily_cost_by_hour,
})

st.subheader(result_label)
metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("HPWH", f"${hpwh_result_cost:,.2f}")
metric_2.metric("Gas water heater", f"${gas_result_cost:,.2f}")
metric_3.metric(
    "HPWH savings vs Gas",
    f"${savings:,.2f}",
)

st.caption(
    f"State: **{state}** · Rate year: **{year}** · "
    f"Electricity: **{electric_provider} / "
    f"{TARIFF_LABELS.get(tariff, tariff)}** · "
    f"Gas: **{gas_provider}** · "
    f"Consumption: **{PROFILE_LABELS[period]} across {sample_size} homes**"
)
st.caption(period_detail)

if data.excluded_building_ids:
    st.caption(
        f"{len(data.excluded_building_ids)} usage-file homes without provider "
        "mapping are excluded from the average."
    )

st.info(
    "The displayed costs include the variable energy rates represented in the "
    "workbooks. Separate fixed customer charges, taxes, riders, minimum bills, "
    "or tiered adjustments are not added unless already embedded in those rates."
)

st.divider()

summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric(
    "Average HPWH energy",
    f"{sum(hpwh_usage):,.3f} kWh/day",
)
summary_2.metric(
    "Average gas-WH energy",
    f"{sum(gas_usage):,.3f} kWh/day",
)
summary_3.metric(
    "HPWH cost per average day",
    f"${sum(hpwh_daily_cost_by_hour):,.3f}",
)
summary_4.metric(
    "Gas-WH cost per average day",
    f"${sum(gas_daily_cost_by_hour):,.3f}",
)

st.subheader("Average hourly profiles")

tab_1, tab_2, tab_3, tab_4 = st.tabs(
    ["Hourly usage", "Hourly rates", "Hourly cost", "Data table"]
)

with tab_1:
    st.line_chart(
        hourly.set_index("hour")[
            [
                "Average HPWH use (kWh/hour)",
                "Average gas-WH use (kWh/hour)",
            ]
        ]
    )

with tab_2:
    st.line_chart(
        hourly.set_index("hour")[
            [
                "Electricity rate ($/kWh)",
                "Gas rate ($/kWh)",
            ]
        ]
    )

with tab_3:
    st.bar_chart(
        hourly.set_index("hour")[
            [
                "HPWH cost ($/average day by hour)",
                "Gas-WH cost ($/average day by hour)",
            ]
        ]
    )

with tab_4:
    st.dataframe(
        hourly.style.format({
            "Average HPWH use (kWh/hour)": "{:.6f}",
            "Electricity rate ($/kWh)": "{:.6f}",
            "Average gas-WH use (kWh/hour)": "{:.6f}",
            "Gas rate ($/kWh)": "{:.6f}",
            "HPWH cost ($/average day by hour)": "{:.6f}",
            "Gas-WH cost ($/average day by hour)": "{:.6f}",
        }),
        use_container_width=True,
    )

st.divider()
st.markdown(
    r"""
### Calculation

For January and August, each hourly usage value is first averaged across the
mapped homes. The monthly variable cost is:

\[
C_m
=
D_m
\sum_{h=0}^{23}
\overline{E}_{s,h} R_{m,h}
\]

For the annual-average selection, the annual usage profile is combined with
each of the 12 monthly electricity-rate profiles. The resulting annual cost is
divided by 12.
"""
)
