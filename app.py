"""
HPWH vs Gas Water Heater — Hourly Monthly Cost Calculator (v2)

The hourly usage workbooks already contain hourly average kWh.
Therefore the calculator does NOT apply a separate water-fixture
usage fraction.

HPWH monthly variable energy cost:
    D(y,m) * sum_h(E_HPWH[b,s,h] * R_e[p,y,m,h])

Gas-WH monthly variable energy cost:
    D(y,m) * sum_h(E_gas[b,s,h] * R_g[g,y,h])

D(y,m) is the actual number of calendar days in the selected month.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from data_loading import (
    PROFILE_LABELS,
    TARIFF_LABELS,
    get_electricity_rate_vector,
    get_gas_rate_vector,
    get_usage_vector,
    load_calculator_data,
)


MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

PROFILE_OPTIONS = {
    "auto": "Auto: January→January profile; August→August profile; otherwise annual",
    "1": PROFILE_LABELS["1"],
    "8": PROFILE_LABELS["8"],
    "year": PROFILE_LABELS["year"],
}


@st.cache_data(show_spinner=False)
def load_all_data():
    return load_calculator_data()


def profile_for_month(selected_option: str, month: int) -> str:
    if selected_option != "auto":
        return selected_option
    if month == 1:
        return "1"
    if month == 8:
        return "8"
    return "year"


st.set_page_config(
    page_title="HPWH vs Gas WH Cost",
    page_icon="💧",
    layout="wide",
)

st.title("HPWH vs Gas Water Heater — Monthly Cost")
st.caption(
    "Building-level hourly ResStock consumption × utility-specific hourly rates"
)

try:
    data = load_all_data()
except Exception as exc:
    st.error(f"Failed to load or validate the files in ./data: {exc}")
    st.stop()

if data.provider_mapping_warning:
    st.warning(data.provider_mapping_warning)

electricity_rates = data.electricity_rates
gas_rates = data.gas_rates
provider_map = data.provider_map

with st.sidebar:
    st.header("Inputs")

    states = sorted(
        set(electricity_rates["state"].dropna())
        & set(gas_rates["state"].dropna())
    )
    if not states:
        st.error("No state is shared by the electricity and gas rate files.")
        st.stop()

    state = st.selectbox("State", states)

    building_ids = provider_map["bldg_id"].astype(str).tolist()
    building_id = st.selectbox("Building ID", building_ids)

    provider_row = provider_map[
        provider_map["bldg_id"].astype(str) == str(building_id)
    ].iloc[0]

    mapped_electric_provider = provider_row["elec_provd"]
    mapped_gas_provider = provider_row["gas_provd"]
    county = provider_row["in.county_name"]

    st.caption(f"County: **{county}**")
    st.caption(f"Mapped electric utility: **{mapped_electric_provider}**")
    st.caption(f"Mapped gas utility: **{mapped_gas_provider}**")

    available_electric_providers = sorted(
        electricity_rates.loc[
            electricity_rates["state"] == state,
            "elec_provd",
        ].unique()
    )
    available_gas_providers = sorted(
        gas_rates.loc[
            gas_rates["state"] == state,
            "gas_provd",
        ].unique()
    )

    override = st.checkbox(
        "Override mapped utility providers",
        value=False,
        help=(
            "Useful for testing. For final building-level results, use the "
            "providers mapped to that building."
        ),
    )

    if override:
        electric_provider = st.selectbox(
            "Electric utility",
            available_electric_providers,
            index=(
                available_electric_providers.index(mapped_electric_provider)
                if mapped_electric_provider in available_electric_providers
                else 0
            ),
        )
        gas_provider = st.selectbox(
            "Gas utility",
            available_gas_providers,
            index=(
                available_gas_providers.index(mapped_gas_provider)
                if mapped_gas_provider in available_gas_providers
                else 0
            ),
        )
    else:
        electric_provider = mapped_electric_provider
        gas_provider = mapped_gas_provider

        if electric_provider not in available_electric_providers:
            st.error(
                f"No electricity-rate data are currently available for "
                f"{electric_provider}. Add that provider to the electricity-rate "
                "workbook or enable provider override for testing."
            )
            st.stop()

        if gas_provider not in available_gas_providers:
            st.error(
                f"No gas-rate data are currently available for {gas_provider}. "
                "Add that provider to the gas-rate workbook or enable provider "
                "override for testing."
            )
            st.stop()

    provider_electric_rates = electricity_rates[
        (electricity_rates["state"] == state)
        & (electricity_rates["elec_provd"] == electric_provider)
    ]

    tariff_codes = sorted(
        provider_electric_rates["tariff"].unique(),
        key=lambda code: (
            list(TARIFF_LABELS).index(code)
            if code in TARIFF_LABELS
            else len(TARIFF_LABELS),
            code,
        ),
    )
    tariff = st.selectbox(
        "Electricity rate plan",
        tariff_codes,
        format_func=lambda code: TARIFF_LABELS.get(code, code),
    )

    tariff_rates = provider_electric_rates[
        provider_electric_rates["tariff"] == tariff
    ]
    electric_years = set(tariff_rates["year"].astype(int))
    gas_years = set(
        gas_rates.loc[
            (gas_rates["state"] == state)
            & (gas_rates["gas_provd"] == gas_provider),
            "year",
        ].astype(int)
    )
    common_years = sorted(electric_years & gas_years)

    if not common_years:
        st.error(
            "The selected electricity and gas providers do not have a common rate year."
        )
        st.stop()

    year = st.selectbox(
        "Rate year",
        common_years,
        index=len(common_years) - 1,
    )

    months = sorted(
        tariff_rates.loc[
            tariff_rates["year"] == year,
            "month",
        ].astype(int).unique()
    )
    month = st.selectbox(
        "Rate month",
        months,
        format_func=lambda value: f"{value:02d} — {MONTH_NAMES[value - 1]}",
    )

    profile_option = st.selectbox(
        "Consumption profile",
        list(PROFILE_OPTIONS),
        format_func=lambda code: PROFILE_OPTIONS[code],
    )
    profile = profile_for_month(profile_option, month)

days_in_month = calendar.monthrange(int(year), int(month))[1]

try:
    hpwh_usage = get_usage_vector(
        data.electricity_usage,
        building_id,
        profile,
    )
    gas_usage = get_usage_vector(
        data.gas_usage,
        building_id,
        profile,
    )
    electric_rate = get_electricity_rate_vector(
        electricity_rates,
        state,
        electric_provider,
        tariff,
        int(year),
        int(month),
    )
    gas_rate = get_gas_rate_vector(
        gas_rates,
        state,
        gas_provider,
        int(year),
    )
except Exception as exc:
    st.error(f"Could not prepare the selected calculation: {exc}")
    st.stop()

hourly = pd.DataFrame({
    "hour": list(range(24)),
    "HPWH usage (kWh/day at hour)": hpwh_usage,
    "Electricity rate ($/kWh)": electric_rate,
    "Gas WH usage (kWh/day at hour)": gas_usage,
    "Gas rate ($/kWh)": gas_rate,
})

hourly["HPWH cost ($/day at hour)"] = (
    hourly["HPWH usage (kWh/day at hour)"]
    * hourly["Electricity rate ($/kWh)"]
)
hourly["Gas WH cost ($/day at hour)"] = (
    hourly["Gas WH usage (kWh/day at hour)"]
    * hourly["Gas rate ($/kWh)"]
)

hpwh_daily_kwh = hourly["HPWH usage (kWh/day at hour)"].sum()
gas_daily_kwh = hourly["Gas WH usage (kWh/day at hour)"].sum()
hpwh_daily_cost = hourly["HPWH cost ($/day at hour)"].sum()
gas_daily_cost = hourly["Gas WH cost ($/day at hour)"].sum()

hpwh_monthly_cost = days_in_month * hpwh_daily_cost
gas_monthly_cost = days_in_month * gas_daily_cost
savings = gas_monthly_cost - hpwh_monthly_cost

st.subheader("Monthly result")
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("HPWH variable cost", f"${hpwh_monthly_cost:,.2f}")
metric_2.metric("Gas WH variable cost", f"${gas_monthly_cost:,.2f}")
metric_3.metric(
    "HPWH savings vs Gas",
    f"${savings:,.2f}",
    delta=f"{savings:,.2f}",
)
metric_4.metric("Calendar days used", f"{days_in_month}")

st.caption(
    f"Building **{building_id}** · **{county}** · "
    f"Electric: **{electric_provider} / {TARIFF_LABELS.get(tariff, tariff)}** · "
    f"Gas: **{gas_provider}** · "
    f"Rate period: **{MONTH_NAMES[month - 1]} {year}** · "
    f"Consumption profile: **{PROFILE_LABELS[profile]}**"
)

st.info(
    "These results include variable energy charges represented in the rate "
    "workbooks. Fixed customer charges, taxes, riders, minimum bills, and "
    "tiered-block adjustments are not added unless they are already embedded "
    "in the hourly $/kWh values."
)

st.divider()
summary_1, summary_2, summary_3, summary_4 = st.columns(4)
summary_1.metric("HPWH daily energy", f"{hpwh_daily_kwh:,.3f} kWh")
summary_2.metric("Gas WH daily energy", f"{gas_daily_kwh:,.3f} kWh")
summary_3.metric("HPWH daily cost", f"${hpwh_daily_cost:,.3f}")
summary_4.metric("Gas WH daily cost", f"${gas_daily_cost:,.3f}")

st.subheader("Hourly profiles")

usage_chart = hourly.set_index("hour")[
    [
        "HPWH usage (kWh/day at hour)",
        "Gas WH usage (kWh/day at hour)",
    ]
]
rate_chart = hourly.set_index("hour")[
    [
        "Electricity rate ($/kWh)",
        "Gas rate ($/kWh)",
    ]
]
cost_chart = hourly.set_index("hour")[
    [
        "HPWH cost ($/day at hour)",
        "Gas WH cost ($/day at hour)",
    ]
]

tab_1, tab_2, tab_3, tab_4 = st.tabs(
    ["Hourly usage", "Hourly rates", "Hourly cost", "Data table"]
)

with tab_1:
    st.line_chart(usage_chart)

with tab_2:
    st.line_chart(rate_chart)

with tab_3:
    st.bar_chart(cost_chart)

with tab_4:
    st.dataframe(
        hourly.style.format({
            "HPWH usage (kWh/day at hour)": "{:.6f}",
            "Electricity rate ($/kWh)": "{:.6f}",
            "Gas WH usage (kWh/day at hour)": "{:.6f}",
            "Gas rate ($/kWh)": "{:.6f}",
            "HPWH cost ($/day at hour)": "{:.6f}",
            "Gas WH cost ($/day at hour)": "{:.6f}",
        }),
        use_container_width=True,
    )

st.divider()
st.markdown(
    r"""
### Formulas

\[
C^{HPWH}_{b,p,m,s}
=
D_{y,m}
\sum_{h=0}^{23}
E^{HPWH}_{b,s,h}
R^{e}_{p,y,m,h}
\]

\[
C^{Gas}_{b,g,m,s}
=
D_{y,m}
\sum_{h=0}^{23}
E^{Gas}_{b,s,h}
R^{g}_{g,y,h}
\]

- \(b\): building ID
- \(p\): electricity provider and tariff
- \(g\): gas provider
- \(m\): selected rate month
- \(s\): January, August, or annual hourly consumption profile
- \(D_{y,m}\): actual number of days in the selected calendar month
"""
)
