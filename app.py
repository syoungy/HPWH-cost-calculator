from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from calculations import calculate_period
from data_loading import (
    PROFILE_LABELS,
    RATE_HOUR_COLUMNS,
    TARIFF_LABELS,
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


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def tariff_name(code: object) -> str:
    text = str(code)
    label = TARIFF_LABELS.get(text, text)
    return f"{label} ({text})" if label != text else label


def show_summary_card(
    column,
    title: str,
    summary: dict[str, float | int],
    denominator: int,
    difference: bool = False,
) -> None:
    if difference:
        column.metric(
            title,
            f"{money(summary['min'])} / "
            f"{money(summary['max'])} / "
            f"{money(summary['mean'])}",
        )
        column.caption(
            "**Min / Max / Average** · "
            f"electric calculation coverage: **{summary['n']}/{denominator}**"
        )
    else:
        column.metric(
            title,
            f"{money(summary['min'])} – {money(summary['max'])}",
        )
        column.caption(
            f"Average: **{money(summary['mean'])}** · "
            f"households: **{summary['n']}/{denominator}**"
        )


def latest_rate_rows(
    rates: pd.DataFrame,
    provider_column: str,
    period: str,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    group_columns = [provider_column, "tariff"]

    for _, group in rates.groupby(group_columns, sort=False):
        if period in {"1", "8"}:
            month = int(period)
            eligible = group[group["month"] == month]
            if eligible.empty:
                continue
            year = int(eligible["year"].max())
            chosen = eligible[eligible["year"] == year]
        else:
            valid_years = []
            for year, year_group in group.groupby("year"):
                if set(year_group["month"].astype(int)) >= set(range(1, 13)):
                    valid_years.append(int(year))
            if not valid_years:
                continue
            year = max(valid_years)
            chosen = group[
                (group["year"] == year)
                & (group["month"].isin(range(1, 13)))
            ]

        records.append(chosen)

    if not records:
        return rates.iloc[0:0].copy()
    return pd.concat(records, ignore_index=True)


st.set_page_config(
    page_title="HPWH vs Gas WH",
    page_icon="💧",
    layout="wide",
)

st.title("HPWH vs Gas Water Heater — Household Monthly Cost")
st.caption(
    "Each household is calculated separately using its mapped utility "
    "provider and every complete applicable electricity tariff."
)

try:
    data = load_all_data()
except Exception as exc:
    st.error(f"Failed to load or validate data: {exc}")
    st.stop()

with st.sidebar:
    st.header("Input")
    period = st.selectbox(
        "Consumption profile",
        list(PERIOD_OPTIONS),
        index=0,
        format_func=lambda code: PERIOD_OPTIONS[code],
    )

try:
    result = calculate_period(data, period)
except Exception as exc:
    st.error(f"Calculation failed: {exc}")
    st.stop()

total_houses = len(data.provider_map)
profile_label = PROFILE_LABELS[period]

st.subheader(f"Results — {profile_label}")
card_1, card_2, card_3 = st.columns(3)

show_summary_card(
    card_1,
    "HPWH monthly electricity cost",
    result.electric_summary,
    total_houses,
)
show_summary_card(
    card_2,
    "Gas-water-heater monthly cost",
    result.gas_summary,
    total_houses,
)
show_summary_card(
    card_3,
    "HPWH − Gas WH cost difference",
    result.difference_summary,
    total_houses,
    difference=True,
)

st.caption(
    "Cost difference follows the requested direction: "
    "**HPWH electricity cost − Gas-WH cost**. "
    "A negative value means HPWH is cheaper; a positive value means HPWH "
    "is more expensive."
)

st.markdown(
    """
**Result definitions**

- HPWH minimum: the lowest value among every household's minimum tariff cost.
- HPWH maximum: the highest value among every household's maximum tariff cost.
- HPWH average: tariffs are averaged within each household first, and those
  household averages are then averaged.
- Gas minimum, maximum, and average: calculated from household gas costs.
- Cost difference: electric minimum − gas minimum; electric maximum −
  gas maximum; electric average − gas average.
"""
)

electric_n = int(result.electric_summary["n"])
gas_n = int(result.gas_summary["n"])
paired_n = int(result.difference_summary["n"])

if electric_n < total_houses:
    missing_by_provider = (
        result.missing_electric
        .groupby("electric_provider", as_index=False)
        .agg(missing_households=("bldg_id", "nunique"))
        .sort_values(
            ["missing_households", "electric_provider"],
            ascending=[False, True],
        )
    )
    st.warning(
        f"The current electricity-rate workbook can calculate "
        f"**{electric_n}/{total_houses} households**. "
        f"The remaining {total_houses - electric_n} households have mapped "
        "electric utilities but no complete 24-hour rate rows in the workbook. "
        "They are not assigned another utility's rate."
    )
    with st.expander("Electric providers still missing complete rates"):
        st.dataframe(
            missing_by_provider,
            use_container_width=True,
            hide_index=True,
        )

if gas_n < total_houses:
    st.warning(
        f"Gas calculation coverage is {gas_n}/{total_houses} households."
    )

st.caption(
    f"Difference statistics use the displayed electricity statistics "
    f"({electric_n} households) and displayed gas statistics "
    f"({gas_n} households), exactly as electric statistic minus gas statistic."
)

st.divider()

# ---------------------------------------------------------------------
# Household-level calculation tables
# ---------------------------------------------------------------------
st.subheader("Household-level calculation")

tab_1, tab_2, tab_3, tab_4 = st.tabs(
    [
        "Household minimum / maximum",
        "Electric tariff calculations",
        "Gas calculations",
        "Paired HPWH − Gas",
    ]
)

with tab_1:
    display = result.electric_households.copy()
    if not display.empty:
        display["electric_min_tariff"] = display[
            "electric_min_tariff"
        ].map(tariff_name)
        display["electric_max_tariff"] = display[
            "electric_max_tariff"
        ].map(tariff_name)
        st.dataframe(
            display.style.format({
                "electric_min": "${:,.2f}",
                "electric_max": "${:,.2f}",
                "electric_average": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No household electricity results are available.")

with tab_2:
    display = result.electric_tariff_costs.copy()
    if not display.empty:
        display["electric_tariff"] = display[
            "electric_tariff"
        ].map(tariff_name)
        st.dataframe(
            display.style.format({
                "electric_monthly_cost": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No electric tariff calculations are available.")

with tab_3:
    display = result.gas_households.copy()
    if not display.empty:
        display["gas_tariff"] = display["gas_tariff"].map(tariff_name)
        st.dataframe(
            display.style.format({
                "gas_monthly_cost": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No gas calculations are available.")

with tab_4:
    display = result.paired_households.copy()
    if not display.empty:
        st.dataframe(
            display.style.format({
                "electric_min": "${:,.2f}",
                "electric_max": "${:,.2f}",
                "electric_average": "${:,.2f}",
                "gas_monthly_cost": "${:,.2f}",
                "difference_min": "${:,.2f}",
                "difference_max": "${:,.2f}",
                "difference_average": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No paired calculations are available.")

st.divider()

# ---------------------------------------------------------------------
# (1) House sampling
# ---------------------------------------------------------------------
st.subheader("(1) House sampling")

sample_summary = pd.DataFrame({
    "Item": [
        "Unique buildings in usage files",
        "Buildings with county/provider mapping",
        "Buildings excluded because provider mapping is missing",
        "Electric calculation coverage",
        "Gas calculation coverage",
        "Households available in the separate paired-detail table",
        "Selected profile",
    ],
    "Value": [
        data.electricity_usage["bldg_id"].nunique(),
        total_houses,
        len(data.excluded_building_ids),
        f"{electric_n} / {total_houses}",
        f"{gas_n} / {total_houses}",
        f"{paired_n} / {total_houses}",
        profile_label,
    ],
})
st.dataframe(
    sample_summary,
    use_container_width=True,
    hide_index=True,
)

county_sample = (
    data.provider_map
    .groupby("in.county_name", as_index=False)
    .agg(households=("bldg_id", "nunique"))
    .sort_values(
        ["households", "in.county_name"],
        ascending=[False, True],
    )
)
county_sample["share (%)"] = (
    county_sample["households"] / total_houses * 100
)

with st.expander("County distribution"):
    st.dataframe(
        county_sample.style.format({"share (%)": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------
# (2) County/provider mapping
# ---------------------------------------------------------------------
st.subheader("(2) Electricity and gas providers by county")

county_provider = (
    data.provider_map
    .groupby(
        ["in.county_name", "elec_provd", "gas_provd"],
        as_index=False,
    )
    .agg(households=("bldg_id", "nunique"))
    .sort_values(
        ["in.county_name", "households", "elec_provd", "gas_provd"],
        ascending=[True, False, True, True],
    )
)
st.dataframe(
    county_provider,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------
# (3) Loaded electricity and gas rates
# ---------------------------------------------------------------------
st.subheader("(3) Electricity and gas rates, including tariff type")

electric_rows = latest_rate_rows(
    data.electricity_rates,
    "elec_provd",
    period,
)

electric_provider_counts = (
    data.provider_map
    .groupby("elec_provd")["bldg_id"]
    .nunique()
    .rename("mapped_households")
)

if not electric_rows.empty:
    electric_rows = electric_rows.copy()
    electric_rows["mapped_households"] = (
        electric_rows["elec_provd"]
        .map(electric_provider_counts)
        .fillna(0)
        .astype(int)
    )
    electric_rows["tariff_display"] = electric_rows[
        "tariff"
    ].map(tariff_name)
    electric_rows["hourly_rate_min"] = electric_rows[
        RATE_HOUR_COLUMNS
    ].min(axis=1)
    electric_rows["hourly_rate_mean"] = electric_rows[
        RATE_HOUR_COLUMNS
    ].mean(axis=1)
    electric_rows["hourly_rate_max"] = electric_rows[
        RATE_HOUR_COLUMNS
    ].max(axis=1)

    electric_columns = [
        "elec_provd",
        "tariff_display",
        "year",
        "month",
        "mapped_households",
        "hourly_rate_min",
        "hourly_rate_mean",
        "hourly_rate_max",
        *RATE_HOUR_COLUMNS,
    ]
    st.markdown("**Electricity rates used**")
    st.dataframe(
        electric_rows[electric_columns].style.format({
            "hourly_rate_min": "${:.6f}",
            "hourly_rate_mean": "${:.6f}",
            "hourly_rate_max": "${:.6f}",
            **{column: "${:.6f}" for column in RATE_HOUR_COLUMNS},
        }),
        use_container_width=True,
        hide_index=True,
    )

gas_rows = data.gas_rates.copy()
if not gas_rows.empty:
    latest_year_by_provider = (
        gas_rows.groupby("gas_provd")["year"].transform("max")
    )
    gas_rows = gas_rows[gas_rows["year"] == latest_year_by_provider].copy()

    gas_provider_counts = (
        data.provider_map
        .groupby("gas_provd")["bldg_id"]
        .nunique()
        .rename("mapped_households")
    )
    gas_rows["mapped_households"] = (
        gas_rows["gas_provd"]
        .map(gas_provider_counts)
        .fillna(0)
        .astype(int)
    )
    gas_rows["tariff_display"] = gas_rows["tariff"].map(tariff_name)
    gas_rows["hourly_rate_min"] = gas_rows[
        RATE_HOUR_COLUMNS
    ].min(axis=1)
    gas_rows["hourly_rate_mean"] = gas_rows[
        RATE_HOUR_COLUMNS
    ].mean(axis=1)
    gas_rows["hourly_rate_max"] = gas_rows[
        RATE_HOUR_COLUMNS
    ].max(axis=1)

    gas_columns = [
        "gas_provd",
        "tariff_display",
        "year",
        "mapped_households",
        "hourly_rate_min",
        "hourly_rate_mean",
        "hourly_rate_max",
        *RATE_HOUR_COLUMNS,
    ]
    st.markdown("**Gas rates used**")
    st.dataframe(
        gas_rows[gas_columns].style.format({
            "hourly_rate_min": "${:.6f}",
            "hourly_rate_mean": "${:.6f}",
            "hourly_rate_max": "${:.6f}",
            **{column: "${:.6f}" for column in RATE_HOUR_COLUMNS},
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.markdown(
    r"""
### Calculation formulas

For household \(i\), electricity tariff \(p\), and January or August:

\[
C^{E}_{i,p,m}
=
D_m
\sum_{h=0}^{23}
E^{HPWH}_{i,m,h}
R^{E}_{p,m,h}
\]

For the annual-average selection, the `year` usage profile is applied to each
of the 12 monthly tariff profiles, annual cost is calculated, and the result is
divided by 12.

For each household:

\[
C^{E,\min}_i=\min_p(C^E_{i,p}), \qquad
C^{E,\max}_i=\max_p(C^E_{i,p})
\]

\[
C^{E,\mathrm{avg}}_i
=
\frac{1}{P_i}\sum_p C^E_{i,p}
\]

The displayed electricity statistics are:

\[
\min_i C^{E,\min}_i,\qquad
\max_i C^{E,\max}_i,\qquad
\frac{1}{N}\sum_i C^{E,\mathrm{avg}}_i
\]

The cost-difference statistics follow the requested direction:

\[
\Delta_{\min}=E_{\min}-G_{\min},\quad
\Delta_{\max}=E_{\max}-G_{\max},\quad
\Delta_{\mathrm{avg}}=E_{\mathrm{avg}}-G_{\mathrm{avg}}
\]
"""
)
