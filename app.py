from __future__ import annotations

import matplotlib.pyplot as plt
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
    # Main line: average. Secondary line: full observed range.
    column.metric(
        title,
        money(summary["mean"]),
    )
    column.caption(
        f"Range: **{money(summary['min'])} – {money(summary['max'])}**  \n"
        f"Households: **{summary['n']}/{denominator}**"
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

st.title("HPWH vs Gas Water Heater — Household Monthly Cost (v3.3)")
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

# Stable display order and sequential household number.
_house_order = data.provider_map[["bldg_id"]].copy()
_house_order["_sort_id"] = pd.to_numeric(
    _house_order["bldg_id"],
    errors="coerce",
)
_house_order = (
    _house_order.sort_values(
        ["_sort_id", "bldg_id"],
        kind="stable",
    )
    .drop(columns="_sort_id")
    .reset_index(drop=True)
)
_house_order.insert(0, "House No.", range(1, len(_house_order) + 1))
HOUSE_NUMBER_MAP = dict(
    zip(_house_order["bldg_id"].astype(str), _house_order["House No."])
)


def add_house_number(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the same 1-149 display number to every household-level table."""
    output = frame.copy()
    output["bldg_id"] = output["bldg_id"].astype(str)
    output.insert(
        0,
        "House No.",
        output["bldg_id"].map(HOUSE_NUMBER_MAP),
    )
    return output.sort_values(
        ["House No.", "bldg_id"],
        kind="stable",
    ).reset_index(drop=True)

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

st.subheader("Average hourly utility rates")
st.caption(
    "Electricity: applicable tariffs are averaged within each household "
    "first, then averaged equally across the 149 households. "
    "Gas: each household's mapped gas-provider rate is averaged equally "
    "across the 149 households. Annual-average electricity rates are "
    "weighted by the actual number of days in each month."
)

hourly_rate_chart = result.hourly_average_rates.copy()
hourly_rate_chart["hour_label"] = hourly_rate_chart["hour"].map(
    lambda hour: f"{int(hour):02d}:00"
)

electric_rate_column, gas_rate_column = st.columns(2)

with electric_rate_column:
    st.markdown("**Average hourly electricity rate**")
    st.bar_chart(
        hourly_rate_chart.set_index("hour_label")[
            "average_electricity_rate"
        ],
        x_label="Hour",
        y_label="Rate ($/kWh)",
        use_container_width=True,
    )

with gas_rate_column:
    st.markdown("**Average hourly gas rate**")
    st.bar_chart(
        hourly_rate_chart.set_index("hour_label")[
            "average_gas_rate"
        ],
        x_label="Hour",
        y_label="Rate ($/kWh)",
        use_container_width=True,
    )

with st.expander("Show hourly average-rate table"):
    rate_table = hourly_rate_chart[[
        "hour_label",
        "average_electricity_rate",
        "average_gas_rate",
    ]].rename(columns={
        "hour_label": "Hour",
        "average_electricity_rate": "Average electricity rate ($/kWh)",
        "average_gas_rate": "Average gas rate ($/kWh)",
    })
    st.dataframe(
        rate_table.style.format({
            "Average electricity rate ($/kWh)": "${:.6f}",
            "Average gas rate ($/kWh)": "${:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
**Result definitions**

- HPWH minimum: the lowest value among every household's minimum tariff cost.
- HPWH maximum: the highest value among every household's maximum tariff cost.
- HPWH average: tariffs are averaged within each household first, and those
  household averages are then averaged.
- Gas minimum, maximum, and average: calculated from household gas costs.
- Cost difference: for each household, electric minimum/maximum/average is
  subtracted from that same household's gas cost; the displayed minimum,
  maximum, and average summarize those 1:1 household differences.
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
    f"Difference statistics use household-level 1:1 matches: "
    f"**{paired_n}/{total_houses} households**."
)

st.divider()

# ---------------------------------------------------------------------
# Household-level calculation tables
# ---------------------------------------------------------------------
st.subheader("Household-level calculation")

tab_1, tab_2, tab_3, tab_4 = st.tabs(
    [
        "All 149 households",
        "Electric tariff calculations",
        "Gas calculations",
        "Paired HPWH − Gas",
    ]
)

with tab_1:
    display = add_house_number(result.all_households)
    if not display.empty:
        for column_name in [
            "electric_min_tariff",
            "electric_max_tariff",
            "gas_tariff",
        ]:
            display[column_name] = display[column_name].map(
                lambda value: tariff_name(value)
                if pd.notna(value)
                else "N/A"
            )

        st.caption(
            f"Rows shown: **{len(display)}/{total_houses} households**. "
            "The table is built from the provider map first, so all mapped "
            "households remain visible even if a rate is missing."
        )
        st.dataframe(
            display.style.format({
                "electric_min": "${:,.2f}",
                "electric_max": "${:,.2f}",
                "electric_average": "${:,.2f}",
                "gas_monthly_cost": "${:,.2f}",
                "difference_min": "${:,.2f}",
                "difference_max": "${:,.2f}",
                "difference_average": "${:,.2f}",
            }, na_rep="N/A"),
            use_container_width=True,
            hide_index=True,
            height=650,
        )
    else:
        st.info("No mapped households are available.")

with tab_2:
    display = add_house_number(result.electric_tariff_costs)
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
    display = add_house_number(result.gas_households)
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
    display = add_house_number(result.paired_households)
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
        "Households with both electric and gas calculations",
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

st.divider()
st.subheader("House sample — County distribution (%)")
st.caption(
    f"County shares are calculated from all **{total_houses} sampled households**."
)

# A pie chart with all 79 counties would be unreadable. Show the ten largest
# county shares and group the remaining counties as Other; retain the complete
# county-percentage table below the chart.
pie_top_n = 10
pie_data = county_sample.head(pie_top_n).copy()
remaining_households = int(
    county_sample.iloc[pie_top_n:]["households"].sum()
)
if remaining_households > 0:
    pie_data = pd.concat(
        [
            pie_data,
            pd.DataFrame({
                "in.county_name": ["Other counties"],
                "households": [remaining_households],
                "share (%)": [
                    remaining_households / total_houses * 100
                ],
            }),
        ],
        ignore_index=True,
    )

chart_column, note_column = st.columns([1.45, 1])

with chart_column:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    wedges, _, autotexts = ax.pie(
        pie_data["households"],
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        startangle=90,
        counterclock=False,
    )
    ax.axis("equal")
    ax.set_title("Share of sampled households by county")
    ax.legend(
        wedges,
        pie_data["in.county_name"],
        title="County",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    for text in autotexts:
        text.set_fontsize(9)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with note_column:
    st.markdown(
        """
**Chart display**

- The ten counties with the largest samples are shown separately.
- All remaining counties are grouped as **Other counties** so the pie chart
  remains readable.
- The full county list and exact percentages are provided below.
"""
    )

county_table = county_sample.rename(columns={
    "in.county_name": "County",
    "households": "Households",
    "share (%)": "Share (%)",
}).reset_index(drop=True)
county_table.insert(0, "Rank", range(1, len(county_table) + 1))

st.dataframe(
    county_table.style.format({"Share (%)": "{:.2f}%"}),
    use_container_width=True,
    hide_index=True,
)

