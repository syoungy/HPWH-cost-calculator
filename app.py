from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from calculations import calculate_period
from data_loading import (
    PROFILE_LABELS,
    RATE_HOUR_COLUMNS,
    USAGE_HOUR_COLUMNS,
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
    """Format every displayed monetary result consistently to two decimals."""
    if pd.isna(value):
        return "N/A"

    numeric = float(value)
    if numeric < 0:
        return f"-${abs(numeric):,.2f}"
    return f"${numeric:,.2f}"


def tariff_name(code: object) -> str:
    text = str(code)
    label = TARIFF_LABELS.get(text, text)
    return f"{label} ({text})" if label != text else label


def show_summary_card(
    column,
    title: str,
    summary: dict[str, float | int],
    denominator: int,
    interval_95_mean: float,
    interval_95_n: int,
) -> None:
    column.metric(
        title,
        money(summary["mean"]),
    )
    column.caption(
        f"Interval 95%: **{money(interval_95_mean)}** "
        f"(**{interval_95_n} households**)  \n"
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

st.title("HPWH vs Gas Water Heater — Household Monthly Cost (v3.6)")
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



# ---------------------------------------------------------------------
# Central 95% interval calculation, based on energy use rather than cost.
# ---------------------------------------------------------------------
mapped_ids = set(data.provider_map["bldg_id"].astype(str))

selected_electric_usage = data.electricity_usage[
    data.electricity_usage["bldg_id"].astype(str).isin(mapped_ids)
    & (data.electricity_usage["season"].astype(str) == str(period))
][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
selected_electric_usage["bldg_id"] = (
    selected_electric_usage["bldg_id"].astype(str)
)
selected_electric_usage["daily_total_kwh"] = (
    selected_electric_usage[USAGE_HOUR_COLUMNS].sum(axis=1)
)

selected_gas_usage = data.gas_usage[
    data.gas_usage["bldg_id"].astype(str).isin(mapped_ids)
    & (data.gas_usage["season"].astype(str) == str(period))
][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
selected_gas_usage["bldg_id"] = selected_gas_usage["bldg_id"].astype(str)
selected_gas_usage["daily_total_kwh"] = (
    selected_gas_usage[USAGE_HOUR_COLUMNS].sum(axis=1)
)


def central_95_ids(
    usage_totals: pd.DataFrame,
) -> tuple[set[str], float, float]:
    """Keep households from the 2.5th through 97.5th usage percentiles."""
    clean = usage_totals[["bldg_id", "daily_total_kwh"]].dropna().copy()
    lower = float(clean["daily_total_kwh"].quantile(0.025))
    upper = float(clean["daily_total_kwh"].quantile(0.975))
    kept = clean[
        clean["daily_total_kwh"].between(lower, upper, inclusive="both")
    ]
    return set(kept["bldg_id"].astype(str)), lower, upper


electric_95_ids, electric_usage_p025, electric_usage_p975 = central_95_ids(
    selected_electric_usage
)
gas_95_ids, gas_usage_p025, gas_usage_p975 = central_95_ids(
    selected_gas_usage
)
paired_95_ids = electric_95_ids & gas_95_ids

electric_interval_rows = result.electric_households[
    result.electric_households["bldg_id"].astype(str).isin(electric_95_ids)
]
gas_interval_rows = result.gas_households[
    result.gas_households["bldg_id"].astype(str).isin(gas_95_ids)
]
difference_interval_rows = result.paired_households[
    result.paired_households["bldg_id"].astype(str).isin(paired_95_ids)
]

electric_interval_95_mean = float(
    electric_interval_rows["electric_average"].mean()
)
gas_interval_95_mean = float(
    gas_interval_rows["gas_monthly_cost"].mean()
)
difference_interval_95_mean = float(
    difference_interval_rows["difference_average"].mean()
)

st.subheader(f"Results — {profile_label}")
card_1, card_2, card_3 = st.columns(3)

show_summary_card(
    card_1,
    "HPWH monthly electricity cost",
    result.electric_summary,
    total_houses,
    electric_interval_95_mean,
    len(electric_interval_rows),
)
show_summary_card(
    card_2,
    "Gas-water-heater monthly cost",
    result.gas_summary,
    total_houses,
    gas_interval_95_mean,
    len(gas_interval_rows),
)
show_summary_card(
    card_3,
    "HPWH − Gas WH cost difference",
    result.difference_summary,
    total_houses,
    difference_interval_95_mean,
    len(difference_interval_rows),
)

st.caption(
    "Cost difference follows the requested direction: "
    "**HPWH electricity cost − Gas-WH cost**. "
    "A negative value means HPWH is cheaper; a positive value means HPWH "
    "is more expensive."
)

st.caption(
    "Interval 95% is a usage-trimmed mean: households below the 2.5th "
    "percentile and above the 97.5th percentile of daily energy use are "
    "removed before averaging cost. HPWH uses HPWH electricity consumption; "
    "Gas-WH uses gas consumption; the cost-difference card keeps households "
    "that pass both usage filters."
)

st.markdown(
    """
**Result definitions**

- HPWH minimum: the lowest value among every household's minimum tariff cost.
- HPWH maximum: the highest value among every household's maximum tariff cost.
- HPWH average: tariffs are averaged within each household first, and those
  household averages are then averaged.
- Gas minimum, maximum, and average: calculated from household gas costs.
- Interval 95%: the mean cost after removing the bottom 2.5% and top 2.5%
  of households based on daily energy use for the selected profile.
- Cost difference: for each household, electric minimum/maximum/average is
  subtracted from that same household's gas cost; the displayed minimum,
  maximum, and average summarize those 1:1 household differences.
"""
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

st.subheader("Average hourly energy use")
st.caption(
    f"Selected profile: **{profile_label}**. "
    "Each bar is the arithmetic mean of the mapped 149 households for the "
    "same hour of the day."
)

average_hpwh_usage = (
    data.electricity_usage[
        data.electricity_usage["bldg_id"].astype(str).isin(mapped_ids)
        & (data.electricity_usage["season"].astype(str) == str(period))
    ][USAGE_HOUR_COLUMNS]
    .mean(axis=0)
    .astype(float)
    .to_numpy()
)

average_gas_usage = (
    data.gas_usage[
        data.gas_usage["bldg_id"].astype(str).isin(mapped_ids)
        & (data.gas_usage["season"].astype(str) == str(period))
    ][USAGE_HOUR_COLUMNS]
    .mean(axis=0)
    .astype(float)
    .to_numpy()
)

hourly_usage_chart = pd.DataFrame({
    "Hour": [f"{hour:02d}:00" for hour in range(24)],
    "Average HPWH electricity use (kWh)": average_hpwh_usage,
    "Average gas-WH energy use (kWh)": average_gas_usage,
})

orange = "#F28C28"
electric_usage_column, gas_usage_column = st.columns(2)

with electric_usage_column:
    st.markdown("**Average hourly HPWH electricity use**")
    st.vega_lite_chart(
        hourly_usage_chart,
        {
            "mark": {
                "type": "bar",
                "color": orange,
                "cornerRadiusTopLeft": 3,
                "cornerRadiusTopRight": 3,
            },
            "encoding": {
                "x": {
                    "field": "Hour",
                    "type": "ordinal",
                    "sort": None,
                    "axis": {
                        "title": "Hour",
                        "labelAngle": -45,
                    },
                },
                "y": {
                    "field": "Average HPWH electricity use (kWh)",
                    "type": "quantitative",
                    "axis": {
                        "title": "Average use (kWh/hour)",
                    },
                },
                "tooltip": [
                    {
                        "field": "Hour",
                        "type": "ordinal",
                    },
                    {
                        "field": "Average HPWH electricity use (kWh)",
                        "type": "quantitative",
                        "title": "Average use (kWh)",
                        "format": ".6f",
                    },
                ],
            },
            "height": 330,
        },
        use_container_width=True,
    )

with gas_usage_column:
    st.markdown("**Average hourly gas-water-heater energy use**")
    st.vega_lite_chart(
        hourly_usage_chart,
        {
            "mark": {
                "type": "bar",
                "color": orange,
                "cornerRadiusTopLeft": 3,
                "cornerRadiusTopRight": 3,
            },
            "encoding": {
                "x": {
                    "field": "Hour",
                    "type": "ordinal",
                    "sort": None,
                    "axis": {
                        "title": "Hour",
                        "labelAngle": -45,
                    },
                },
                "y": {
                    "field": "Average gas-WH energy use (kWh)",
                    "type": "quantitative",
                    "axis": {
                        "title": "Average use (kWh/hour)",
                    },
                },
                "tooltip": [
                    {
                        "field": "Hour",
                        "type": "ordinal",
                    },
                    {
                        "field": "Average gas-WH energy use (kWh)",
                        "type": "quantitative",
                        "title": "Average use (kWh)",
                        "format": ".6f",
                    },
                ],
            },
            "height": 330,
        },
        use_container_width=True,
    )

with st.expander("Show hourly average-usage table"):
    st.dataframe(
        hourly_usage_chart.style.format({
            "Average HPWH electricity use (kWh)": "{:.6f}",
            "Average gas-WH energy use (kWh)": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
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

# Annual-average usage columns shown in the household table regardless of
# which cost profile is selected in the sidebar.
annual_electric_usage = data.electricity_usage[
    data.electricity_usage["bldg_id"].astype(str).isin(mapped_ids)
    & (data.electricity_usage["season"].astype(str) == "year")
][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
annual_electric_usage["bldg_id"] = (
    annual_electric_usage["bldg_id"].astype(str)
)
annual_electric_hour_columns = [
    f"HPWH annual avg {hour:02d}:00 (kWh)"
    for hour in range(24)
]
annual_electric_usage = annual_electric_usage.rename(
    columns=dict(zip(USAGE_HOUR_COLUMNS, annual_electric_hour_columns))
)
annual_electric_usage["HPWH annual avg daily total (kWh)"] = (
    annual_electric_usage[annual_electric_hour_columns].sum(axis=1)
)

annual_gas_usage = data.gas_usage[
    data.gas_usage["bldg_id"].astype(str).isin(mapped_ids)
    & (data.gas_usage["season"].astype(str) == "year")
][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
annual_gas_usage["bldg_id"] = annual_gas_usage["bldg_id"].astype(str)
annual_gas_hour_columns = [
    f"Gas-WH annual avg {hour:02d}:00 (kWh)"
    for hour in range(24)
]
annual_gas_usage = annual_gas_usage.rename(
    columns=dict(zip(USAGE_HOUR_COLUMNS, annual_gas_hour_columns))
)
annual_gas_usage["Gas-WH annual avg daily total (kWh)"] = (
    annual_gas_usage[annual_gas_hour_columns].sum(axis=1)
)

annual_usage_table = annual_electric_usage.merge(
    annual_gas_usage,
    on="bldg_id",
    how="inner",
    validate="one_to_one",
)

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
    display = (
        add_house_number(result.all_households)
        .merge(
            annual_usage_table,
            on="bldg_id",
            how="left",
            validate="one_to_one",
        )
    )

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

        hpwh_daily_total_column = "HPWH annual avg daily total (kWh)"
        gas_daily_total_column = "Gas-WH annual avg daily total (kWh)"

        existing_columns = [
            column
            for column in display.columns
            if column not in {
                "House No.",
                "bldg_id",
                hpwh_daily_total_column,
                gas_daily_total_column,
                *annual_electric_hour_columns,
                *annual_gas_hour_columns,
            }
        ]

        display = display[
            [
                "House No.",
                "bldg_id",
                hpwh_daily_total_column,
                gas_daily_total_column,
                *existing_columns,
                *annual_electric_hour_columns,
                *annual_gas_hour_columns,
            ]
        ]

        st.caption(
            f"Rows shown: **{len(display)}/{total_houses} households**. "
            "The two daily totals appear directly after bldg_id. "
            "The 00:00–23:00 annual-average usage columns appear at the "
            "far right and always use the `year` profile."
        )

        currency_columns = [
            "electric_min",
            "electric_max",
            "electric_average",
            "gas_monthly_cost",
            "difference_min",
            "difference_max",
            "difference_average",
        ]

        column_config = {
            column: st.column_config.NumberColumn(
                column,
                format="$%.2f",
            )
            for column in currency_columns
            if column in display.columns
        }
        column_config.update({
            hpwh_daily_total_column: st.column_config.NumberColumn(
                hpwh_daily_total_column,
                format="%.4f",
            ),
            gas_daily_total_column: st.column_config.NumberColumn(
                gas_daily_total_column,
                format="%.4f",
            ),
        })
        column_config.update({
            column: st.column_config.NumberColumn(
                column,
                format="%.6f",
            )
            for column in [
                *annual_electric_hour_columns,
                *annual_gas_hour_columns,
            ]
        })

        st.dataframe(
            display,
            column_config=column_config,
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

# County sample data are retained for the bottom county-distribution chart.
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
st.subheader("(1) Electricity and gas providers by county")

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
st.subheader("(2) Electricity and gas rates, including tariff type")

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
    pie_chart_data = pie_data.rename(columns={
        "in.county_name": "County",
        "households": "Households",
        "share (%)": "Share (%)",
    })

    st.vega_lite_chart(
        pie_chart_data,
        {
            "mark": {
                "type": "arc",
                "innerRadius": 45,
            },
            "encoding": {
                "theta": {
                    "field": "Households",
                    "type": "quantitative",
                    "stack": True,
                },
                "color": {
                    "field": "County",
                    "type": "nominal",
                    "legend": {
                        "title": "County",
                        "orient": "right",
                    },
                },
                "tooltip": [
                    {
                        "field": "County",
                        "type": "nominal",
                        "title": "County",
                    },
                    {
                        "field": "Households",
                        "type": "quantitative",
                        "title": "Households",
                        "format": ",d",
                    },
                    {
                        "field": "Share (%)",
                        "type": "quantitative",
                        "title": "Share (%)",
                        "format": ".2f",
                    },
                ],
            },
            "title": "Share of sampled households by county",
            "height": 460,
        },
        use_container_width=True,
    )

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

st.title("HPWH vs Gas Water Heater — Household Monthly Cost (v3.5.1)")
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


def household_option_label(row: pd.Series) -> str:
    return (
        f"House {int(row['House No.']):03d} — "
        f"bldg_id {row['bldg_id']} — {row['county']}"
    )


def build_household_usage_chart(
    chart_data: pd.DataFrame,
    bar_field: str,
    line_field: str,
    line_title: str,
    y_title: str,
) -> dict:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "color": "#F28C28",
                    "opacity": 0.75,
                    "cornerRadiusTopLeft": 3,
                    "cornerRadiusTopRight": 3,
                },
                "encoding": {
                    "x": {
                        "field": "Hour",
                        "type": "ordinal",
                        "sort": None,
                        "axis": {"title": "Hour", "labelAngle": -45},
                    },
                    "y": {
                        "field": bar_field,
                        "type": "quantitative",
                        "axis": {"title": y_title},
                    },
                    "tooltip": [
                        {"field": "Hour", "type": "ordinal"},
                        {
                            "field": bar_field,
                            "type": "quantitative",
                            "title": "149-house average",
                            "format": ".6f",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "line",
                    "color": "#1f77b4",
                    "point": True,
                    "strokeWidth": 3,
                },
                "encoding": {
                    "x": {
                        "field": "Hour",
                        "type": "ordinal",
                        "sort": None,
                        "axis": {"title": "Hour", "labelAngle": -45},
                    },
                    "y": {
                        "field": line_field,
                        "type": "quantitative",
                        "axis": {"title": y_title},
                    },
                    "tooltip": [
                        {"field": "Hour", "type": "ordinal"},
                        {
                            "field": line_field,
                            "type": "quantitative",
                            "title": line_title,
                            "format": ".6f",
                        },
                    ],
                },
            },
        ],
        "height": 340,
    }

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
- Cost difference: for each household, electric minimum/maximum/average is
  subtracted from that same household's gas cost; the displayed minimum,
  maximum, and average summarize those 1:1 household differences.
"""
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

st.subheader("Average hourly energy use")
st.caption(
    f"Selected profile: **{profile_label}**. "
    "Each bar is the arithmetic mean of the mapped 149 households for the "
    "same hour of the day."
)

mapped_ids = set(data.provider_map["bldg_id"].astype(str))

average_hpwh_usage = (
    data.electricity_usage[
        data.electricity_usage["bldg_id"].astype(str).isin(mapped_ids)
        & (data.electricity_usage["season"].astype(str) == str(period))
    ][USAGE_HOUR_COLUMNS]
    .mean(axis=0)
    .astype(float)
    .to_numpy()
)

average_gas_usage = (
    data.gas_usage[
        data.gas_usage["bldg_id"].astype(str).isin(mapped_ids)
        & (data.gas_usage["season"].astype(str) == str(period))
    ][USAGE_HOUR_COLUMNS]
    .mean(axis=0)
    .astype(float)
    .to_numpy()
)

hourly_usage_chart = pd.DataFrame({
    "Hour": [f"{hour:02d}:00" for hour in range(24)],
    "Average HPWH electricity use (kWh)": average_hpwh_usage,
    "Average gas-WH energy use (kWh)": average_gas_usage,
})

orange = "#F28C28"
electric_usage_column, gas_usage_column = st.columns(2)

with electric_usage_column:
    st.markdown("**Average hourly HPWH electricity use**")
    st.vega_lite_chart(
        hourly_usage_chart,
        {
            "mark": {
                "type": "bar",
                "color": orange,
                "cornerRadiusTopLeft": 3,
                "cornerRadiusTopRight": 3,
            },
            "encoding": {
                "x": {
                    "field": "Hour",
                    "type": "ordinal",
                    "sort": None,
                    "axis": {
                        "title": "Hour",
                        "labelAngle": -45,
                    },
                },
                "y": {
                    "field": "Average HPWH electricity use (kWh)",
                    "type": "quantitative",
                    "axis": {
                        "title": "Average use (kWh/hour)",
                    },
                },
                "tooltip": [
                    {
                        "field": "Hour",
                        "type": "ordinal",
                    },
                    {
                        "field": "Average HPWH electricity use (kWh)",
                        "type": "quantitative",
                        "title": "Average use (kWh)",
                        "format": ".6f",
                    },
                ],
            },
            "height": 330,
        },
        use_container_width=True,
    )

with gas_usage_column:
    st.markdown("**Average hourly gas-water-heater energy use**")
    st.vega_lite_chart(
        hourly_usage_chart,
        {
            "mark": {
                "type": "bar",
                "color": orange,
                "cornerRadiusTopLeft": 3,
                "cornerRadiusTopRight": 3,
            },
            "encoding": {
                "x": {
                    "field": "Hour",
                    "type": "ordinal",
                    "sort": None,
                    "axis": {
                        "title": "Hour",
                        "labelAngle": -45,
                    },
                },
                "y": {
                    "field": "Average gas-WH energy use (kWh)",
                    "type": "quantitative",
                    "axis": {
                        "title": "Average use (kWh/hour)",
                    },
                },
                "tooltip": [
                    {
                        "field": "Hour",
                        "type": "ordinal",
                    },
                    {
                        "field": "Average gas-WH energy use (kWh)",
                        "type": "quantitative",
                        "title": "Average use (kWh)",
                        "format": ".6f",
                    },
                ],
            },
            "height": 330,
        },
        use_container_width=True,
    )

with st.expander("Show hourly average-usage table"):
    st.dataframe(
        hourly_usage_chart.style.format({
            "Average HPWH electricity use (kWh)": "{:.6f}",
            "Average gas-WH energy use (kWh)": "{:.6f}",
        }),
        use_container_width=True,
        hide_index=True,
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
        st.caption(
            "Click any cell in a household row—including its bldg_id—to "
            "update the charts directly below."
        )
        household_table_event = st.dataframe(
            display,
            column_config={
                "electric_min": st.column_config.NumberColumn(
                    "electric_min", format="$%.2f"
                ),
                "electric_max": st.column_config.NumberColumn(
                    "electric_max", format="$%.2f"
                ),
                "electric_average": st.column_config.NumberColumn(
                    "electric_average", format="$%.2f"
                ),
                "gas_monthly_cost": st.column_config.NumberColumn(
                    "gas_monthly_cost", format="$%.2f"
                ),
                "difference_min": st.column_config.NumberColumn(
                    "difference_min", format="$%.2f"
                ),
                "difference_max": st.column_config.NumberColumn(
                    "difference_max", format="$%.2f"
                ),
                "difference_average": st.column_config.NumberColumn(
                    "difference_average", format="$%.2f"
                ),
            },
            use_container_width=True,
            hide_index=True,
            height=650,
            on_select="rerun",
            selection_mode="single-row",
            key="all_households_table",
        )
    else:
        st.info("No mapped households are available.")
        st.stop()

    selected_rows = household_table_event.selection.rows
    selected_row_index = int(selected_rows[0]) if selected_rows else 0
    selected_row = display.iloc[selected_row_index]
    selected_bldg_id = str(selected_row["bldg_id"])
    selected_house_no = int(selected_row["House No."])

    st.markdown("### Selected household hourly energy use")
    st.caption(
        "The selected household is the blue line. The orange bars show the "
        "149-house average for the same hour and selected period."
    )

    selected_electric_usage = (
        data.electricity_usage[
            (data.electricity_usage["bldg_id"].astype(str) == selected_bldg_id)
            & (data.electricity_usage["season"].astype(str) == str(period))
        ][USAGE_HOUR_COLUMNS]
        .iloc[0]
        .astype(float)
        .to_numpy()
    )
    selected_gas_usage = (
        data.gas_usage[
            (data.gas_usage["bldg_id"].astype(str) == selected_bldg_id)
            & (data.gas_usage["season"].astype(str) == str(period))
        ][USAGE_HOUR_COLUMNS]
        .iloc[0]
        .astype(float)
        .to_numpy()
    )

    selected_usage_chart = pd.DataFrame({
        "Hour": [f"{hour:02d}:00" for hour in range(24)],
        "Average HPWH electricity use (kWh)": average_hpwh_usage,
        "Selected HPWH electricity use (kWh)": selected_electric_usage,
        "Average gas-WH energy use (kWh)": average_gas_usage,
        "Selected gas-WH energy use (kWh)": selected_gas_usage,
    })

    st.markdown(
        f"**Selected household:** House **{selected_house_no}** · "
        f"bldg_id **{selected_bldg_id}**"
    )

    info_cols = st.columns(4)
    house_row = display[display["bldg_id"].astype(str) == selected_bldg_id].iloc[0]
    info_cols[0].metric("County", str(house_row["county"]))
    info_cols[1].metric("Electric provider", str(house_row["electric_provider"]))
    info_cols[2].metric("Gas provider", str(house_row["gas_provider"]))
    info_cols[3].metric("Selected profile", profile_label)

    sel_col_1, sel_col_2 = st.columns(2)
    with sel_col_1:
        st.markdown("**HPWH electricity use: selected household vs average**")
        st.vega_lite_chart(
            selected_usage_chart,
            build_household_usage_chart(
                selected_usage_chart,
                "Average HPWH electricity use (kWh)",
                "Selected HPWH electricity use (kWh)",
                "Selected household",
                "Electric use (kWh/hour)",
            ),
            use_container_width=True,
        )

    with sel_col_2:
        st.markdown("**Gas-water-heater use: selected household vs average**")
        st.vega_lite_chart(
            selected_usage_chart,
            build_household_usage_chart(
                selected_usage_chart,
                "Average gas-WH energy use (kWh)",
                "Selected gas-WH energy use (kWh)",
                "Selected household",
                "Gas use (kWh/hour)",
            ),
            use_container_width=True,
        )

    with st.expander("Show selected-household hourly usage table"):
        st.dataframe(
            selected_usage_chart.style.format({
                "Average HPWH electricity use (kWh)": "{:.6f}",
                "Selected HPWH electricity use (kWh)": "{:.6f}",
                "Average gas-WH energy use (kWh)": "{:.6f}",
                "Selected gas-WH energy use (kWh)": "{:.6f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

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

# County sample data are retained for the bottom county-distribution chart.
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
st.subheader("(1) Electricity and gas providers by county")

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
st.subheader("(2) Electricity and gas rates, including tariff type")

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
    pie_chart_data = pie_data.rename(columns={
        "in.county_name": "County",
        "households": "Households",
        "share (%)": "Share (%)",
    })

    st.vega_lite_chart(
        pie_chart_data,
        {
            "mark": {
                "type": "arc",
                "innerRadius": 45,
            },
            "encoding": {
                "theta": {
                    "field": "Households",
                    "type": "quantitative",
                    "stack": True,
                },
                "color": {
                    "field": "County",
                    "type": "nominal",
                    "legend": {
                        "title": "County",
                        "orient": "right",
                    },
                },
                "tooltip": [
                    {
                        "field": "County",
                        "type": "nominal",
                        "title": "County",
                    },
                    {
                        "field": "Households",
                        "type": "quantitative",
                        "title": "Households",
                        "format": ",d",
                    },
                    {
                        "field": "Share (%)",
                        "type": "quantitative",
                        "title": "Share (%)",
                        "format": ".2f",
                    },
                ],
            },
            "title": "Share of sampled households by county",
            "height": 460,
        },
        use_container_width=True,
    )

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
