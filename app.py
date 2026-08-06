from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from calculations import (
    DEFAULT_PROPANE_PRICE_PER_GALLON,
    GAS_WH_EFFICIENCY,
    PROPANE_KWH_PER_GALLON,
    PROPANE_WH_EFFICIENCY,
    RESISTANCE_WH_EFFICIENCY,
    StatewideResult,
    available_electric_tariffs,
    available_gas_tariffs,
    calculate_county_scenario,
    calculate_statewide,
)
from data_loading import (
    ELECTRICITY_RATE_FILE,
    ELECTRICITY_USAGE_FILE,
    GAS_RATE_FILE,
    GAS_USAGE_FILE,
    GT_ELECTRICITY_USAGE_FILE,
    GT_GAS_USAGE_FILE,
    PROFILE_LABELS,
    PROVIDER_FILE,
    TARIFF_LABELS,
    USAGE_HOUR_COLUMNS,
    load_calculator_data,
)


ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    ELECTRICITY_USAGE_FILE,
    GAS_USAGE_FILE,
    GT_ELECTRICITY_USAGE_FILE,
    GT_GAS_USAGE_FILE,
    PROVIDER_FILE,
    ELECTRICITY_RATE_FILE,
    GAS_RATE_FILE,
]
PERIOD_OPTIONS = {
    "1": "January",
    "8": "August",
    "year": "Annual average",
}
REGION_CONFIGS = {
    "Wayne County": {
        "counties": ("Wayne County",),
    },
    "Kent County": {
        "counties": ("Kent County",),
    },
    "Washtenaw County": {
        "counties": ("Washtenaw County",),
    },
    "Holland": {
        "counties": ("Ottawa County", "Allegan County"),
        "electric_provider": "Holland board of public works",
        "gas_provider": "SEMCO Energy Gas Company",
    },
    "Traverse City": {
        "counties": ("Grand Traverse County",),
        "electric_provider": "Traverse city light & power",
        "gas_provider": "DTE Gas Company",
        "usage_sample": "grand_traverse_dedicated",
    },
}
REGION_OPTIONS = list(REGION_CONFIGS)


def tariff_name(value: object) -> str:
    text = str(value)
    if text == "-":
        return "Standard Rate"
    label = TARIFF_LABELS.get(text, text)
    return f"{label} ({text})" if label != text else text


def money(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    numeric = float(value)
    return f"-${abs(numeric):,.2f}" if numeric < 0 else f"${numeric:,.2f}"


def _data_file_path(filename: str) -> Path:
    root_path = ROOT / filename
    data_path = ROOT / "data" / filename
    if root_path.exists():
        return root_path
    return data_path


def data_signature() -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for filename in REQUIRED_FILES:
        path = _data_file_path(filename)
        if not path.exists():
            signature.append((filename, -1, -1))
        else:
            stat = path.stat()
            signature.append((filename, int(stat.st_mtime_ns), int(stat.st_size)))
    return tuple(signature)


@st.cache_data(show_spinner="Loading and validating input data...")
def load_all_data(signature: tuple[tuple[str, int, int], ...]):
    _ = signature
    return load_calculator_data()


@st.cache_data(show_spinner=False)
def calculate_statewide_cached(
    signature: tuple[tuple[str, int, int], ...],
    period: str,
    propane_price: float,
) -> StatewideResult:
    data = load_all_data(signature)
    return calculate_statewide(data, period, propane_price)


@st.cache_data(show_spinner=False)
def calculate_county_cached(
    signature: tuple[tuple[str, int, int], ...],
    geography_label: str,
    sample_counties: tuple[str, ...],
    electric_provider: str,
    electric_tariff: str,
    gas_provider: str,
    gas_tariff: str,
    period: str,
    propane_price: float,
    usage_sample: str,
):
    data = load_all_data(signature)
    return calculate_county_scenario(
        data=data,
        county=sample_counties,
        geography_label=geography_label,
        electric_provider=electric_provider,
        electric_tariff=electric_tariff,
        gas_provider=gas_provider,
        gas_tariff=gas_tariff,
        period=period,
        propane_price_per_gallon=propane_price,
        usage_sample=usage_sample,
    )


def show_summary_card(
    column,
    title: str,
    summary: dict[str, float | int],
    interval: dict[str, float | int] | None = None,
) -> None:
    column.metric(title, money(float(summary["mean"])))

    # Markdown interprets HTML lines indented by four or more spaces as a
    # code block. Build the markup without any leading indentation.
    interval_html = ""
    if interval is not None:
        interval_html = (
            '<div style="font-size:1.14rem;font-weight:750;'
            'margin-bottom:0.32rem;color:inherit;">'
            f'Interval 95%: {money(float(interval["mean"]))} '
            '<span style="font-size:0.90rem;font-weight:600;color:inherit;">'
            f'({int(interval["n"])} households)'
            '</span>'
            '</div>'
        )

    card_html = (
        '<div style="margin-top:0.32rem;line-height:1.45;color:inherit;">'
        f'{interval_html}'
        '<div style="font-size:0.96rem;font-weight:500;'
        'margin-bottom:0.16rem;color:inherit;">'
        f'Range: {money(float(summary["min"]))} – '
        f'{money(float(summary["max"]))}'
        '</div>'
        '<div style="font-size:0.96rem;font-weight:500;color:inherit;">'
        f'Households: {int(summary["n"])}'
        '</div>'
        '</div>'
    )
    column.markdown(card_html, unsafe_allow_html=True)


def add_house_number(frame: pd.DataFrame, data) -> pd.DataFrame:
    order = data.provider_map[["bldg_id"]].copy()
    order["bldg_id"] = order["bldg_id"].astype(str)
    order["_sort"] = pd.to_numeric(order["bldg_id"], errors="coerce")
    order = order.sort_values(["_sort", "bldg_id"], kind="stable").reset_index(drop=True)
    order["House No."] = range(1, len(order) + 1)
    mapping = dict(zip(order["bldg_id"], order["House No."]))

    output = frame.copy()
    output["bldg_id"] = output["bldg_id"].astype(str)
    output.insert(0, "House No.", output["bldg_id"].map(mapping))
    return output.sort_values(["House No.", "bldg_id"], kind="stable").reset_index(drop=True)


def annual_hourly_table(data, household_summary: pd.DataFrame) -> pd.DataFrame:
    electric = data.electricity_usage[
        data.electricity_usage["season"].astype(str) == "year"
    ][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
    gas = data.gas_usage[
        data.gas_usage["season"].astype(str) == "year"
    ][["bldg_id", *USAGE_HOUR_COLUMNS]].copy()
    electric["bldg_id"] = electric["bldg_id"].astype(str)
    gas["bldg_id"] = gas["bldg_id"].astype(str)

    electric_columns = [f"HPWH {hour:02d}:00 (kWh)" for hour in range(24)]
    gas_columns = [f"Gas-WH {hour:02d}:00 (kWh)" for hour in range(24)]
    electric = electric.rename(columns=dict(zip(USAGE_HOUR_COLUMNS, electric_columns)))
    gas = gas.rename(columns=dict(zip(USAGE_HOUR_COLUMNS, gas_columns)))
    electric["HPWH annual avg daily total (kWh)"] = electric[electric_columns].sum(axis=1)
    gas["Gas-WH annual avg daily total (kWh)"] = gas[gas_columns].sum(axis=1)

    base = household_summary[["bldg_id", "county"]].copy()
    return (
        base.merge(electric, on="bldg_id", how="left", validate="one_to_one")
        .merge(gas, on="bldg_id", how="left", validate="one_to_one")
    )


def render_statewide_details(result: StatewideResult, data) -> None:
    st.subheader("Detailed results")
    table_choice = st.selectbox(
        "Detail table",
        [
            "Household cost summary",
            "HPWH tariff calculations",
            "Electric resistance tariff calculations",
            "Gas-water-heater calculations",
            "Paired technology comparisons",
            "Annual hourly energy use",
        ],
        key="statewide_detail_table",
    )

    if table_choice == "Household cost summary":
        display = add_house_number(result.household_summary, data)
    elif table_choice == "HPWH tariff calculations":
        display = add_house_number(result.hpwh_tariff_costs, data)
    elif table_choice == "Electric resistance tariff calculations":
        display = add_house_number(result.resistance_tariff_costs, data)
    elif table_choice == "Gas-water-heater calculations":
        display = add_house_number(result.gas_households, data)
    elif table_choice == "Paired technology comparisons":
        display = add_house_number(result.paired_households, data)
    else:
        display = add_house_number(annual_hourly_table(data, result.household_summary), data)

    st.dataframe(display, use_container_width=True, hide_index=True, height=620)
    st.download_button(
        "Download displayed table as CSV",
        display.to_csv(index=False).encode("utf-8"),
        file_name=f"{table_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_statewide_charts(result: StatewideResult) -> None:
    st.subheader("Average hourly profiles")
    usage = result.hourly_average_usage.set_index("hour")
    rates = result.hourly_average_rates.set_index("hour")

    left, right = st.columns(2)
    with left:
        st.caption("Average energy use across the 149-household sample")
        st.line_chart(usage, height=330)
    with right:
        st.caption("Equal-household average hourly variable rates")
        st.line_chart(rates, height=330)


def provider_for_counties(
    data,
    counties: tuple[str, ...],
    column: str,
) -> str:
    values = sorted(
        data.provider_map.loc[
            data.provider_map["in.county_name"].astype(str).isin(counties),
            column,
        ].dropna().astype(str).unique().tolist()
    )
    if not values:
        raise ValueError(
            f"No {column} mapping is available for {', '.join(counties)}."
        )
    return values[0]


def region_settings(
    data,
    region_label: str,
) -> tuple[tuple[str, ...], str, str, str]:
    config = REGION_CONFIGS[region_label]
    counties = tuple(str(value) for value in config["counties"])
    electric_provider = str(
        config.get("electric_provider")
        or provider_for_counties(data, counties, "elec_provd")
    )
    gas_provider = str(
        config.get("gas_provider")
        or provider_for_counties(data, counties, "gas_provd")
    )
    usage_sample = str(config.get("usage_sample", "mapped"))
    return counties, electric_provider, gas_provider, usage_sample


def region_sample_count(data, usage_sample: str, counties: tuple[str, ...]) -> int:
    if usage_sample == "grand_traverse_dedicated":
        return int(data.gt_electricity_usage["bldg_id"].nunique())
    return int(
        data.provider_map["in.county_name"]
        .astype(str)
        .isin(counties)
        .sum()
    )


st.set_page_config(
    page_title="Water-heater energy cost calculator",
    page_icon="💧",
    layout="wide",
)

st.title("Residential Water-Heater Energy Cost Calculator — v4.3")
st.caption(
    "HPWH, natural gas, propane, and electric resistance scenarios. "
    "All costs are variable energy charges only; fixed customer charges, taxes, "
    "installation, and maintenance are not included."
)

signature = data_signature()
try:
    data = load_all_data(signature)
except Exception as exc:
    st.error(f"Failed to load or validate data: {exc}")
    st.stop()


# Applied settings are separate from the live widget values. Changing a widget
# reruns only the light UI; the scenario changes after Calculate / Update.
st.session_state.setdefault(
    "applied_statewide",
    {"period": "1", "propane_price": DEFAULT_PROPANE_PRICE_PER_GALLON},
)

with st.sidebar:
    st.header("Input")
    analysis_mode = st.radio(
        "Analysis mode",
        ["All Michigan sample", "County / area scenario"],
        key="analysis_mode",
    )

    if analysis_mode == "All Michigan sample":
        selected_period = st.selectbox(
            "Consumption profile",
            list(PERIOD_OPTIONS),
            format_func=lambda code: PERIOD_OPTIONS[code],
            key="statewide_period_input",
        )
        selected_propane_price = st.number_input(
            "Propane price ($/gallon)",
            min_value=0.0,
            value=float(DEFAULT_PROPANE_PRICE_PER_GALLON),
            step=0.01,
            format="%.3f",
            key="statewide_propane_price_input",
        )
        if st.button("Calculate / Update", type="primary", use_container_width=True):
            st.session_state["applied_statewide"] = {
                "period": selected_period,
                "propane_price": float(selected_propane_price),
            }

        applied = st.session_state["applied_statewide"]
        st.caption(
            f"Applied: {PERIOD_OPTIONS[applied['period']]} · "
            f"Propane ${applied['propane_price']:.3f}/gal"
        )

    else:
        selected_region = st.selectbox(
            "County / area",
            REGION_OPTIONS,
            key="region_input",
        )
        selected_region_period = st.selectbox(
            "Period",
            list(PERIOD_OPTIONS),
            format_func=lambda code: PERIOD_OPTIONS[code],
            key="region_period_input",
        )

        try:
            (
                sample_counties,
                electric_provider,
                gas_provider,
                usage_sample,
            ) = region_settings(data, selected_region)
            electric_tariffs = available_electric_tariffs(
                data, electric_provider, selected_region_period
            )
            gas_tariffs = available_gas_tariffs(data, gas_provider)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        if not electric_tariffs or not gas_tariffs:
            st.error("No complete tariff is available for the selected scenario.")
            st.stop()

        current_electric = st.session_state.get("region_electric_tariff_input")
        if current_electric not in electric_tariffs:
            st.session_state["region_electric_tariff_input"] = electric_tariffs[0]
        current_gas = st.session_state.get("region_gas_tariff_input")
        if current_gas not in gas_tariffs:
            st.session_state["region_gas_tariff_input"] = gas_tariffs[0]

        selected_electric_tariff = st.selectbox(
            f"Electricity tariff — {electric_provider}",
            electric_tariffs,
            format_func=tariff_name,
            key="region_electric_tariff_input",
        )
        selected_gas_tariff = st.selectbox(
            f"Gas tariff — {gas_provider}",
            gas_tariffs,
            format_func=tariff_name,
            key="region_gas_tariff_input",
        )
        selected_region_propane_price = st.number_input(
            "Propane price ($/gallon)",
            min_value=0.0,
            value=float(DEFAULT_PROPANE_PRICE_PER_GALLON),
            step=0.01,
            format="%.3f",
            key="region_propane_price_input",
        )

        sample_count = region_sample_count(
            data, usage_sample, sample_counties
        )
        st.caption(
            f"Sample counties: {', '.join(sample_counties)} · "
            f"Sample households: {sample_count}"
        )

        default_region_applied = {
            "region": selected_region,
            "sample_counties": sample_counties,
            "period": selected_region_period,
            "electric_provider": electric_provider,
            "electric_tariff": electric_tariffs[0],
            "gas_provider": gas_provider,
            "gas_tariff": gas_tariffs[0],
            "usage_sample": usage_sample,
            "propane_price": float(DEFAULT_PROPANE_PRICE_PER_GALLON),
        }
        st.session_state.setdefault("applied_region", default_region_applied)

        if st.button("Calculate / Update", type="primary", use_container_width=True):
            st.session_state["applied_region"] = {
                "region": selected_region,
                "sample_counties": sample_counties,
                "period": selected_region_period,
                "electric_provider": electric_provider,
                "electric_tariff": selected_electric_tariff,
                "gas_provider": gas_provider,
                "gas_tariff": selected_gas_tariff,
                "usage_sample": usage_sample,
                "propane_price": float(selected_region_propane_price),
            }

        applied = st.session_state["applied_region"]
        st.caption(
            f"Applied: {applied['region']} · "
            f"{PERIOD_OPTIONS[applied['period']]}"
        )


if analysis_mode == "All Michigan sample":
    applied = st.session_state["applied_statewide"]
    with st.spinner("Calculating statewide results..."):
        result = calculate_statewide_cached(
            signature,
            applied["period"],
            float(applied["propane_price"]),
        )

    st.subheader(f"Results — {PROFILE_LABELS[applied['period']]}")
    st.caption(
        f"149 mapped households · Propane price: "
        f"${float(applied['propane_price']):.3f}/gallon"
    )

    first_1, first_2, first_3 = st.columns(3)
    show_summary_card(first_1, "HPWH monthly electricity cost", result.hpwh_summary, result.interval_95["hpwh"])
    show_summary_card(first_2, "Natural-gas WH monthly cost", result.gas_summary, result.interval_95["gas"])
    show_summary_card(first_3, "HPWH − Gas WH cost difference", result.hpwh_minus_gas_summary, result.interval_95["hpwh_minus_gas"])

    st.markdown("### Additional water-heater scenarios")
    second_1, second_2 = st.columns(2)
    show_summary_card(second_1, "Propane WH monthly cost", result.propane_summary, result.interval_95["propane"])
    show_summary_card(second_2, "Electric resistance WH monthly cost", result.resistance_summary, result.interval_95["resistance"])

    st.caption(
        "Gas, propane, and electric-resistance Central 95% means use the same "
        "gas-consumption-based household set. HPWH uses its HPWH-consumption set."
    )

    with st.expander("Calculation assumptions and formulas", expanded=False):
        st.markdown(
            f"""
- Baseline natural-gas WH efficiency: `{GAS_WH_EFFICIENCY:.2f}`
- Propane WH efficiency: `{PROPANE_WH_EFFICIENCY:.2f}`
- Electric resistance WH efficiency: `{RESISTANCE_WH_EFFICIENCY:.2f}`
- Propane energy content: `{PROPANE_KWH_PER_GALLON:.3f} kWh/gallon`
- Propane input = natural-gas input × `{GAS_WH_EFFICIENCY:.2f}/{PROPANE_WH_EFFICIENCY:.2f}`
- Electric resistance use = natural-gas input × `{GAS_WH_EFFICIENCY:.2f}/{RESISTANCE_WH_EFFICIENCY:.2f}`
- Electric resistance uses the same regional electricity tariffs as HPWH.
- January/August use the selected average daily profile multiplied by calendar days.
- Annual average applies the annual-average daily profile to all 12 monthly tariffs, totals the year, and divides by 12.
            """
        )

    show_charts = st.checkbox("Show hourly charts", value=False)
    if show_charts:
        render_statewide_charts(result)

    show_details = st.checkbox("Show detailed household tables", value=False)
    if show_details:
        render_statewide_details(result, data)

else:
    applied = st.session_state["applied_region"]
    with st.spinner("Calculating county / area scenario..."):
        result = calculate_county_cached(
            signature,
            applied["region"],
            tuple(applied["sample_counties"]),
            applied["electric_provider"],
            applied["electric_tariff"],
            applied["gas_provider"],
            applied["gas_tariff"],
            applied["period"],
            float(applied["propane_price"]),
            applied.get(
                "usage_sample",
                REGION_CONFIGS.get(applied["region"], {}).get(
                    "usage_sample", "mapped"
                ),
            ),
        )

    st.subheader(
        f"Results — {applied['region']} · "
        f"{PROFILE_LABELS[applied['period']]}"
    )
    st.caption(
        f"Sample counties: {', '.join(result.sample_counties)} · "
        f"Households: {len(result.households)} · "
        f"Electricity: {result.electric_provider} / "
        f"{tariff_name(result.electric_tariff)} · "
        f"Gas: {result.gas_provider} / {tariff_name(result.gas_tariff)} · "
        f"Propane: ${float(applied['propane_price']):.3f}/gallon"
    )
    if result.usage_sample == "grand_traverse_dedicated":
        st.caption(
            f"Usage source: dedicated Grand Traverse sample — "
            f"{len(result.households)} paired HPWH and natural-gas households."
        )

    first_1, first_2, first_3 = st.columns(3)
    show_summary_card(
        first_1,
        "HPWH monthly electricity cost",
        result.hpwh_summary,
    )
    show_summary_card(
        first_2,
        "Natural-gas WH monthly cost",
        result.gas_summary,
    )
    show_summary_card(
        first_3,
        "HPWH − Gas WH cost difference",
        result.hpwh_minus_gas_summary,
    )

    st.markdown("### Additional water-heater scenarios")
    second_1, second_2 = st.columns(2)
    show_summary_card(
        second_1,
        "Propane WH monthly cost",
        result.propane_summary,
    )
    show_summary_card(
        second_2,
        "Electric resistance WH monthly cost",
        result.resistance_summary,
    )

    st.caption(
        "The selected electricity and gas tariffs are applied to every sampled "
        "household in the selected county or multi-county area. Holland pools "
        "Ottawa and Allegan County households and applies Holland Board of Public "
        "Works and SEMCO tariffs. Traverse City uses the separate 75-household "
        "Grand Traverse sample and applies Traverse City Light & Power and DTE "
        "Gas tariffs."
    )

    if len(result.households) < 10:
        st.warning(
            f"Small pilot sample: this scenario contains only "
            f"{len(result.households)} households and should not be interpreted "
            "as representative of the full service territory."
        )

    if st.checkbox("Show regional household table", value=False):
        display = result.households.sort_values(
            ["county", "bldg_id"],
            key=lambda series: (
                pd.to_numeric(series, errors="coerce")
                if series.name == "bldg_id"
                else series.astype(str)
            ),
            kind="stable",
        ).reset_index(drop=True)
        display.insert(0, "Area House No.", range(1, len(display) + 1))
        st.dataframe(display, use_container_width=True, hide_index=True, height=620)
        safe_name = (
            applied["region"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("+", "and")
        )
        st.download_button(
            "Download regional results as CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_name}_scenario.csv",
            mime="text/csv",
        )
