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
    SpaceHeatingResult,
    StatewideResult,
    available_electric_tariffs,
    available_gas_tariffs,
    calculate_county_scenario,
    calculate_space_heating_scenario,
    calculate_statewide,
)
from data_loading import (
    ELECTRICITY_RATE_FILE,
    ELECTRICITY_USAGE_FILE,
    GAS_RATE_FILE,
    GAS_USAGE_FILE,
    GT_ELECTRICITY_USAGE_FILE,
    GT_GAS_USAGE_FILE,
    GT_SPACE_ELECTRICITY_USAGE_FILE,
    GT_SPACE_GAS_USAGE_FILE,
    PROFILE_LABELS,
    PROVIDER_FILE,
    RATE_HOUR_COLUMNS,
    TARIFF_LABELS,
    USAGE_HOUR_COLUMNS,
    load_county_data,
    load_space_heating_data,
    load_statewide_data,
)


ROOT = Path(__file__).resolve().parent
STATEWIDE_REQUIRED_FILES = [
    ELECTRICITY_USAGE_FILE,
    GAS_USAGE_FILE,
    PROVIDER_FILE,
    ELECTRICITY_RATE_FILE,
    GAS_RATE_FILE,
]
COUNTY_REQUIRED_FILES = [
    ELECTRICITY_USAGE_FILE,
    GAS_USAGE_FILE,
    GT_ELECTRICITY_USAGE_FILE,
    GT_GAS_USAGE_FILE,
    PROVIDER_FILE,
    ELECTRICITY_RATE_FILE,
    GAS_RATE_FILE,
]
SPACE_HEATING_REQUIRED_FILES = [
    GT_SPACE_ELECTRICITY_USAGE_FILE,
    GT_SPACE_GAS_USAGE_FILE,
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

SPACE_HEATING_REGION_CONFIGS = {
    "Traverse City": {
        "counties": ("Grand Traverse County",),
        "electric_provider": "Traverse city light & power",
        "gas_provider": "DTE Gas Company",
    },
}
SPACE_HEATING_REGION_OPTIONS = list(SPACE_HEATING_REGION_CONFIGS)


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



MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


def _hour_display_label(hour: int) -> str:
    normalized = int(hour) % 24
    suffix = "AM" if normalized < 12 else "PM"
    clock_hour = normalized % 12 or 12
    return f"{clock_hour} {suffix}"


def traverse_city_hourly_rate_table(
    data,
    tariff: str,
) -> pd.DataFrame:
    provider = "Traverse city light & power"

    rates = data.electricity_rates.loc[
        (data.electricity_rates["elec_provd"].astype(str) == provider)
        & (data.electricity_rates["tariff"].astype(str) == str(tariff)),
        ["month", *RATE_HOUR_COLUMNS],
    ].copy()

    if rates.empty:
        raise ValueError(
            f"No electricity-rate rows were found for {provider} / {tariff}."
        )

    rates["month"] = pd.to_numeric(
        rates["month"],
        errors="raise",
    ).astype(int)

    # Identical duplicate rows are harmless, but conflicting rate rows for
    # the same month must not be silently displayed or used.
    rates = rates.drop_duplicates()
    duplicate_months = rates.loc[
        rates.duplicated(subset=["month"], keep=False),
        "month",
    ].sort_values()

    if not duplicate_months.empty:
        months = ", ".join(
            str(month)
            for month in duplicate_months.unique()
        )
        raise ValueError(
            f"{provider} / {tariff} contains conflicting rate profiles "
            f"for month(s): {months}."
        )

    expected_months = list(range(1, 13))
    actual_months = sorted(rates["month"].tolist())
    if actual_months != expected_months:
        raise ValueError(
            f"{provider} / {tariff} must contain exactly one row for "
            "each month from January through December."
        )

    rates = rates.sort_values("month").reset_index(drop=True)

    table = pd.DataFrame(
        {
            "Month": rates["month"].map(MONTH_NAMES),
        }
    )

    for hour, rate_column in enumerate(RATE_HOUR_COLUMNS):
        table[_hour_display_label(hour)] = rates[rate_column].map(
            lambda value: f"${float(value):.5f}"
        )

    return table


def show_traverse_city_rate_tables(
    data,
    selected_tariff: str,
) -> None:
    st.markdown(
        "#### Traverse City Light & Power — monthly hourly rate tables"
    )

    for tariff in ["Phase In", "Phase Out"]:
        selected_label = (
            " — Selected for this calculation"
            if tariff == str(selected_tariff)
            else ""
        )
        st.markdown(f"**{tariff}{selected_label}**")
        st.dataframe(
            traverse_city_hourly_rate_table(data, tariff),
            use_container_width=True,
            hide_index=True,
            height=455,
        )

    st.caption(
        "Each table shows the variable electricity rate for all 24 hours "
        "of every month. Rates are displayed in $/kWh and are read directly "
        "from electricity_rates_weekdays_202607.xlsx."
    )



def _data_file_path(filename: str) -> Path:
    data_path = ROOT / "data" / filename
    root_path = ROOT / filename
    if data_path.exists():
        return data_path
    return root_path


def data_signature(
    filenames: list[str] | tuple[str, ...],
) -> tuple[tuple[str, int, int], ...]:
    """Create a cache key only from files used by the active mode."""
    signature: list[tuple[str, int, int]] = []
    for filename in filenames:
        path = _data_file_path(filename)
        if not path.exists():
            signature.append((filename, -1, -1))
        else:
            stat = path.stat()
            signature.append(
                (filename, int(stat.st_mtime_ns), int(stat.st_size))
            )
    return tuple(signature)


@st.cache_resource(show_spinner="Loading statewide water-heating data...")
def load_statewide_data_cached(
    signature: tuple[tuple[str, int, int], ...],
):
    _ = signature
    return load_statewide_data()


@st.cache_resource(show_spinner="Loading county / area water-heating data...")
def load_county_data_cached(
    signature: tuple[tuple[str, int, int], ...],
):
    _ = signature
    return load_county_data()


@st.cache_resource(show_spinner="Loading Traverse City space-heating data...")
def load_space_heating_data_cached(
    signature: tuple[tuple[str, int, int], ...],
):
    _ = signature
    return load_space_heating_data()


@st.cache_data(show_spinner=False)
def calculate_statewide_cached(
    signature: tuple[tuple[str, int, int], ...],
    period: str,
    propane_price: float,
) -> StatewideResult:
    data = load_statewide_data_cached(signature)
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
    data = load_county_data_cached(signature)
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


@st.cache_data(show_spinner=False)
def calculate_space_heating_cached(
    signature: tuple[tuple[str, int, int], ...],
    geography_label: str,
    sample_counties: tuple[str, ...],
    electric_provider: str,
    electric_tariff: str,
    gas_provider: str,
    gas_tariff: str,
    period: str,
    propane_price: float,
) -> SpaceHeatingResult:
    data = load_space_heating_data_cached(signature)
    return calculate_space_heating_scenario(
        data=data,
        county=sample_counties,
        geography_label=geography_label,
        electric_provider=electric_provider,
        electric_tariff=electric_tariff,
        gas_provider=gas_provider,
        gas_tariff=gas_tariff,
        period=period,
        propane_price_per_gallon=propane_price,
    )

def show_summary_card(
    column,
    title: str,
    summary: dict[str, float | int],
    interval: dict[str, float | int] | None = None,
    details: list[tuple[str, float]] | None = None,
) -> None:
    column.metric(title, money(float(summary["mean"])))

    # Optional average-cost breakdown shown directly under the total.
    # These component means add to the displayed overall mean before rounding.
    details_html = ""
    if details:
        detail_rows = "".join(
            '<div style="font-size:0.84rem;font-weight:450;'
            'line-height:1.35;color:inherit;opacity:0.82;">'
            f'{label}: ${float(value):,.3f}'
            '</div>'
            for label, value in details
        )
        detail_total = sum(float(value) for _, value in details)
        details_html = (
            '<div style="margin-top:0.08rem;margin-bottom:0.48rem;">'
            '<div style="font-size:0.78rem;font-weight:600;'
            'line-height:1.25;color:inherit;opacity:0.65;">'
            'Average cost breakdown'
            '</div>'
            f'{detail_rows}'
            '<div style="font-size:0.80rem;font-weight:650;'
            'line-height:1.35;margin-top:0.08rem;color:inherit;opacity:0.78;">'
            f'= Total: {money(detail_total)}'
            '</div>'
            '</div>'
        )

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
        '<div style="margin-top:0.20rem;line-height:1.45;color:inherit;">'
        f'{details_html}'
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
    page_title="Residential energy cost calculator",
    page_icon="🏠",
    layout="wide",
)

title_placeholder = st.empty()
caption_placeholder = st.empty()

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
        [
            "All Michigan sample",
            "County / area scenario",
            "Space heating",
        ],
        key="analysis_mode",
    )

    if analysis_mode == "All Michigan sample":
        signature = data_signature(STATEWIDE_REQUIRED_FILES)
        loader = load_statewide_data_cached
    elif analysis_mode == "County / area scenario":
        signature = data_signature(COUNTY_REQUIRED_FILES)
        loader = load_county_data_cached
    else:
        signature = data_signature(SPACE_HEATING_REQUIRED_FILES)
        loader = load_space_heating_data_cached

    try:
        data = loader(signature)
    except Exception as exc:
        st.error(f"Failed to load or validate data: {exc}")
        st.stop()

    if analysis_mode == "Space heating":
        display_title = "Residential Space-Heating Cost Calculator"
        display_caption = (
            "Space-heating technology scenarios. All costs are variable energy "
            "charges only; fixed customer charges, taxes, installation, and "
            "maintenance are not included."
        )
    else:
        display_title = "Residential Water-Heating Cost Calculator"
        display_caption = (
            "Water-heating technology scenarios. All costs are variable energy "
            "charges only; fixed customer charges, taxes, installation, and "
            "maintenance are not included."
        )

    title_placeholder.title(display_title)
    caption_placeholder.caption(display_caption)

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
        if st.button(
            "Calculate / Update",
            type="primary",
            use_container_width=True,
            key="statewide_calculate_button",
        ):
            st.session_state["applied_statewide"] = {
                "period": selected_period,
                "propane_price": float(selected_propane_price),
            }

        applied = st.session_state["applied_statewide"]
        st.caption(
            f"Applied: {PERIOD_OPTIONS[applied['period']]} · "
            f"Propane ${applied['propane_price']:.3f}/gal"
        )

    elif analysis_mode == "County / area scenario":
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
                data,
                electric_provider,
                selected_region_period,
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
            data,
            usage_sample,
            sample_counties,
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

        if st.button(
            "Calculate / Update",
            type="primary",
            use_container_width=True,
            key="region_calculate_button",
        ):
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

    else:
        selected_space_region = st.selectbox(
            "County / area",
            SPACE_HEATING_REGION_OPTIONS,
            key="space_region_input",
        )
        selected_space_period = st.selectbox(
            "Period",
            list(PERIOD_OPTIONS),
            format_func=lambda code: PERIOD_OPTIONS[code],
            key="space_period_input",
        )

        space_config = SPACE_HEATING_REGION_CONFIGS[selected_space_region]
        space_counties = tuple(space_config["counties"])
        space_electric_provider = str(space_config["electric_provider"])
        space_gas_provider = str(space_config["gas_provider"])

        try:
            space_electric_tariffs = available_electric_tariffs(
                data,
                space_electric_provider,
                selected_space_period,
            )
            space_gas_tariffs = available_gas_tariffs(
                data,
                space_gas_provider,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        if not space_electric_tariffs or not space_gas_tariffs:
            st.error("No complete tariff is available for the selected scenario.")
            st.stop()

        current_space_electric = st.session_state.get(
            "space_electric_tariff_input"
        )
        if current_space_electric not in space_electric_tariffs:
            st.session_state["space_electric_tariff_input"] = (
                space_electric_tariffs[0]
            )
        current_space_gas = st.session_state.get("space_gas_tariff_input")
        if current_space_gas not in space_gas_tariffs:
            st.session_state["space_gas_tariff_input"] = space_gas_tariffs[0]

        selected_space_electric_tariff = st.selectbox(
            f"Electricity tariff — {space_electric_provider}",
            space_electric_tariffs,
            format_func=tariff_name,
            key="space_electric_tariff_input",
        )
        selected_space_gas_tariff = st.selectbox(
            f"Gas tariff — {space_gas_provider}",
            space_gas_tariffs,
            format_func=tariff_name,
            key="space_gas_tariff_input",
        )
        selected_space_propane_price = st.number_input(
            "Propane price ($/gallon)",
            min_value=0.0,
            value=float(DEFAULT_PROPANE_PRICE_PER_GALLON),
            step=0.01,
            format="%.3f",
            key="space_propane_price_input",
        )

        space_sample_count = int(
            data.gt_space_electricity_usage["bldg_id"].nunique()
        )
        excluded_space_count = len(data.excluded_building_ids)
        sample_caption = (
            f"Sample counties: {', '.join(space_counties)} · "
            f"Eligible paired households: {space_sample_count}"
        )
        if excluded_space_count:
            sample_caption += (
                " · Excluded non-applicable/all-zero households: "
                f"{excluded_space_count}"
            )
        st.caption(sample_caption)

        default_space_applied = {
            "region": selected_space_region,
            "sample_counties": space_counties,
            "period": selected_space_period,
            "electric_provider": space_electric_provider,
            "electric_tariff": space_electric_tariffs[0],
            "gas_provider": space_gas_provider,
            "gas_tariff": space_gas_tariffs[0],
            "propane_price": float(DEFAULT_PROPANE_PRICE_PER_GALLON),
        }
        st.session_state.setdefault(
            "applied_space_heating",
            default_space_applied,
        )

        if st.button(
            "Calculate / Update",
            type="primary",
            use_container_width=True,
            key="space_calculate_button",
        ):
            st.session_state["applied_space_heating"] = {
                "region": selected_space_region,
                "sample_counties": space_counties,
                "period": selected_space_period,
                "electric_provider": space_electric_provider,
                "electric_tariff": selected_space_electric_tariff,
                "gas_provider": space_gas_provider,
                "gas_tariff": selected_space_gas_tariff,
                "propane_price": float(selected_space_propane_price),
            }

        applied = st.session_state["applied_space_heating"]
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
    show_summary_card(
        first_1,
        "HPWH monthly electricity cost",
        result.hpwh_summary,
        result.interval_95["hpwh"],
    )
    show_summary_card(
        first_2,
        "Natural-gas WH monthly cost",
        result.gas_summary,
        result.interval_95["gas"],
    )
    show_summary_card(
        first_3,
        "HPWH − Gas WH cost difference",
        result.hpwh_minus_gas_summary,
        result.interval_95["hpwh_minus_gas"],
    )

    st.markdown("### Additional water-heater scenarios")
    second_1, second_2 = st.columns(2)
    show_summary_card(
        second_1,
        "Propane WH monthly cost",
        result.propane_summary,
        result.interval_95["propane"],
    )
    show_summary_card(
        second_2,
        "Electric resistance WH monthly cost",
        result.resistance_summary,
        result.interval_95["resistance"],
    )

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

elif analysis_mode == "County / area scenario":
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
                    "usage_sample",
                    "mapped",
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

    if applied["region"] == "Traverse City":
        show_traverse_city_rate_tables(
            data,
            str(applied["electric_tariff"]),
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
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
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

else:
    applied = st.session_state["applied_space_heating"]
    with st.spinner("Calculating space-heating scenario..."):
        result = calculate_space_heating_cached(
            signature,
            applied["region"],
            tuple(applied["sample_counties"]),
            applied["electric_provider"],
            applied["electric_tariff"],
            applied["gas_provider"],
            applied["gas_tariff"],
            applied["period"],
            float(applied["propane_price"]),
        )

    st.subheader(
        f"Space-heating results — {applied['region']} · "
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
    usage_source_caption = (
        "Usage source: dedicated Grand Traverse component-level paired sample. "
        "Upgrade 4 contains five heating-system components and Upgrade 0 contains "
        "two components. Each electricity component is billed at the selected "
        "electric tariff and each natural-gas component at the selected gas tariff. "
        "Zero-use August/component rows are retained."
    )
    if data.excluded_building_ids:
        usage_source_caption += (
            f" {len(data.excluded_building_ids)} household(s) with no positive "
            "space-heating use across either energy source were excluded."
        )
    st.caption(usage_source_caption)

    # Average component costs. Scenario totals are calculated household-by-
    # household from these component costs, so their averages add to the total
    # before display rounding.
    hp_main_mean = float(
        result.households["hp_main_heating_electricity_cost"].mean()
    )
    hp_main_fan_mean = float(
        result.households["hp_main_fan_pump_electricity_cost"].mean()
    )
    hp_aux_electric_mean = float(
        result.households["hp_aux_electricity_cost"].mean()
    )
    hp_aux_fan_mean = float(
        result.households["hp_aux_fan_pump_electricity_cost"].mean()
    )
    hp_aux_gas_mean = float(
        result.households["hp_aux_natural_gas_cost"].mean()
    )

    gas_fuel_mean = float(
        result.households["gas_fuel_cost"].mean()
    )
    gas_fan_pump_mean = float(
        result.households["gas_fan_pump_electricity_cost"].mean()
    )
    propane_fuel_mean = float(
        result.households["propane_fuel_cost"].mean()
    )
    propane_fan_pump_mean = float(
        result.households["propane_fan_pump_electricity_cost"].mean()
    )

    first_1, first_2, first_3 = st.columns(3)
    show_summary_card(
        first_1,
        "Heat-pump space-heating monthly cost",
        result.heat_pump_summary,
        result.interval_95["heat_pump"],
        details=[
            ("(Main) HP space heating system", hp_main_mean),
            ("(Main_furnace) fan/pump with electricity", hp_main_fan_mean),
            ("(Aux) Auxiliary heating system powered by electricity", hp_aux_electric_mean),
            ("(Aux_furnace) fan/pump with electricity", hp_aux_fan_mean),
            ("(Aux) Auxiliary heating system powered by Natural gas", hp_aux_gas_mean),
        ],
    )
    show_summary_card(
        first_2,
        "Natural-gas space-heating monthly cost",
        result.gas_summary,
        result.interval_95["gas"],
        details=[
            ("(Main) Natural gas heating system", gas_fuel_mean),
            ("(Main_furnace) Blower/fan with electricity", gas_fan_pump_mean),
        ],
    )
    show_summary_card(
        first_3,
        "Heat pump − Gas cost difference",
        result.heat_pump_minus_gas_summary,
        result.interval_95["heat_pump_minus_gas"],
    )

    st.markdown("### Additional space-heating scenario")
    second_1, second_2 = st.columns(2)
    show_summary_card(
        second_1,
        "Propane space-heating monthly cost",
        result.propane_summary,
        result.interval_95["propane"],
        details=[
            ("Propane fuel", propane_fuel_mean),
            ("Fan/pump electricity", propane_fan_pump_mean),
        ],
    )
    second_2.empty()

    st.caption(
        "Central 95% keeps the original trimming basis: Upgrade 4 is trimmed "
        "by its electricity-use profile, while Upgrade 0 / propane are trimmed "
        "by main natural-gas use; the cost difference uses the intersection."
    )

    with st.expander("Space-heating assumptions and formulas", expanded=False):
        st.markdown(
            f"""
- Upgrade 4 profile: `Grand Traverse_Space heating_upgrade4_hourly_average_kwh.xlsx`
- Upgrade 0 profile: `Grand Traverse_Space heating_upgrade0_hourly_average_kwh.xlsx`
- Upgrade 4 components: (Main) HP space heating system, (Main_furnace) fan/pump with electricity, (Aux) auxiliary heating system powered by electricity, (Aux_furnace) fan/pump with electricity, and (Aux) auxiliary heating system powered by natural gas.
- Upgrade 4 cost = four electricity-component costs + natural-gas auxiliary cost.
- Upgrade 0 components: main natural-gas heating and blower/fan electricity.
- Upgrade 0 cost = main gas-heating cost + blower/fan electricity cost.
- Propane uses the Upgrade 0 main gas-heating input converted to gallons and retains the Upgrade 0 blower/fan electricity cost.
- Propane assumes equivalent combustion efficiency, so propane input energy = baseline natural-gas input energy.
- Propane gallons = natural-gas input kWh ÷ `{PROPANE_KWH_PER_GALLON:.3f} kWh/gallon`.
- Propane price: `${float(applied['propane_price']):.3f}/gallon`.
- January/August use the selected average daily profile multiplied by calendar days.
- Annual average applies the annual-average daily profile to all 12 monthly tariffs, totals the year, and divides by 12.
            """
        )

    show_traverse_city_rate_tables(
        data,
        str(applied["electric_tariff"]),
    )

    if st.checkbox(
        "Show space-heating household table",
        value=False,
        key="show_space_household_table",
    ):
        display = result.households.sort_values(
            "bldg_id",
            key=lambda series: pd.to_numeric(series, errors="coerce"),
            kind="stable",
        ).reset_index(drop=True)
        display.insert(0, "Area House No.", range(1, len(display) + 1))
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        st.download_button(
            "Download space-heating results as CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name="traverse_city_space_heating_scenario.csv",
            mime="text/csv",
        )
