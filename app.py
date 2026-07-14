"""
HPWH vs Gas Water Heater - Monthly Cost Calculator (DTE, Michigan)

Formulas
--------
HPWH:    Cost_{p,m} = 30 * E     * sum_h( f_h * r_{p,m,h} )
Gas WH:  Cost_g     = 30 * E_gas * r_g          (flat rate; sum_h f_h = 1)

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from data_loading import (
    TARIFF_LABELS,
    USAGE_FRACTIONS,
    get_hourly_rates,
    load_daily_consumption,
    load_electricity_rates,
    load_gas_rate,
)

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@st.cache_data
def load_all():
    rates = load_electricity_rates()
    r_g = load_gas_rate()
    consumption = load_daily_consumption()
    return rates, r_g, consumption


st.set_page_config(page_title="HPWH vs Gas WH Cost", page_icon="💧", layout="centered")
st.title("HPWH vs Gas Water Heater — Monthly Cost")
st.caption("ResStock MI (upgrade0) consumption × DTE tariffs × water-fixture hourly schedule")

try:
    rates, r_g_default, consumption = load_all()
except Exception as e:
    st.error(f"Failed to load data files in ./data — {e}")
    st.stop()

E_default = consumption["hpwh"]["daily_kwh"]
Eg_default = consumption["gas"]["daily_kwh"]

# ------------------------------------------------------------- inputs -------
with st.sidebar:
    st.header("Inputs")

    tariff_codes = [t for t in TARIFF_LABELS if t in set(rates.tariff.unique())]
    tariff = st.selectbox(
        "Electricity rate plan (p)", tariff_codes,
        format_func=lambda t: TARIFF_LABELS.get(t, t),
    )
    years = sorted(rates.year.unique())
    year = st.selectbox("Year", years, index=len(years) - 1)
    months = sorted(rates[(rates.tariff == tariff) & (rates.year == year)].month.unique())
    month = st.selectbox(
        "Month (m)", months, format_func=lambda m: f"{m:02d} — {MONTH_NAMES[m-1]}"
    )

    st.divider()
    st.subheader("Consumption & gas rate")
    E_hpwh = st.number_input("HPWH E (kWh/day)", 0.0, 100.0, float(E_default), 0.1,
                             help=f"From ResStock: {consumption['hpwh']['n_rows']} homes, "
                                  f"annual avg {consumption['hpwh']['annual_avg_kwh']:.0f} kWh")
    E_gas = st.number_input("Gas WH E_gas (kWh/day)", 0.0, 200.0, float(Eg_default), 0.1,
                            help=f"From ResStock: {consumption['gas']['n_rows']} homes, "
                                 f"annual avg {consumption['gas']['annual_avg_kwh']:.0f} kWh")
    r_g = st.number_input("Gas rate r_g ($/kWh)", 0.0, 1.0, float(r_g_default), 0.001,
                          format="%.6f")

# ------------------------------------------------------------- compute ------
hourly_rates = get_hourly_rates(rates, tariff, year, month)
cost_hpwh = 30.0 * E_hpwh * sum(f * r for f, r in zip(USAGE_FRACTIONS, hourly_rates))
cost_gas = 30.0 * E_gas * r_g
savings = cost_gas - cost_hpwh

# ------------------------------------------------------------- results ------
c1, c2, c3 = st.columns(3)
c1.metric("HPWH monthly cost", f"${cost_hpwh:,.2f}")
c2.metric("Gas WH monthly cost", f"${cost_gas:,.2f}")
c3.metric("HPWH savings vs Gas", f"${savings:,.2f}")

st.caption(
    f"Plan: **{TARIFF_LABELS.get(tariff, tariff)}** · **{MONTH_NAMES[month-1]} {year}** · "
    f"E = {E_hpwh:.2f} kWh/day · E_gas = {E_gas:.2f} kWh/day · r_g = {r_g:.6f} $/kWh"
)

st.divider()
st.subheader("Hourly profile")
df = pd.DataFrame({
    "hour": list(range(24)),
    "usage fraction f_h": USAGE_FRACTIONS,
    "electricity rate ($/kWh)": hourly_rates,
})
df["HPWH hourly cost ($/day)"] = (
    E_hpwh * df["usage fraction f_h"] * df["electricity rate ($/kWh)"]
)

tab1, tab2, tab3 = st.tabs(["Rate by hour", "Usage by hour", "HPWH cost by hour"])
with tab1:
    st.bar_chart(df.set_index("hour")["electricity rate ($/kWh)"])
with tab2:
    st.bar_chart(df.set_index("hour")["usage fraction f_h"])
with tab3:
    st.bar_chart(df.set_index("hour")["HPWH hourly cost ($/day)"])

with st.expander("Show hourly table"):
    st.dataframe(df, use_container_width=True)

st.divider()
st.markdown(
    r"""
**Formulas** &nbsp;&nbsp;
$\text{Cost}_{p,m} = 30 \times E \times \sum_{h=1}^{24} f_h \cdot r_{p,m,h}$
&nbsp;&nbsp;·&nbsp;&nbsp;
$\text{Cost}_{g} = 30 \times E_{gas} \times r_g$
"""
)
