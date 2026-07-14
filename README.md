# HPWH vs Gas Water Heater — Monthly Cost Calculator (DTE, Michigan)

Interactive Streamlit app comparing the monthly operating cost of a **Heat Pump Water Heater (HPWH)** vs a **standard natural-gas water heater** in Michigan, using ResStock consumption data and DTE tariffs.

## Formulas

```
HPWH:   Cost_{p,m} = 30 × E × Σ_h ( f_h · r_{p,m,h} )
Gas WH: Cost_g     = 30 × E_gas × r_g        (flat rate; Σ f_h = 1)
```

| Symbol | Meaning | Source |
|---|---|---|
| `E` | daily HPWH energy consumption (kWh/day) | ResStock MI_upgrade0, rows where `in.water_heater_efficiency` contains "Electric Heat Pump" |
| `E_gas` | daily Gas WH consumption (kWh/day) | same file, rows containing "Natural Gas Standard" |
| `f_h` | usage fraction at hour h (Σ = 1) | `base-schedules-simple.xml`, water fixtures schedule |
| `r_{p,m,h}` | electricity rate, plan p / month m / hour h ($/kWh) | `data/electricity_rates_weekdays_202607.xlsx` |
| `r_g` | flat gas rate ($/kWh) | `data/gas_rates_weekdays_converted_to_kwh_202607.xlsx` |

## Repository structure

```
app.py                # Streamlit web app
data_loading.py       # reads/cleans the data files below
data/
  MI_upgrade0_downsize.xlsx                        # 4 columns needed by the app (557KB)
  electricity_rates_weekdays_202607.xlsx           # TOU/TOD/OS × 12 months × h0-h23
  gas_rates_weekdays_converted_to_kwh_202607.xlsx  # flat $/kWh
requirements.txt
```

**Why a downsized file?** The original `MI_upgrade0.xlsx` is 76 MB (771 columns), which
exceeds GitHub's comfortable file-size limits. Only the 4 columns the app needs are kept
in `MI_upgrade0_downsize.xlsx` (557 KB), which is committed instead.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app → pick the repo and `app.py`.
3. Share the generated public URL.

## Data notes

- The electricity-rates file uses merged cells for `state/tariff/year` and has a
  legend row at the top; `data_loading.py` handles both automatically.
- The water-heater type filter uses **`in.water_heater_efficiency`**
  (not `in.water_heater_fuel`, which only contains fuel names).
- HPWH sample size in MI_upgrade0 (baseline) is small (26 homes); consumption
  defaults can be overridden in the app sidebar.
