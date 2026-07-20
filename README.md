# HPWH vs Gas Water Heater Calculator — v2.1

The new provider workbook contains a real building ID, so the calculator no
longer connects provider rows to buildings by row order.

## Data folder

Place the following files in `data/`:

- `MI_housesample_elec_hourly_average_kwh.xlsx`
- `MI_housesample_gas_hourly_average_kwh.xlsx`
- `MI_provider_county_with_utility_providers.xlsx`
- `electricity_rates_weekdays_202607.xlsx`
- `gas_rates_weekdays_converted_to_kwh_202607.xlsx`

The included provider workbook standardizes `MI_bldg_id` to `bldg_id`.
The loader accepts either column name.

## Provider-file validation

- Buildings in each hourly-usage file: 153
- Buildings in the new provider file: 149
- Duplicate provider building IDs: 0
- Blank required provider values: 0
- Missing provider mappings: `149617`, `179261`, `288908`, `448182`

The four unmapped buildings are excluded from the selector. No provider is
guessed or assigned by row order.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
