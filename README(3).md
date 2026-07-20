# HPWH vs Gas Water Heater Calculator — v2.4 Range Version

## User input

Only one input remains:

- Annual average
- January
- August

Provider and tariff selectors are removed.

## Calculation logic

Each mapped household is automatically assigned its electricity and gas
providers from the provider workbook.

For every household:

1. The app finds all complete electricity tariffs for the mapped electricity
   provider.
2. It calculates an HPWH cost under every applicable tariff.
3. It calculates the mapped gas-provider cost.
4. It calculates savings for each electricity-tariff scenario.

The result cards display:

- Minimum across all valid household–tariff scenarios
- Maximum across all valid household–tariff scenarios
- Household-weighted average

For the average, tariff results are first averaged within each household. The
household means are then averaged, so utilities with more tariff options do not
give their households extra weight.

## Result documentation

The app includes:

1. House sampling and county distribution
2. County-level electricity and gas provider mapping
3. Electricity and gas rates, including tariff types and hourly detail
4. Household–tariff calculation detail
5. Missing-rate coverage records

## Current workbook coverage

The current electricity workbook contains complete numeric rates for DTE
Electric Company and Consumers Energy Company. Other mapped providers will be
included automatically as complete rows are added with the same workbook
structure.

## GitHub structure

```text
hpwh-cost-calculator/
├── app.py
├── data_loading.py
├── requirements.txt
├── README.md
└── data/
    ├── MI_housesample_elec_hourly_average_kwh.xlsx
    ├── MI_housesample_gas_hourly_average_kwh.xlsx
    ├── MI_provider_county_with_utility_providers.xlsx
    ├── electricity_rates_weekdays_202607.xlsx
    └── gas_rates_weekdays_converted_to_kwh_202607.xlsx
```
