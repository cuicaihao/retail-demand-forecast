"""Generate BigQuery-compatible schema JSON files with complete English descriptions

for all teaching datasets in data/teaching/.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "teaching"

TABLE_SCHEMAS: dict[str, list[dict[str, str]]] = {
    "grocery_scenario_data": [
        {
            "name": "ds",
            "type": "DATE",
            "mode": "REQUIRED",
            "description": "Daily calendar date in YYYY-MM-DD format.",
        },
        {
            "name": "y",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Observed daily sales quantity in units (numeric/float representation, e.g. 153.0). Truncated on stockout days, zero on closed holiday days.",
        },
        {
            "name": "is_open",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Store operational status indicator (True = open for trading, False = closed). Closed on Good Friday and Christmas Day.",
        },
        {
            "name": "is_closed_holiday",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator whether the store is closed due to a designated public holiday (Good Friday or Christmas Day).",
        },
        {
            "name": "discount",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Promotional price discount rate (0.20 for 20% discount on alternating 7-day cycles, 0.0 otherwise).",
        },
        {
            "name": "stockout",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Inventory stockout indicator (True = store ran out of stock with observed sales capped at 20 units, False = normal inventory).",
        },
        {
            "name": "conditional_mean",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "True expected unconstrained demand under normal trading conditions (data-generating process ground truth; diagnostic only).",
        },
        {
            "name": "latent_demand",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Simulated unconstrained latent demand with additive Gaussian noise before stockout truncation and store closure (diagnostic only).",
        },
    ],
    "wine_scenario_data": [
        {
            "name": "ds",
            "type": "DATE",
            "mode": "REQUIRED",
            "description": "Daily calendar date in YYYY-MM-DD format.",
        },
        {
            "name": "conditional_mean",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "True Poisson intensity parameter (lambda) representing expected daily demand in bottles (data-generating process ground truth; diagnostic only).",
        },
        {
            "name": "latent_demand",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Simulated unconstrained latent demand sampled from Poisson distribution before applying store closure (diagnostic only).",
        },
        {
            "name": "promo_case_buy",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Indicator for recurring 3-day case-buy promotional discount event every 28 days (1 = active, 0 = inactive).",
        },
        {
            "name": "is_december_peak",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Indicator for high-season pre-Christmas wine buying period from December 15 to 24 (1 = active, 0 = inactive).",
        },
        {
            "name": "is_party_event",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Indicator for major celebration/party events: AFL Grand Final Friday, Melbourne Cup Eve/Day, and New Year's Eve (1 = active, 0 = inactive).",
        },
        {
            "name": "is_open",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Store operational status indicator (True = open for trading, False = closed). Closed on Good Friday and Christmas Day.",
        },
        {
            "name": "y",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Observed daily sales quantity in 750mL bottles (zero on closed holiday days).",
        },
        {
            "name": "stockout",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Inventory stockout indicator (False throughout this scenario; no stockout simulated).",
        },
    ],
    "garden_scenario_data": [
        {
            "name": "ds",
            "type": "DATE",
            "mode": "REQUIRED",
            "description": "Daily calendar date in YYYY-MM-DD format.",
        },
        {
            "name": "conditional_mean",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "True expected daily sales demand under normal operating conditions with annual seasonality (data-generating process ground truth; diagnostic only).",
        },
        {
            "name": "latent_demand",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Simulated continuous demand with additive Gaussian noise before store closure (diagnostic only).",
        },
        {
            "name": "is_long_weekend_diy",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Indicator for home improvement DIY demand surge over long weekend windows (Labour Day weekend and Easter weekend; 1 = active, 0 = inactive).",
        },
        {
            "name": "is_end_of_autumn_clearance",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Indicator for annual end-of-autumn clearance sale from May 25 to May 31 (1 = active, 0 = inactive).",
        },
        {
            "name": "is_open",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Store operational status indicator (True = open for trading, False = closed). Closed on Good Friday and Christmas Day.",
        },
        {
            "name": "y",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Observed daily sales quantity in units (rounded to integer, formatted as float; zero on closed holiday days).",
        },
        {
            "name": "stockout",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Inventory stockout indicator (False throughout this scenario; no stockout simulated).",
        },
    ],
    "luxury_scenario_data": [
        {
            "name": "ds",
            "type": "DATE",
            "mode": "REQUIRED",
            "description": "Daily calendar date in YYYY-MM-DD format.",
        },
        {
            "name": "y",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Observed daily sales quantity in units of high-value luxury handbags (zero on days with zero demand or store closure).",
        },
        {
            "name": "is_open",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Store operational status indicator (True = open for trading, False = closed). Closed on Good Friday and Christmas Day.",
        },
        {
            "name": "is_closed_holiday",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator whether the store is closed due to a designated public holiday (Good Friday or Christmas Day).",
        },
        {
            "name": "is_saturday",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator for Saturday shopping traffic.",
        },
        {
            "name": "is_sunday",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator for Sunday shopping traffic.",
        },
        {
            "name": "is_december_peak",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator for December luxury holiday gifting season.",
        },
        {
            "name": "is_vip_event",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Indicator for quarterly private VIP preview shopping weekend events (first weekend of March, June, September, and December).",
        },
        {
            "name": "discount",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Mid-year EOFY clearance discount rate (0.20 from June 20 to 26, 0.0 otherwise).",
        },
        {
            "name": "stockout",
            "type": "BOOLEAN",
            "mode": "REQUIRED",
            "description": "Inventory stockout indicator (False throughout this scenario; no stockout simulated).",
        },
        {
            "name": "occurrence_probability",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Daily Bernoulli probability of positive purchase occurrence from logistic regression formula (data-generating process ground truth; diagnostic only).",
        },
        {
            "name": "positive_demand",
            "type": "FLOAT",
            "mode": "NULLABLE",
            "description": "Positive purchase quantity sampled from discrete distribution {1:0.75, 2:0.15, 5:0.08, 10:0.02} when demand occurs (NaN on zero-demand days; diagnostic only).",
        },
        {
            "name": "conditional_mean",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "True conditional expected daily sales demand (occurrence_probability * 1.65; data-generating process ground truth; diagnostic only).",
        },
        {
            "name": "latent_demand",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Simulated unconstrained daily sales quantity before store closure (diagnostic only).",
        },
    ],
    "vic_holidays": [
        {
            "name": "ds",
            "type": "DATE",
            "mode": "REQUIRED",
            "description": "Calendar date of the holiday event in YYYY-MM-DD format.",
        },
        {
            "name": "holiday",
            "type": "STRING",
            "mode": "REQUIRED",
            "description": "Official English name of the public holiday or special observance in Victoria (VIC), Australia.",
        },
        {
            "name": "lower_window",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Prior window offset in days for Prophet holiday impact modeling (default: 0).",
        },
        {
            "name": "upper_window",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Forward window offset in days for Prophet holiday impact modeling (default: 0).",
        },
    ],
    "scenario_diagnostics": [
        {
            "name": "scenario",
            "type": "STRING",
            "mode": "REQUIRED",
            "description": "Retail scenario identifier (grocery, wine, garden, luxury).",
        },
        {
            "name": "rows",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Total number of calendar days in the simulation timeline (1096 days).",
        },
        {
            "name": "closed_days",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Total number of days when store was closed for business (Good Friday and Christmas Day).",
        },
        {
            "name": "stockout_days",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Total number of days affected by stockout truncation.",
        },
        {
            "name": "open_mean_sales",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Average daily observed sales during store-open days.",
        },
        {
            "name": "open_zero_rate",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Proportion of store-open days with zero observed sales.",
        },
        {
            "name": "positive_demand_days",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Count of store-open days with positive sales (y > 0).",
        },
        {
            "name": "ADI",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Average Demand Interval (open_days / positive_demand_days), measuring demand intermittency.",
        },
        {
            "name": "CV_plus_squared",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Squared coefficient of variation of positive sales quantities, measuring demand lumpiness.",
        },
        {
            "name": "open_min_conditional_mean",
            "type": "FLOAT",
            "mode": "REQUIRED",
            "description": "Minimum conditional expected daily demand during open days.",
        },
        {
            "name": "final_test_days",
            "type": "INTEGER",
            "mode": "REQUIRED",
            "description": "Number of days reserved in the final holdout test evaluation period (28 days).",
        },
    ],
}


def validate_and_write_schemas() -> None:
    for table_name, schema_fields in TABLE_SCHEMAS.items():
        csv_file = DATA_DIR / f"{table_name}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(f"Missing CSV file: {csv_file}")

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            csv_header = next(reader)

        schema_cols = [col["name"] for col in schema_fields]
        if csv_header != schema_cols:
            raise ValueError(
                f"Column mismatch for {table_name}:\n"
                f"  CSV:    {csv_header}\n"
                f"  Schema: {schema_cols}"
            )

        schema_file = DATA_DIR / f"{table_name}.schema.json"
        content = json.dumps(schema_fields, indent=2, ensure_ascii=False) + "\n"
        schema_file.write_text(content, encoding="utf-8")
        print(f"Generated {schema_file.name} ({len(schema_fields)} fields)")


if __name__ == "__main__":
    validate_and_write_schemas()
