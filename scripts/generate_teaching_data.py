#!/usr/bin/env python3
"""
Generate Canonical Teaching Data for Melbourne Retail Demand Forecasting.

Temporal Horizon: 2022-07-04 (Monday) to 2027-06-27 (Sunday)
Duration: 260 complete weeks (1,820 calendar days)

Scenarios:
1. grocery: FMCG daily continuous demand with stockout censoring and Good Friday/Christmas closures.
2. wine: Poisson count pulses with 3-day case-buy promotions and party events.
3. garden: Strong annual sinusoidal seasonality with DIY long weekend spikes and autumn clearances.
4. luxury: Extreme zero-inflated intermittent demand (93%+ zeros), high unit value, quarterly VIP events.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "teaching"

START_DATE = "2022-07-04"  # Monday
END_DATE = "2027-06-27"  # Sunday
TIME_ORIGIN = pd.Timestamp("2023-01-01")  # Consistent seasonal phase origin
HORIZON_DAYS = 28

SEEDS = {
    "grocery": 42,
    "wine": 2027,
    "garden": 2027,
    "luxury": 2085,
}

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
            "description": "Total number of calendar days in the simulation timeline (1820 days).",
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


def build_timeline():
    ds = pd.date_range(START_DATE, END_DATE, freq="D")
    assert ds[0].day_name() == "Monday", "Timeline must start on a Monday"
    assert ds[-1].day_name() == "Sunday", "Timeline must end on a Sunday"
    assert len(ds) % 7 == 0, "Timeline must contain complete weeks"
    t = ((ds - TIME_ORIGIN) / pd.Timedelta(days=1)).to_numpy()
    return ds, t


def build_vic_holidays(ds):
    calendar_years = list(range(ds.min().year, ds.max().year + 1))
    vic_calendar = holidays.country_holidays(
        "AU", subdiv="VIC", years=calendar_years, expand=False, language="en_US"
    )
    holiday_rows = [
        {"ds": pd.Timestamp(date), "holiday": name}
        for date in sorted(vic_calendar)
        for name in vic_calendar.get_list(date)
    ]
    vic_holidays = (
        pd.DataFrame(holiday_rows).sort_values(["ds", "holiday"]).reset_index(drop=True)
    )
    vic_holidays["lower_window"] = 0
    vic_holidays["upper_window"] = 0
    return vic_holidays


def generate_grocery(ds, t, vic_holidays, is_open):
    rng = np.random.default_rng(SEEDS["grocery"])
    discount = np.where(t % 14 < 7, 0.20, 0.0)
    weekend = (ds.dayofweek >= 5).astype(float)
    is_vic_holiday = ds.isin(vic_holidays["ds"].unique()).astype(float)
    annual = 14 * np.sin(2 * np.pi * (t - 15) / 365.25)
    conditional_mean = np.maximum(
        0,
        85 + 0.015 * t + 16 * weekend + annual + 130 * discount + 28 * is_vic_holiday,
    )
    latent = np.maximum(0, conditional_mean + rng.normal(0, 6, len(ds)))
    stockout = (rng.random(len(ds)) < 0.03) & is_open
    sales = np.where(stockout, np.minimum(latent, 20), latent)
    return pd.DataFrame(
        {
            "ds": ds,
            "y": np.where(is_open, np.rint(sales), 0.0),
            "is_open": is_open,
            "is_closed_holiday": ~is_open,
            "discount": discount,
            "stockout": stockout,
            "conditional_mean": conditional_mean,
            "latent_demand": latent,
        }
    )


def generate_wine(ds, t, vic_holidays, is_open):
    rng = np.random.default_rng(SEEDS["wine"])
    dow = ds.dayofweek
    december_peak = (ds.month == 12) & (ds.day >= 15) & (ds.day <= 24)
    cup_dates = pd.DatetimeIndex(
        vic_holidays.loc[
            vic_holidays.holiday.str.contains("Melbourne Cup", case=False), "ds"
        ]
    )
    afl_dates = pd.DatetimeIndex(
        vic_holidays.loc[
            vic_holidays.holiday.str.contains("Grand Final", case=False), "ds"
        ]
    )
    party = (
        ds.isin(cup_dates)
        | ds.isin(cup_dates - pd.Timedelta(days=1))
        | ds.isin(afl_dates)
        | ((ds.month == 12) & (ds.day == 31))
    )
    promo = (t % 28 < 3).astype(int)
    mean = (
        4.5
        + 0.002 * t
        + 2.8 * (dow == 3)
        + 7.5 * ((dow == 4) | (dow == 5))
        + 2 * (dow == 6)
    )
    mean += (
        3.5 * np.cos(2 * np.pi * (t - 190) / 365.25)
        + 6 * december_peak
        + 12 * party
        + 15 * promo
    )
    intensity = np.maximum(mean, 1e-6)
    latent = rng.poisson(intensity)
    return pd.DataFrame(
        {
            "ds": ds,
            "conditional_mean": intensity,
            "latent_demand": latent,
            "promo_case_buy": promo,
            "is_december_peak": december_peak.astype(int),
            "is_party_event": party.astype(int),
            "is_open": is_open,
            "y": np.where(is_open, np.rint(latent).astype(float), 0.0),
            "stockout": False,
        }
    )


def generate_garden(ds, t, vic_holidays, is_open):
    rng = np.random.default_rng(SEEDS["garden"])
    dow = ds.dayofweek
    labour = pd.DatetimeIndex(
        vic_holidays.loc[
            vic_holidays.holiday.str.contains(r"Labou?r Day", regex=True), "ds"
        ]
    )
    easter = pd.DatetimeIndex(
        vic_holidays.loc[
            vic_holidays.holiday.isin(
                ["Easter Saturday", "Easter Sunday", "Easter Monday"]
            ),
            "ds",
        ]
    )
    diy_dates = (
        labour.union(labour - pd.Timedelta(days=1))
        .union(labour - pd.Timedelta(days=2))
        .union(easter)
    )
    long_weekend = ds.isin(diy_dates)
    clearance = ((ds.month == 5) & (ds.day >= 25)).astype(int)
    mean = (
        50
        + 0.01 * t
        + 35 * (dow == 5)
        + 25 * (dow == 6)
        + 38 * np.sin(2 * np.pi * (t - 10) / 365.25)
        + 30 * long_weekend
        + 12 * clearance
    )
    latent = np.maximum(0, mean + rng.normal(0, 7, len(ds)))
    return pd.DataFrame(
        {
            "ds": ds,
            "conditional_mean": mean,
            "latent_demand": latent,
            "is_long_weekend_diy": long_weekend.astype(int),
            "is_end_of_autumn_clearance": clearance,
            "is_open": is_open,
            "y": np.where(is_open, np.rint(latent).astype(float), 0.0),
            "stockout": False,
        }
    )


def generate_luxury(ds, t, vic_holidays, is_open):
    rng = np.random.default_rng(SEEDS["luxury"])
    is_saturday = ds.dayofweek == 5
    is_sunday = ds.dayofweek == 6
    is_december_peak = ds.month == 12

    vip_dates = []
    for year in sorted(ds.year.unique()):
        for month in (3, 6, 9, 12):
            first_week = pd.date_range(f"{year}-{month:02d}-01", periods=7, freq="D")
            first_saturday = first_week[first_week.dayofweek == 5][0]
            vip_dates.extend([first_saturday, first_saturday + pd.Timedelta(days=1)])
    is_vip_event = ds.isin(pd.DatetimeIndex(vip_dates))

    discount = np.where((ds.month == 6) & (ds.day >= 20) & (ds.day <= 26), 0.20, 0.0)
    logit_probability = (
        -3.2
        + 0.5 * is_saturday
        + 0.3 * is_sunday
        + 1.1 * is_december_peak
        + 1.4 * is_vip_event
        + 0.8 * discount
    )
    occurrence_probability = 1 / (1 + np.exp(-logit_probability))
    has_demand = rng.binomial(1, occurrence_probability).astype(bool)
    sampled_positive_demand = rng.choice(
        [1, 2, 5, 10], size=len(ds), p=[0.75, 0.15, 0.08, 0.02]
    )
    positive_demand = np.where(has_demand, sampled_positive_demand, np.nan)
    latent_demand = np.where(has_demand, sampled_positive_demand, 0)
    conditional_mean = occurrence_probability * 1.65
    stockout = np.zeros(len(ds), dtype=bool)

    return pd.DataFrame(
        {
            "ds": ds,
            "y": np.where(is_open, latent_demand.astype(float), 0.0),
            "is_open": is_open,
            "is_closed_holiday": ~is_open,
            "is_saturday": is_saturday,
            "is_sunday": is_sunday,
            "is_december_peak": is_december_peak,
            "is_vip_event": is_vip_event,
            "discount": discount,
            "stockout": stockout,
            "occurrence_probability": occurrence_probability,
            "positive_demand": positive_demand,
            "conditional_mean": conditional_mean,
            "latent_demand": latent_demand,
        }
    )


def generate_all():
    print("\n========================================================")
    print("Generating 5-Year Teaching Datasets (2022-07-04 to 2027-06-27)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("========================================================\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds, t = build_timeline()
    vic_holidays = build_vic_holidays(ds)

    closed_dates = pd.DatetimeIndex(
        vic_holidays.loc[
            vic_holidays["holiday"].isin(["Good Friday", "Christmas Day"]), "ds"
        ]
    )
    is_open = ~ds.isin(closed_dates)

    cutoff = ds[-HORIZON_DAYS - 1]

    datasets = {
        "grocery": generate_grocery(ds, t, vic_holidays, is_open),
        "wine": generate_wine(ds, t, vic_holidays, is_open),
        "garden": generate_garden(ds, t, vic_holidays, is_open),
        "luxury": generate_luxury(ds, t, vic_holidays, is_open),
    }

    summary_rows = []
    csv_sha256 = {}

    for name, df in datasets.items():
        table_name = f"{name}_scenario_data"
        schema_fields = TABLE_SCHEMAS[table_name]
        expected_cols = [c["name"] for c in schema_fields]
        assert list(df.columns) == expected_cols, f"Column mismatch in {table_name}"
        assert df.ds.is_monotonic_increasing
        assert (df.y >= 0).all()
        assert (df.loc[~df.is_open, "y"] == 0).all()

        csv_path = OUTPUT_DIR / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
        csv_sha256[f"{table_name}.csv"] = hashlib.sha256(
            csv_path.read_bytes()
        ).hexdigest()

        # Schema JSON
        schema_path = OUTPUT_DIR / f"{table_name}.schema.json"
        schema_path.write_text(
            json.dumps(schema_fields, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        opened = df.loc[df.is_open]
        positive = opened.loc[opened.y > 0, "y"]
        pos_days = len(positive)
        adi = len(opened) / pos_days if pos_days else np.nan
        cv_plus_sq = (
            positive.var(ddof=1) / positive.mean() ** 2 if pos_days > 1 else np.nan
        )

        summary_rows.append(
            {
                "scenario": name,
                "rows": len(df),
                "closed_days": int((~df.is_open).sum()),
                "stockout_days": int(df.stockout.sum()),
                "open_mean_sales": float(opened.y.mean()),
                "open_zero_rate": float(opened.y.eq(0).mean()),
                "positive_demand_days": pos_days,
                "ADI": float(adi),
                "CV_plus_squared": float(cv_plus_sq),
                "open_min_conditional_mean": float(opened.conditional_mean.min()),
                "final_test_days": int((df.ds > cutoff).sum()),
            }
        )
        print(f"Generated {table_name}.csv ({len(df)} rows, ADI={adi:.2f})")

    # Save vic_holidays
    vic_holidays_path = OUTPUT_DIR / "vic_holidays.csv"
    vic_holidays.to_csv(vic_holidays_path, index=False, date_format="%Y-%m-%d")
    csv_sha256["vic_holidays.csv"] = hashlib.sha256(
        vic_holidays_path.read_bytes()
    ).hexdigest()
    (OUTPUT_DIR / "vic_holidays.schema.json").write_text(
        json.dumps(TABLE_SCHEMAS["vic_holidays"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Generated vic_holidays.csv ({len(vic_holidays)} holiday records)")

    # Save scenario_diagnostics
    diag_df = pd.DataFrame(summary_rows)
    diag_path = OUTPUT_DIR / "scenario_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    csv_sha256["scenario_diagnostics.csv"] = hashlib.sha256(
        diag_path.read_bytes()
    ).hexdigest()
    (OUTPUT_DIR / "scenario_diagnostics.schema.json").write_text(
        json.dumps(TABLE_SCHEMAS["scenario_diagnostics"], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print("Generated scenario_diagnostics.csv")

    # Save generation_metadata.json
    metadata = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "time_origin": str(TIME_ORIGIN.date()),
        "horizon_days": HORIZON_DAYS,
        "cutoff": str(cutoff.date()),
        "diagnostics_file": "scenario_diagnostics.csv",
        "seeds": SEEDS,
        "versions": {
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "holidays": version("holidays"),
        },
        "closed_holiday_names": ["Good Friday", "Christmas Day"],
        "regressor_columns": {
            "grocery": ["discount"],
            "wine": ["promo_case_buy", "is_december_peak", "is_party_event"],
            "garden": ["is_end_of_autumn_clearance", "is_long_weekend_diy"],
            "luxury": [
                "is_saturday",
                "is_sunday",
                "is_december_peak",
                "is_vip_event",
                "discount",
            ],
        },
        "diagnostic_only_columns": [
            "conditional_mean",
            "occurrence_probability",
            "positive_demand",
            "latent_demand",
        ],
        "csv_sha256": csv_sha256,
    }

    meta_path = OUTPUT_DIR / "generation_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Saved generation_metadata.json with updated SHA-256 hashes.")
    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    generate_all()
