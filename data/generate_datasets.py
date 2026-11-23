"""
generate_datasets.py
--------------------
Creates the three Sri Lankan-context datasets used in this project.

IMPORTANT (stated honestly in the report/video):
  * sl_telecom_churn.csv  -> SYNTHETIC. Simulated customers of a Sri Lankan mobile
    operator (25 districts, LKR spend, eZ Cash/reload payments, dual-SIM behaviour).
  * sl_vehicle_prices.csv -> SYNTHETIC. Simulated ikman.lk-style used-vehicle listings
    (common Sri Lankan models, LKR prices, recondition vs registered, import-ban effect).
  * colombo_weather_daily.csv -> SYNTHETIC FALLBACK only. The notebook and dashboard first
    try to download REAL Colombo daily weather from the free Open-Meteo archive API and
    use this file only if there is no internet.

The generating rules are based on public, well-known patterns of the Sri Lankan context
(population by district, monsoon rainfall seasons, vehicle depreciation, 2020-2024 vehicle
import restrictions). Fixed random seed -> fully reproducible.

Run:  python generate_datasets.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(2026)

# --------------------------------------------------------------------------------------
# 1. TELECOM CHURN
# --------------------------------------------------------------------------------------
# district: (population '000, province, urban share, network quality baseline 1-5, tamil share)
DISTRICTS = {
    "Colombo": (2470, "Western", 0.85, 4.5, 0.12), "Gampaha": (2430, "Western", 0.55, 4.3, 0.05),
    "Kalutara": (1300, "Western", 0.30, 4.0, 0.06), "Kandy": (1460, "Central", 0.35, 3.9, 0.15),
    "Matale": (520, "Central", 0.15, 3.4, 0.10), "Nuwara Eliya": (760, "Central", 0.12, 3.0, 0.55),
    "Galle": (1100, "Southern", 0.30, 4.0, 0.03), "Matara": (840, "Southern", 0.25, 3.8, 0.03),
    "Hambantota": (650, "Southern", 0.12, 3.4, 0.02), "Jaffna": (620, "Northern", 0.30, 3.3, 0.98),
    "Kilinochchi": (130, "Northern", 0.10, 2.6, 0.97), "Mannar": (110, "Northern", 0.10, 2.6, 0.90),
    "Vavuniya": (190, "Northern", 0.20, 2.9, 0.85), "Mullaitivu": (100, "Northern", 0.05, 2.4, 0.95),
    "Batticaloa": (590, "Eastern", 0.25, 3.1, 0.90), "Ampara": (720, "Eastern", 0.20, 3.0, 0.60),
    "Trincomalee": (420, "Eastern", 0.25, 3.1, 0.70), "Kurunegala": (1700, "North Western", 0.20, 3.7, 0.05),
    "Puttalam": (820, "North Western", 0.18, 3.4, 0.25), "Anuradhapura": (930, "North Central", 0.15, 3.3, 0.05),
    "Polonnaruwa": (440, "North Central", 0.12, 3.1, 0.05), "Badulla": (870, "Uva", 0.12, 3.0, 0.25),
    "Monaragala": (500, "Uva", 0.08, 2.8, 0.05), "Ratnapura": (1150, "Sabaragamuwa", 0.15, 3.3, 0.10),
    "Kegalle": (870, "Sabaragamuwa", 0.15, 3.5, 0.08),
}


def make_churn(n=8000):
    names = list(DISTRICTS)
    w = np.array([DISTRICTS[d][0] for d in names], float)
    district = rng.choice(names, n, p=w / w.sum())
    prov = np.array([DISTRICTS[d][1] for d in district])
    urban_p = np.array([DISTRICTS[d][2] for d in district])
    netq_base = np.array([DISTRICTS[d][3] for d in district])
    tamil_p = np.array([DISTRICTS[d][4] for d in district])

    area = np.where(rng.random(n) < urban_p, "Urban", "Rural")
    lang_r = rng.random(n)
    language = np.where(lang_r < tamil_p, "Tamil", np.where(lang_r < tamil_p + 0.08, "English", "Sinhala"))
    age = np.clip(rng.normal(36, 12, n), 16, 78).round().astype(int)
    gender = rng.choice(["Male", "Female"], n, p=[0.52, 0.48])
    plan = np.where(rng.random(n) < 0.22 + 0.15 * (area == "Urban"), "Postpaid", "Prepaid")
    tenure = np.clip(rng.gamma(1.6, 18, n), 1, 144).round().astype(int)

    is_post = plan == "Postpaid"
    spend = np.where(is_post, rng.lognormal(np.log(2600), 0.45, n), rng.lognormal(np.log(850), 0.6, n))
    spend = np.round(spend / 10) * 10
    data_gb = np.clip(rng.gamma(2.0, 4.0, n) * (1.6 - (age - 16) / 80) * np.where(area == "Urban", 1.35, 0.9), 0, 120).round(1)
    voice_min = np.clip(rng.gamma(2.2, 90, n) * (0.6 + (age - 16) / 60), 0, 2500).round().astype(int)
    netq = np.clip(netq_base - np.where(area == "Rural", 0.4, 0) + rng.normal(0, 0.45, n), 1, 5).round(1)
    drop_rate = np.clip(rng.gamma(1.5, 0.6, n) * (5.6 - netq) / 2, 0, 15).round(2)
    complaints = rng.poisson(np.clip(0.25 + 0.35 * (4.2 - netq) + 0.15 * drop_rate, 0.05, None))
    dual_sim = (rng.random(n) < 0.30 + 0.20 * (age < 30)).astype(int)
    fibre = (rng.random(n) < np.where(area == "Urban", 0.22, 0.06)).astype(int)
    comp_offer = (rng.random(n) < 0.35).astype(int)
    days_reload = np.where(is_post, 0, np.clip(rng.exponential(9, n), 0, 60).round()).astype(int)
    late_bill = np.where(is_post, rng.poisson(0.6, n), 0)
    contract_left = np.where(is_post, rng.integers(0, 25, n), 0)
    pay = np.where(is_post, rng.choice(["Bank card", "Bill payment", "eZ Cash"], n, p=[0.45, 0.35, 0.20]),
                   rng.choice(["Reload card", "Retailer reload", "eZ Cash", "Bank card"], n, p=[0.25, 0.45, 0.22, 0.08]))
    pts = (rng.random(n) < 0.3 + 0.004 * tenure).astype(int)

    z = (-2.35
         + 1.00 * dual_sim
         + 0.90 * comp_offer * dual_sim            # interaction: competitor offer hits dual-SIM users
         + 0.25 * comp_offer
         + 0.38 * complaints
         + 0.85 * (netq < 2.6)
         + 0.75 * ((data_gb > 15) & (netq < 3.2))  # heavy data users on poor network
         + 0.055 * np.minimum(days_reload, 40)      # inactivity (prepaid)
         + 0.30 * late_bill
         - 0.55 * np.log1p(tenure) + 1.5
         - 1.20 * fibre
         - 0.90 * (is_post & (contract_left > 6))
         - 0.45 * pts
         + 0.45 * (age < 25)
         + 0.25 * (spend < 400)
         + rng.normal(0, 0.55, n))
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame({
        "customer_id": [f"SLC{100000 + i}" for i in range(n)],
        "district": district, "province": prov, "area_type": area, "language": language,
        "age": age, "gender": gender, "plan_type": plan, "tenure_months": tenure,
        "monthly_spend_lkr": spend, "data_usage_gb": data_gb, "voice_minutes": voice_min,
        "network_quality_score": netq, "dropped_call_rate_pct": drop_rate,
        "complaints_6m": complaints, "dual_sim": dual_sim, "has_fibre_bundle": fibre,
        "competitor_offer_received": comp_offer, "days_since_last_reload": days_reload,
        "late_bill_payments": late_bill, "contract_months_left": contract_left,
        "payment_method": pay, "uses_loyalty_points": pts, "churn": churn,
    })


# --------------------------------------------------------------------------------------
# 2. VEHICLE PRICES (ikman.lk style)
# --------------------------------------------------------------------------------------
# model: (brand, fuel, engine_cc, body, base price LKR mn for a 2018-ish unit, first yom, annual depreciation)
MODELS = {
    "Aqua": ("Toyota", "Hybrid", 1500, "Hatchback", 9.6, 2012, 0.055),
    "Vitz": ("Toyota", "Petrol", 1000, "Hatchback", 8.2, 2005, 0.050),
    "Prius": ("Toyota", "Hybrid", 1800, "Sedan", 13.8, 2010, 0.065),
    "Axio": ("Toyota", "Hybrid", 1500, "Sedan", 11.8, 2008, 0.050),
    "Premio": ("Toyota", "Petrol", 1500, "Sedan", 17.5, 2008, 0.045),
    "C-HR": ("Toyota", "Hybrid", 1800, "SUV", 16.8, 2017, 0.060),
    "Land Cruiser Prado": ("Toyota", "Diesel", 2800, "SUV", 48.0, 2005, 0.045),
    "Wagon R": ("Suzuki", "Hybrid", 660, "Hatchback", 6.9, 2014, 0.055),
    "Alto": ("Suzuki", "Petrol", 800, "Hatchback", 3.6, 2005, 0.050),
    "Swift": ("Suzuki", "Petrol", 1200, "Hatchback", 6.6, 2008, 0.055),
    "Fit": ("Honda", "Hybrid", 1500, "Hatchback", 8.6, 2011, 0.060),
    "Vezel": ("Honda", "Hybrid", 1500, "SUV", 14.2, 2014, 0.055),
    "Grace": ("Honda", "Hybrid", 1500, "Sedan", 10.2, 2014, 0.060),
    "Civic": ("Honda", "Petrol", 1500, "Sedan", 15.5, 2006, 0.055),
    "Leaf": ("Nissan", "Electric", 0, "Hatchback", 7.8, 2012, 0.110),
    "X-Trail": ("Nissan", "Hybrid", 2000, "SUV", 16.2, 2010, 0.065),
    "Montero": ("Mitsubishi", "Diesel", 3200, "SUV", 29.0, 2005, 0.050),
    "Axia": ("Perodua", "Petrol", 1000, "Hatchback", 4.9, 2015, 0.060),
    "Panda": ("Micro", "Petrol", 1000, "Hatchback", 2.7, 2010, 0.070),
    "Axela": ("Mazda", "Petrol", 1500, "Sedan", 9.2, 2010, 0.065),
    "Tucson": ("Hyundai", "Diesel", 2000, "SUV", 15.2, 2010, 0.065),
    "Sorento": ("Kia", "Diesel", 2200, "SUV", 17.4, 2010, 0.065),
}
MODEL_POP = {"Aqua": 12, "Vitz": 10, "Prius": 6, "Axio": 7, "Premio": 6, "C-HR": 3, "Land Cruiser Prado": 2,
             "Wagon R": 11, "Alto": 9, "Swift": 4, "Fit": 7, "Vezel": 6, "Grace": 4, "Civic": 2, "Leaf": 3,
             "X-Trail": 2, "Montero": 2, "Axia": 3, "Panda": 2, "Axela": 2, "Tucson": 1.5, "Sorento": 1.5}
CITY = {"Colombo": 0.34, "Gampaha": 0.20, "Kandy": 0.09, "Kurunegala": 0.08, "Kalutara": 0.06, "Galle": 0.06,
        "Matara": 0.04, "Ratnapura": 0.03, "Anuradhapura": 0.03, "Jaffna": 0.03, "Badulla": 0.02, "Kegalle": 0.02}


def make_vehicles(n=6000, ref_year=2025):
    names = list(MODEL_POP)
    p = np.array([MODEL_POP[m] for m in names]); p = p / p.sum()
    model = rng.choice(names, n, p=p)
    rows = []
    for m in model:
        brand, fuel, cc, body, base, first, dep = MODELS[m]
        yom = int(np.clip(rng.normal(2015, 3.6), first, 2024))
        age = ref_year - yom
        recon = yom >= 2022 and rng.random() < 0.55          # post-2025 import reopening: unregistered recon units
        condition = "Recondition" if recon else "Used"
        mileage = int(max(500, rng.normal(11500 * age + (2000 if not recon else 0), 5500 + 1500 * age)))
        if recon:
            mileage = int(rng.uniform(5000, 40000))
        trans = "Manual" if (m in ("Alto", "Panda", "Montero") and rng.random() < 0.45) else "Automatic"
        owners = 0 if recon else int(np.clip(rng.poisson(0.6 + age / 7) + 1, 1, 5))
        district = rng.choice(list(CITY), p=list(CITY.values()))
        seller = rng.choice(["Dealer", "Owner"], p=[0.45, 0.55])
        on_lease = int(rng.random() < 0.18 and age < 9)
        rows.append([brand, m, fuel, cc, body, yom, age, mileage, trans, condition, owners, district, seller,
                     on_lease, base, dep])
    df = pd.DataFrame(rows, columns=["brand", "model", "fuel_type", "engine_cc", "body_type", "year_of_manufacture",
                                     "vehicle_age", "mileage_km", "transmission", "condition", "previous_owners",
                                     "district", "seller_type", "on_lease", "_base", "_dep"])
    age = df.vehicle_age.values
    km = df.mileage_km.values
    price = (df._base.values * np.exp(-df._dep.values * (age - 7))
             * np.exp(-0.0000028 * np.maximum(km - 12000 * age, -40000))   # above-average mileage penalty
             * np.where(df.condition == "Recondition", 1.18, 1.0)
             * np.where((df.year_of_manufacture >= 2015) & (df.year_of_manufacture <= 2019), 1.08, 1.0)  # import-ban scarcity
             * np.where(df.transmission == "Manual", 0.93, 1.0)
             * (1 - 0.025 * np.maximum(df.previous_owners.values - 1, 0))
             * np.where(df.district == "Colombo", 1.03, 1.0)
             * np.where(df.seller_type == "Dealer", 1.04, 1.0)
             * np.where(df.on_lease == 1, 0.97, 1.0)
             * np.where((df.fuel_type == "Hybrid") & (age > 10), 0.90, 1.0)   # old hybrid battery risk
             * rng.lognormal(0, 0.07, len(df)))
    df["price_lkr_mn"] = np.round(price, 2)
    return df.drop(columns=["_base", "_dep"])


# --------------------------------------------------------------------------------------
# 3. COLOMBO DAILY WEATHER (synthetic fallback; same column names as Open-Meteo)
# --------------------------------------------------------------------------------------
def make_weather(start="2015-01-01", end="2025-12-31"):
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    month_rain = {1: 60, 2: 70, 3: 130, 4: 250, 5: 390, 6: 190, 7: 135, 8: 110, 9: 225, 10: 350, 11: 320, 12: 150}
    m = dates.month.values
    mean_rain = np.array([month_rain[k] for k in m]) / dates.days_in_month.values
    p_wet = np.clip(0.18 + mean_rain / 18, 0.15, 0.85)
    wet = rng.random(n) < p_wet
    rain = np.where(wet, rng.gamma(0.9, mean_rain / p_wet / 0.9), 0.0)
    doy = dates.dayofyear.values
    seasonal = 27.4 + 0.75 * np.sin(2 * np.pi * (doy - 30) / 365.25) + 0.35 * np.sin(4 * np.pi * (doy - 60) / 365.25)
    trend = 0.025 * (dates.year.values - 2015)
    enso = 0.35 * np.sin(2 * np.pi * np.arange(n) / (365.25 * 3.7))       # slow multi-year cycle (ENSO-like)
    anom = np.zeros(n)
    eps = rng.normal(0, 0.38, n)
    for t in range(1, n):
        anom[t] = 0.72 * anom[t - 1] - 0.10 * (anom[t - 2] if t > 1 else 0) + eps[t]
    temp = seasonal + trend + enso + anom - 0.018 * np.minimum(rain, 80)
    sw = np.isin(m, [5, 6, 7, 8, 9])
    wind = np.clip(rng.normal(np.where(sw, 24, 15), 4, n) + 0.08 * np.minimum(rain, 60), 5, None)
    rad = np.clip(rng.normal(21, 2.5, n) - 0.10 * np.minimum(rain, 70), 6, 28)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "temperature_2m_mean": temp.round(1),
        "temperature_2m_max": (temp + rng.normal(3.6, 0.6, n) - 0.01 * np.minimum(rain, 80)).round(1),
        "temperature_2m_min": (temp - rng.normal(2.9, 0.5, n)).round(1),
        "precipitation_sum": rain.round(1),
        "wind_speed_10m_max": wind.round(1),
        "shortwave_radiation_sum": rad.round(2),
    })


if __name__ == "__main__":
    c = make_churn(); c.to_csv(OUT / "sl_telecom_churn.csv", index=False)
    v = make_vehicles(); v.to_csv(OUT / "sl_vehicle_prices.csv", index=False)
    w = make_weather(); w.to_csv(OUT / "colombo_weather_daily.csv", index=False)
    print("churn", c.shape, "churn rate", round(c.churn.mean(), 3))
    print("vehicles", v.shape, v.price_lkr_mn.describe().round(2).to_dict())
    print("weather", w.shape, w.temperature_2m_mean.describe().round(2).to_dict())
