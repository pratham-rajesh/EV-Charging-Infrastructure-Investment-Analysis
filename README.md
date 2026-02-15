# EV Charging Infrastructure Investment Analysis

A machine learning pipeline that identifies optimal California census tracts for electric vehicle charging infrastructure investment by integrating EV sales, demographic, and charging station data to classify tracts into three investment tiers.

---

## Problem Statement

As EV adoption accelerates across California, charging infrastructure deployment lags behind in many regions. This project answers a core business question: **which census tracts represent the strongest opportunities for new EV charging investment?**

A tract is considered high-priority when EV demand is strong, existing charger supply is low, and local socioeconomic and housing context supports future growth.

## Approach

The pipeline follows a four-stage progressive data amalgamation strategy, where each stage adds richer features before modeling:

| Stage | Data Added | Feature Count |
|-------|-----------|---------------|
| 1 — Base | ACS 2022 census tract demographics | 9 |
| 2 — + Demand | County-level EV/ZEV sales (2008–2025) | 13 |
| 3 — + Supply | Active EV charging station infrastructure | 18 |
| 4 — + Latent | Engineered supply-demand ratio features | 26 |

Models are trained and evaluated at every stage to measure the marginal lift from each data source.

## Data Sources

| Dataset | Records | Granularity | Description |
|---------|---------|-------------|-------------|
| ACS 2022 Census Tracts | 9,129 | Tract-level | Population, income, households, vehicle ownership, housing units, commuting |
| ZEV Sales (CVRP) | 72,384 | County × Quarter | EV sales by county, fuel type, make, and model |
| Alt Fuel Stations (AFDC) | 84,681 | Station-level | Charging station locations, charger types (Level 2, DC Fast), status |

All three datasets are merged at the tract level via county-key joins, producing a final modeling frame of **9,129 tracts × 37 columns**.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Data Ingestion → 3 raw datasets (ACS tracts, EV sales, stations)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Preprocessing → cleaning, county mapping, county-level rollups    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Progressive Amalgamation → 4 feature stages via left joins        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Latent Feature Engineering → 8 derived ratio/gap variables        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌──────────────────┬───────────┴───────────┬──────────────────────────┐
│   Clustering     │   Classification      │   Regression             │
│   KMeans (2-stage│   5 models + SMOTE    │   7 models               │
│   golden cluster)│   → 3-class labels    │   → EV-ready density     │
└──────────────────┴───────────────────────┴──────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Explainability → SHAP TreeExplainer + Gini feature importances    │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Engineering

Eight latent variables were engineered to capture supply-demand dynamics:

| Feature | Formula |
|---------|---------|
| `ev_per_1000_households` | EV vehicles / households × 1000 |
| `stations_per_1000_households` | Charging stations / households × 1000 |
| `chargers_per_1000_households` | Total chargers / households × 1000 |
| `dc_fast_per_1000_evs` | DC fast chargers / EV vehicles × 1000 |
| `workers_per_household` | Commuting workers / households |
| `low_car_household_share` | Zero-vehicle households / total households |
| `multi_unit_housing_share` | Multi-unit housing / total housing units |
| `supply_demand_gap` | Standardized EV demand − charger supply |

## Clustering — Two-Stage Golden Cluster

A distinctive two-stage approach isolates the highest-opportunity tracts:

1. **Outlier removal** — Isolation Forest flags 1,659 anomalous tracts from the training set
2. **Stage 1** — KMeans (k=3) segments 5,644 remaining tracts into *least*, *more*, and *most desirable* clusters, ranked by a composite golden score (`income + multi-unit housing + commuter density − DC fast coverage`)
3. **Stage 2** — KMeans (k=2) subdivides the *most desirable* cluster to extract a refined **golden cluster of 812 tracts** representing the top investment opportunities

## Classification Results

Five classifiers trained with SMOTE oversampling across all four feature stages:

| Model | Accuracy | F1 (macro) | AUC (macro) |
|-------|----------|------------|-------------|
| **Gradient Boosting** | **0.9945** | **0.9946** | **0.9998** |
| Random Forest | 0.9934 | 0.9936 | 0.9999 |
| Logistic Regression | 0.9825 | 0.9815 | 0.9968 |
| SVC | 0.9715 | 0.9721 | 0.9991 |
| KNN | 0.9381 | 0.9424 | 0.9875 |

*Metrics shown for the final feature stage (Stage 4). 3-fold cross-validation confirmed generalization (best CV F1-macro: 0.9894).*

## Regression Results

Seven regressors predict log-transformed EV-ready density:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Random Forest** | **0.2099** | **0.2853** | **0.8811** |
| Gradient Boosting | 0.2249 | 0.3018 | 0.8669 |
| KNN | 0.2585 | 0.3558 | 0.8150 |
| Ridge | 0.2706 | 0.4762 | 0.6686 |
| Linear Regression | 0.2708 | 0.4763 | 0.6685 |

## Explainability

SHAP TreeExplainer identifies the most influential features driving investment desirability:

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `median_household_income` | 0.2284 |
| 2 | `total_level2_chargers` | — |
| 3 | `unique_models` | — |
| 4 | `total_stations` | — |
| 5 | `unique_makes` | — |

Gini feature importances confirm `median_household_income` as the dominant predictor (importance = 0.367).

## Tech Stack

- **Language:** Python 3
- **Data:** pandas, NumPy, openpyxl, gdown
- **ML:** scikit-learn, imbalanced-learn (SMOTE)
- **Clustering:** KMeans, Isolation Forest
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn, ydata-profiling
- **Environment:** Google Colab, Google Drive (model caching)

## Repository Structure

```
├── ML_Numeric_Project_EV_Charging.ipynb   # Full pipeline notebook
├── README.md                              # This file
├── cache/                                 # Cached raw datasets
│   ├── dataset1.xlsx                      # ZEV sales data
│   ├── dataset2.csv                       # ACS census tract data
│   └── dataset3.csv                       # Alt fuel station data
├── saved_models/                          # Serialized model artifacts (.joblib)
└── outputs/                               # Final predictions and exports
```

## Key Findings

- **Median household income** is the single strongest predictor of investment desirability, followed by existing Level 2 charger density
- The **supply-demand gap** latent variable provides significant marginal lift in both clustering quality and classification accuracy
- The two-stage golden cluster isolates **812 tracts** (8.9% of all California tracts) as top-tier investment targets, concentrated in urban cores with high EV adoption but low DC fast charger coverage
- Progressive feature amalgamation improves model performance at each stage, validating the multi-source integration approach

## License

This project was developed as part of an ML coursework assignment. See individual dataset licenses for data usage terms.
