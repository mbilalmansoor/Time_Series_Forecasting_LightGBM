# Advanced Time-Series Recursive Forecasting Engine

An enterprise-grade time-series forecasting pipeline designed to process large volumes of transactional records, engineer robust temporal features, and generate multi-step recursive forecasts with predictive uncertainty.

## 📬 Technology Stack

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-green?logo=scikit-learn&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# 📌 Overview

This project combines an efficient preprocessing stage with a recursive forecasting engine to transform raw transactional data into long-horizon time-series forecasts. The workflow is designed to efficiently process large datasets while producing both point forecasts and prediction intervals.

## Pipeline Architecture

### 1. Streaming Data Preprocessing

Rather than loading entire raw datasets into memory, the preprocessing stage streams input files sequentially to minimize memory consumption.

Key capabilities include:

- Recursive directory traversal
- Case-insensitive discovery of text files
- Row-by-row streaming
- Removal of invalid control characters
- Parsing of relevant fields
- Daily aggregation into a structured dataset ready for modeling

This approach enables processing of very large collections of raw files while maintaining a small memory footprint.

---

### 2. Feature Engineering

The forecasting pipeline constructs a rich set of temporal features, including:

- Calendar-based features
- Lag features
- Rolling statistics
- Exponentially weighted moving averages (EWMA)
- Seasonal Fourier components
- Interaction features
- Cyclical encodings for periodic behavior

These engineered features allow the model to capture recurring seasonal patterns and temporal dependencies.

---

### 3. Recursive Multi-Step Forecasting

The forecasting engine uses recursive prediction to extend forecasts over arbitrary horizons.

For each future time step:

1. Generate features using the most recent available history.
2. Predict the next observation.
3. Append the prediction to the historical sequence.
4. Recompute lag and rolling features.
5. Repeat until the desired forecast horizon is reached.

This recursive strategy preserves temporal consistency throughout the forecasting process.

---

### 4. Quantile Regression

Instead of producing only a single point estimate, the pipeline trains multiple LightGBM regression models to estimate conditional quantiles.

The generated forecasts include:

| Quantile | Interpretation |
|----------|----------------|
| **q0.1** | Conservative lower bound |
| **q0.5** | Median forecast |
| **q0.9** | Upper prediction bound |

A separate regression model is also trained to estimate the conditional mean.

Together, these models provide both expected values and predictive uncertainty.

---

# 📂 Repository Structure

```text
time-series-forecasting/
│
├── .gitignore
├── main.py
├── pyproject.toml
├── README.md
└── LICENSE
```

---

# 🚀 Installation

Create the virtual environment and install all dependencies:

```bash
uv sync
```

Run the forecasting pipeline:

```bash
uv run main.py
```

---

# ✨ Key Features

- Memory-efficient streaming preprocessing
- Automatic temporal feature engineering
- Recursive multi-step forecasting
- Quantile regression for uncertainty estimation
- LightGBM-based gradient boosting models
- Daily time-series aggregation
- Modular and portable project structure
- Cross-platform execution (Windows, Linux, and macOS)

---

# 📈 Forecast Output

The forecasting engine generates:

- Point forecasts (mean prediction)
- Median forecasts (50th percentile)
- Lower confidence estimates (10th percentile)
- Upper confidence estimates (90th percentile)
- Recursive future predictions over configurable forecast horizons

These outputs can be visualized alongside historical observations to monitor long-term trends and forecast uncertainty.
