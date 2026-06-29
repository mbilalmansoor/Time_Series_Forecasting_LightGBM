# Advanced Time-Series Recursive Forecasting Engine

An enterprise-grade time-series pipeline built to parse unstructured transactional logs recursively, engineer robust temporal variations, and deliver multi-step recursive quantile projections.

📬 **Stack Profile:**
![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-green?logo=scikit-learn&logoColor=white)
![uv](https://img.shields.io/badge/Package%20Manager-uv-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview & Pipeline Flow

The engine operates via a two-stage unified architecture: a high-speed streaming data preprocessor and an iterative forecasting pipeline that models both expected behavior (mean regression) and volatility boundaries (quantile regression).

### 1. High-Speed Native Streaming Preprocessor
Instead of loading gigabytes of raw text data directly into system memory (which causes out-of-memory crashes), the pipeline implements a low-overhead file system stream. 
* It recursively scans target directories case-insensitively for nested raw text logs (`*.txt` or `*.TXT`).
* It streams data row-by-row, stripping DOS End-of-File markers (`\x1A`), clearing nested string mutations, and isolating the primary target features (`DATE` and `TARGET_VALUE`).
* It aggregates matches instantly into a structured daily master ledger (`aggregated_target_data.csv`), bypassing large memory structures.

### 2. Multi-Quantile Recursive Forecasting Engine
Time-series forecasting over extended horizons suffers from variance accumulation. To combat this, the script uses a recursive feedback strategy combined with gradient-boosted decision trees (GBDT):
* **Feature Propagation**: The engine extracts calendar cycles, fourier harmonics for weekly rhythm tracking, rolling averages, exponentially weighted moving averages (EWMA), and lag configurations.
* **Quantile Volatility Horizons**: Rather than predicting a single arbitrary value, the engine trains three independent quantile regressor models alongside a core conditional mean model:
  * **$q_{0.1}$ (10th Percentile)**: Represents the conservative lower-bound floor.
  * **$q_{0.5}$ (50th Percentile / Median)**: Represents the highly stable robust midpoint trend.
  * **$q_{0.9}$ (90th Percentile)**: Represents the aggressive upper-bound spike ceiling.
* **Iterative Feedback Loop**: For every projected day up to the forecast target date, the engine calculates features for the new step, generates predictions, and **pipes the $q_{0.5}$ median prediction back into the history array**. The subsequent step treats that prediction as real historical data to compute its own rolling metrics, safely preserving time dependency.

---

## 📂 Repository Structure

```text
time-series-forecasting/
│
├── .gitignore                      # Configured to ignore all tracking datasets (*.csv) & local environments
├── main.py                         # Primary engine script containing preprocessing and modeling
├── pyproject.toml                  # Project metadata and lock dependencies for uv setup
└── README.md                       # System documentation


### 🚀 Setup & Execution Guide

**Initialize the virtual environment and sync dependencies automatically

```bash
uv sync

uv run main.py
