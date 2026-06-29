# Advanced Time-Series Recursive Forecasting Engine

A high-performance machine learning pipeline built with **Python**, **Pathlib**, and **LightGBM** designed to aggregate raw text transaction logs recursively, construct deep temporal features, and perform multi-step recursive quantile time-series forecasting.

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
├── .venv/                          # Project virtual environment
├── .gitignore                      # Configured to ignore all tracking datasets (*.csv)
├── main.py                         # Primary engine script containing preprocessing and modeling
├── data/                           # Local runtime sandbox (Git ignored)
│   ├── raw_source/                 # Source directory for unprocessed logs
│   └── processed_sink/             # Target repository for master logs and projections
└── README.md                       # System documentation