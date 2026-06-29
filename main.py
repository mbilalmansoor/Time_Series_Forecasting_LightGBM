import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# ⚙️ USER-CONFIGURABLE VARIABLES (MAIN PANEL)
# =====================================================================
# Using Environment Variables with generic fallbacks so no local paths are leaked.
INPUT_DIR = os.environ.get("TS_INPUT_DIR", r"./data/raw_source")
OUTPUT_DIR = os.environ.get("TS_OUTPUT_DIR", r"./data/processed_sink")
FORECAST_END_DATE = os.environ.get("TS_FORECAST_END", "2026-12-31")

# Derived agnostic file-system path definitions
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "aggregated_target_data.csv")
OUTPUT_FORECAST_CSV = os.path.join(OUTPUT_DIR, "forecast_projections_2026.csv") 
ERROR_LOG = os.path.join(OUTPUT_DIR, "failed_records.log")

# Global feature engineering overrides
FLAGS = [3, 7, 14, 30]
LAGS = [3, 7, 14, 30]
WINDOWS = [3]


# =====================================================================
# 🛠️ NATIVE PYTHON STREAMING PREPROCESSING (DEEP DIRECTORY RESOLUTION)
# =====================================================================
def stream_txt_to_csv(input_dir, output_file, error_log_file):
    """
    Streams raw text data files, extracts DATE and TARGET columns,
    and streams them directly to an output CSV file.
    """
    print("🔹 Running native Python data streaming preprocessing...")
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: The path '{input_dir}' does not exist.")
        return

    # Scan case-insensitively for all text files within any subfolders
    all_files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"]
    total_files = len(all_files)
    
    if total_files == 0:
        print(f"⚠️ No source text logs found inside '{input_dir}' or its subfolders.")
        return

    failed_files = []
    processed_count = 0

    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write("DATE,TARGET_VALUE\n")
            
            for file_path in all_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                        for line in in_f:
                            clean = line.replace('\x1A', '').strip().replace('"', '')
                            if not clean:
                                continue
                            
                            parts = clean.split(',')
                            if len(parts) >= 5:
                                date = parts[0].strip()
                                target_val = parts[4].strip()
                                out_f.write(f"{date},{target_val}\n")
                                
                except Exception:
                    failed_files.append(str(file_path))
                
                processed_count += 1
                percent = int((processed_count / total_files) * 100)
                sys.stdout.write(f"\rProcessing [ {percent}% ] ({processed_count}/{total_files} files)")
                sys.stdout.flush()
                
        print("\n✅ Stream parsing finished successfully.")
        
        if failed_files:
            print(f"⚠️ Some records failed to convert. See {error_log_file}")
            with open(error_log_file, 'w', encoding='utf-8') as log_f:
                log_f.write("\n".join(failed_files))
        print(f"📁 Combined master ledger saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error creating output file pipeline: {e}")


# =====================================================================
# ⚙️ MODELING FEATURE FUNCTIONS
# =====================================================================
def create_features(df, target='TARGET_VALUE', lags=[3,7], windows=[7,14], fourier_harmonics=2):            
    df['DAY'] = df['DATE'].dt.day.astype(int)
    df['DAY_OF_MONTH'] = df['DATE'].dt.day.astype(int)
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear.astype(int)
    df['MONTH'] = df['DATE'].dt.month.astype(int)
    df['YEAR'] = df['DATE'].dt.year.astype(int)
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek.astype(int)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5,6]).astype(int)
    df['HALF_YEAR'] = ((df['MONTH'] - 1) // 6 + 1).astype(int)
    df['QUARTER'] = ((df['MONTH'] - 1) // 3 + 1).astype(int)        

    for lag in lags:
        df[f'{target}_lag_{lag}'] = df[target].shift(lag)
    for w in windows:
        df[f'{target}_roll_mean_{w}'] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f'{target}_roll_std_{w}'] = df[target].shift(1).rolling(w, min_periods=1).std()
        df[f'{target}_ewm_{w}'] = df[target].ewm(span=w, adjust=False).mean()
    for lag in lags:
        df[f'{target}_diff_{lag}'] = df[target].diff(lag)
    t_week = df['DAY_OF_WEEK'].values
    for k in range(1, fourier_harmonics + 1):
        df[f'sin_week_{k}'] = np.sin(2 * np.pi * k * t_week / 7)
        df[f'cos_week_{k}'] = np.cos(2 * np.pi * k * t_week / 7)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def create_time_features(df):
    df = df.copy()
    df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['QUARTER'] = df['DATE'].dt.quarter
    df['WEEK_OF_MONTH'] = df['DATE'].dt.day // 7 + 1
    df['DAY_OF_MONTH'] = df['DATE'].dt.day
    df['DAY_OF_YEAR'] = df['DATE'].dt.dayofyear
    df['QUAD'] = ((df['MONTH'] - 1) // 4) + 1      
    df['SIN_DAY_OF_WEEK'] = np.sin(2*np.pi*df['DAY_OF_WEEK']/7)
    df['COS_DAY_OF_WEEK'] = np.cos(2*np.pi*df['DAY_OF_WEEK']/7)
    df['SIN_MONTH'] = np.sin(2*np.pi*df['MONTH']/12)
    df['COS_MONTH'] = np.cos(2*np.pi*df['MONTH']/12)
    df['SIN_QUAD'] = np.sin(2 * np.pi * df['QUAD'] / 3)
    df['COS_QUAD'] = np.cos(2 * np.pi * df['QUAD'] / 3)
    df['MONTH_INDEX'] = (df['YEAR'] - df['YEAR'].min())*12 + df['MONTH']
    df['SIN_8M'] = np.sin(2 * np.pi * df['MONTH_INDEX'] / 8)
    df['COS_8M'] = np.cos(2 * np.pi * df['MONTH_INDEX'] / 8)
    df['IS_PEAK_8M'] = ((df['SIN_8M'] > 0.9) | (df['SIN_8M'] < -0.9)).astype(int)
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([5,6]).astype(int)
    df['IS_WEDNESDAY'] = (df['DAY_OF_WEEK'] == 2).astype(int)
    df['IS_MONTH_08'] = (df['MONTH'] == 8).astype(int)        
    return df

def add_lag_and_roll_features(df, target_col='TARGET_VALUE'):
    df = df.copy()
    for lag in LAGS:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    for w in WINDOWS:
        df[f'{target_col}_roll_mean_{w}'] = df[target_col].shift(1).rolling(window=w).mean()
        df[f'{target_col}_roll_std_{w}'] = df[target_col].shift(1).rolling(window=w).std()
    t = np.arange(len(df))
    for k in range(1,4):
        df[f'sin_year_{k}'] = np.sin(2*np.pi*k*t/365)
        df[f'cos_year_{k}'] = np.cos(2*np.pi*k*t/365)
    return df

def add_advanced_features(df):
    df = df.copy()
    df['WEEKEND_X_MONTH'] = df['IS_WEEKEND'] * df['MONTH']
    df['DAYOFWEEK_X_QUARTER'] = df['DAY_OF_WEEK'] * df['QUARTER']
    return df

def recursive_forecast(models_dict, mean_model, df_history, horizon=30, feature_cols=None):
    last_date = df_history['DATE'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
    hist = df_history.copy().reset_index(drop=True)
    future = pd.DataFrame({'DATE': future_dates, 'TARGET_VALUE': [np.nan]*horizon})
    combined = pd.concat([hist[['DATE','TARGET_VALUE']], future], ignore_index=True)
    out_rows = []
    for i in range(len(hist), len(combined)):
        df_slice = combined.iloc[:i].copy()
        df_feat = create_time_features(df_slice)
        df_feat = add_lag_and_roll_features(df_feat)
        df_feat = add_advanced_features(df_feat)
        row_X = df_feat.iloc[[-1]][feature_cols]
        preds = {f'pred_q_{q}': m.predict(row_X)[0] for q, m in models_dict.items()}
        preds['pred_mean'] = mean_model.predict(row_X)[0]
        combined.loc[i, 'TARGET_VALUE'] = preds.get('pred_q_0.5', preds['pred_mean'])
        out_rows.append({'DATE': combined.loc[i, 'DATE'], **preds})
    return pd.DataFrame(out_rows)


# =====================================================================
# 🚀 EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Run parsing pipeline
    stream_txt_to_csv(INPUT_DIR, OUTPUT_CSV, ERROR_LOG)
    
    # 2. Part 1 Processing & Classic Modeling
    if not os.path.exists(OUTPUT_CSV):
        print(f"❌ Aborting. Processed data tracking asset file {OUTPUT_CSV} does not exist.")
        sys.exit(1)
        
    df = pd.read_csv(OUTPUT_CSV)
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df['TARGET_VALUE'] = pd.to_numeric(df['TARGET_VALUE'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('DATE').reset_index(drop=True)
    df_grouped = df.groupby('DATE', as_index=False)['TARGET_VALUE'].sum()
    
    df_features = create_features(df_grouped.copy())
    
    target = 'TARGET_VALUE'
    X = df_features.drop(columns=['DATE', target])
    y = df_features[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l2': 0.1
    }
    
    print("\nTraining Phase 1 Baseline Engine Model...")
    bst = lgb.train(params, lgb_train, valid_sets=[lgb_test], num_boost_round=1000)
    
    # 3. Deep Feature Engineering & Recursive Quantile Pipeline
    df_fresh = pd.read_csv(OUTPUT_CSV)
    df_fresh['DATE'] = pd.to_datetime(df_fresh['DATE'], errors='coerce', format='%d/%m/%Y', dayfirst=True)
    df_fresh['TARGET_VALUE'] = (df_fresh['TARGET_VALUE'].astype(str).str.strip().str.replace(r'^-\s*', '', regex=True))
    df_fresh['TARGET_VALUE'] = pd.to_numeric(df_fresh['TARGET_VALUE'], errors='coerce').fillna(0).astype(int)
    df_fresh = df_fresh.groupby('DATE')['TARGET_VALUE'].sum().reset_index().sort_values('DATE').reset_index(drop=True)
    df_fresh = df_fresh.set_index('DATE').asfreq('D', fill_value=0).reset_index()
    
    df_feat = create_time_features(df_fresh)
    df_feat = add_lag_and_roll_features(df_feat)
    df_feat = add_advanced_features(df_feat)
    df_trainable = df_feat.dropna().reset_index(drop=True)
    
    drop_cols = ['DATE']
    feature_cols = [c for c in df_trainable.columns if c not in drop_cols + ['TARGET_VALUE']]
    X_full = df_trainable[feature_cols]
    y_full = df_trainable['TARGET_VALUE']
    
    # Quantile Models
    quantiles = [0.1, 0.5, 0.9]
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective='quantile', alpha=q, learning_rate=0.035, n_estimators=1000,
            num_leaves=160, max_depth=12, subsample=0.85, colsample_bytree=0.85,
            min_child_samples=15, reg_alpha=0.05, reg_lambda=0.15, random_state=42, verbosity=-1
        )
        m.fit(X_full, y_full)
        models[q] = m
        
    mean_model = lgb.LGBMRegressor(
        learning_rate=0.035, num_leaves=200, max_depth=12, n_estimators=1000,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=15,
        reg_alpha=0.05, reg_lambda=0.15, random_state=42, verbosity=-1
    )
    mean_model.fit(X_full, y_full)
    
    # 4. Run Forecasting
    last_date = df_fresh['DATE'].max()
    end_date = pd.Timestamp(FORECAST_END_DATE)
    HORIZON = (end_date - last_date).days
    
    if HORIZON <= 0:
        HORIZON = 30
        
    print(f"\n🔮 Projecting forecast for an extended horizon of {HORIZON} days...")
    forecast_df = recursive_forecast(models, mean_model, df_fresh, horizon=HORIZON, feature_cols=feature_cols)
    print(forecast_df.head())
    
    # 💾 SAVE FORECAST RESULTS TO CSV FILE
    try:
        forecast_df.to_csv(OUTPUT_FORECAST_CSV, index=False)
        print(f"📁 Forecast results saved successfully to: {OUTPUT_FORECAST_CSV}")
    except Exception as e:
        print(f"⚠️ Failed to save forecast CSV: {e}")
    
    # 5. Visualizing
    print("\n📊 Generating Plots...")
    hist_plot = df_fresh[['DATE', 'TARGET_VALUE']].copy()
    plt.figure(figsize=(14,6))
    plt.plot(hist_plot["DATE"], hist_plot["TARGET_VALUE"], label="Historical Actuals")
    plt.plot(forecast_df["DATE"], forecast_df["pred_q_0.5"], label="Forecast Median (q0.5)")
    plt.plot(forecast_df["DATE"], forecast_df["pred_mean"], linestyle="--", label="Forecast Mean")
    plt.fill_between(forecast_df["DATE"], forecast_df["pred_q_0.1"], forecast_df["pred_q_0.9"], alpha=0.3, label="Forecast Uncertainty Interval")
    plt.xlabel("Timeline Date")
    plt.ylabel("Value Metrics")
    plt.title("Time-Series Forecast Profile Matrix")
    plt.legend()
    plt.tight_layout()
    plt.show()