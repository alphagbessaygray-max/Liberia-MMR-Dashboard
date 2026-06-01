"""
Maternal Mortality Rate (MMR) Forecasting Model - Liberia (2000-2021)
=====================================================================
This script builds a full ML pipeline for forecasting MMR, including:
  - Feature engineering (lags, rates-of-change, composite index)
  - Time-aware cross-validation (no data leakage)
  - Three models: Ridge regression, Random Forest, Gradient Boosting
  - Policy scenario simulation
  - Forecast to 2027

Usage:
    python mmr_forecast_liberia.py
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

DATA_PATH = Path('liberia_health_dataset_cleaned.csv')
OUTPUT_PLOT = Path('mmr_forecast_liberia.png')


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {path.resolve()}')
    df = pd.read_csv(path)
    return df.sort_values('Year').reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    fe = df.copy()
    fe['SBA_lag1'] = fe['SBA_pct'].shift(1)
    fe['CHE_lag1'] = fe['CHE_GDP'].shift(1)
    fe['Malaria_lag1'] = fe['Malaria_Inc'].shift(1)
    fe['SBA_delta'] = fe['SBA_pct'].diff()
    fe['CHE_delta'] = fe['CHE_GDP'].diff()
    fe['Malaria_delta'] = fe['Malaria_Inc'].diff()
    fe['MMR_lag1'] = fe['MMR'].shift(1)
    fe['access_index'] = (
        (fe['SBA_pct'] - fe['SBA_pct'].min()) /
        (fe['SBA_pct'].max() - fe['SBA_pct'].min()) * 0.6 +
        (fe['CHE_GDP'] - fe['CHE_GDP'].min()) /
        (fe['CHE_GDP'].max() - fe['CHE_GDP'].min()) * 0.4
    )
    fe['time_idx'] = fe['Year'] - fe['Year'].min()
    return fe


def forecast_scenario(last_row: dict,
                      final_ridge,
                      final_rf,
                      final_gb,
                      scaler: StandardScaler,
                      df: pd.DataFrame,
                      FEATURES: list,
                      n_years: int = 6,
                      sba_growth: float = 1.0,
                      che_growth: float = 0.3,
                      malaria_reduction: float = 9.0,
                      label: str = 'Baseline') -> pd.DataFrame:
    rows = []
    last = last_row.copy()

    for i in range(1, n_years + 1):
        yr = 2021 + i
        sba = min(100, last['SBA_pct'] + sba_growth)
        che = last['CHE_GDP'] + che_growth
        malaria = max(100, last['Malaria_Inc'] - malaria_reduction)
        row = {
            'SBA_pct': sba,
            'SBA_lag1': last['SBA_pct'],
            'SBA_delta': sba - last['SBA_pct'],
            'CHE_GDP': che,
            'CHE_lag1': last['CHE_GDP'],
            'CHE_delta': che - last['CHE_GDP'],
            'Malaria_Inc': malaria,
            'Malaria_lag1': last['Malaria_Inc'],
            'Malaria_delta': malaria - last['Malaria_Inc'],
            'MMR_lag1': last['MMR'],
            'access_index': (
                (sba - df.SBA_pct.min()) / (df.SBA_pct.max() - df.SBA_pct.min()) * 0.6 +
                (che - df.CHE_GDP.min()) / (df.CHE_GDP.max() - df.CHE_GDP.min()) * 0.4
            ),
            'time_idx': yr - df.Year.min(),
        }
        row_df = pd.DataFrame([row])[FEATURES]
        row_sc = scaler.transform(row_df)
        pred_r = final_ridge.predict(row_sc)[0]
        pred_rf = final_rf.predict(row_df)[0]
        pred_gb = final_gb.predict(row_df)[0]
        pred = max(300, (pred_r + pred_rf + pred_gb) / 3)
        row['MMR'] = pred
        rows.append({'Year': yr, 'MMR_pred': pred, 'scenario': label})
        last = row

    return pd.DataFrame(rows)


def main():
    df = load_data()
    print('=' * 60)
    print('MATERNAL MORTALITY FORECASTING - LIBERIA')
    print('=' * 60)
    print(f"\nDataset: {len(df)} years ({df.Year.min()}-{df.Year.max()})")
    print(f"MMR range: {df.MMR.max()} (2000) -> {df.MMR.min()} (2021)")
    print(f"Total reduction: {df.MMR.max() - df.MMR.min()} deaths / 100k live births")

    fe_df = engineer_features(df)
    fe_df = fe_df.dropna().reset_index(drop=True)

    FEATURES = [
        'SBA_pct', 'SBA_lag1', 'SBA_delta',
        'CHE_GDP', 'CHE_lag1', 'CHE_delta',
        'Malaria_Inc', 'Malaria_lag1', 'Malaria_delta',
        'MMR_lag1', 'access_index', 'time_idx',
    ]
    TARGET = 'MMR'
    X = fe_df[FEATURES]
    y = fe_df[TARGET]

    print(f"\n{'-' * 60}")
    print('FEATURE ENGINEERING SUMMARY')
    print(f"{'-' * 60}")
    print(f"Engineered features: {len(FEATURES)}")
    print(f"Training rows available (after lags): {len(X)}")
    print('\nFeature groups:')
    print('  [Policy levers]  SBA_pct, CHE_GDP, Malaria_Inc')
    print('  [Lagged (t-1)]   SBA_lag1, CHE_lag1, Malaria_lag1')
    print('  [Momentum]       SBA_delta, CHE_delta, Malaria_delta')
    print('  [Autoregressive] MMR_lag1')
    print('  [Composite]      access_index')
    print('  [Time]           time_idx')

    print(f"\n{'-' * 60}")
    print('TIME-SERIES CROSS-VALIDATION (no data leakage)')
    print(f"{'-' * 60}")

    tscv = TimeSeriesSplit(n_splits=5, test_size=3)
    models = {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42),
    }
    scaler = StandardScaler()
    cv_results = {name: {'mae': [], 'rmse': [], 'r2': []} for name in models}

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        for name, model in models.items():
            if name == 'Ridge Regression':
                model.fit(X_train_sc, y_train)
                preds = model.predict(X_test_sc)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            cv_results[name]['mae'].append(mean_absolute_error(y_test, preds))
            cv_results[name]['rmse'].append(np.sqrt(mean_squared_error(y_test, preds)))
            cv_results[name]['r2'].append(r2_score(y_test, preds))

    print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print('-' * 50)
    best_model_name = None
    best_mae = float('inf')
    for name, res in cv_results.items():
        mae = np.mean(res['mae'])
        rmse = np.mean(res['rmse'])
        r2 = np.mean(res['r2'])
        flag = ' <- best' if mae < best_mae else ''
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
        print(f"{name:<22} {mae:>8.1f} {rmse:>8.1f} {r2:>8.3f}{flag}")

    print(f"\n{'-' * 60}")
    print(f'FINAL MODEL: {best_model_name} (trained on full dataset)')
    print(f"{'-' * 60}")

    X_sc = scaler.fit_transform(X)
    final_ridge = Ridge(alpha=10.0).fit(X_sc, y)
    final_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    final_gb = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42).fit(X, y)

    ridge_fitted = final_ridge.predict(X_sc)
    rf_fitted = final_rf.predict(X)
    gb_fitted = final_gb.predict(X)
    ensemble_fit = (ridge_fitted + rf_fitted + gb_fitted) / 3

    print(f"\nIn-sample MAE (ensemble): {mean_absolute_error(y, ensemble_fit):.1f}")
    print(f"In-sample R2  (ensemble): {r2_score(y, ensemble_fit):.4f}")

    feat_imp = pd.Series(final_rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print('\nTop feature importances (Random Forest):')
    for feat, imp in feat_imp.head(6).items():
        bar = '█' * int(imp * 40)
        print(f'  {feat:<20} {imp:.3f}  {bar}')

    scenarios = {
        'Baseline (status quo)': dict(sba_growth=1.0, che_growth=0.3, malaria_reduction=9.0),
        'Accelerated (SBA->95%)': dict(sba_growth=2.5, che_growth=0.6, malaria_reduction=12.0),
        'Low investment': dict(sba_growth=0.5, che_growth=0.1, malaria_reduction=4.0),
    }

    print(f"\n{'-' * 60}")
    print('FORECAST 2022-2027 - SCENARIO ANALYSIS')
    print(f"{'-' * 60}")
    forecast_dfs = []
    for label, params in scenarios.items():
        fcast = forecast_scenario(
            last_row=fe_df.iloc[-1].to_dict(),
            final_ridge=final_ridge,
            final_rf=final_rf,
            final_gb=final_gb,
            scaler=scaler,
            df=df,
            FEATURES=FEATURES,
            label=label,
            **params,
        )
        forecast_dfs.append(fcast)
        print(f"\n{label}")
        for _, row in fcast.iterrows():
            print(f"  {int(row.Year)}: {row.MMR_pred:.0f}")

    all_forecasts = pd.concat(forecast_dfs)

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#F8F7F4')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    colors = {
        'Baseline (status quo)': '#3B8BD4',
        'Accelerated (SBA->95%)': '#1D9E75',
        'Low investment': '#D85A30',
    }

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#F8F7F4')
    ax1.plot(df.Year, df.MMR, 'o-', color='#2C2C2A', linewidth=2, markersize=5, label='Actual MMR', zorder=5)
    ax1.plot(fe_df['Year'], ensemble_fit, '--', color='#888780', linewidth=1.5, label='Model fit (ensemble)', zorder=4)

    for label, grp in all_forecasts.groupby('scenario'):
        bridge_x = [df.Year.iloc[-1], grp.Year.iloc[0]]
        bridge_y = [df.MMR.iloc[-1], grp.MMR_pred.iloc[0]]
        ax1.plot(bridge_x, bridge_y, '-', color=colors[label], linewidth=1.5, alpha=0.5)
        ax1.plot(grp.Year, grp.MMR_pred, 'o-', color=colors[label], linewidth=2, markersize=6, label=label)

    ax1.axvline(df.Year.iloc[-1], color='#73726c', linewidth=1, linestyle=':', alpha=0.7)
    ax1.text(df.Year.iloc[-1] + 0.2, ax1.get_ylim()[1] * 0.95, 'Forecast ->', fontsize=10, color='#73726c')
    ax1.set_title('Maternal Mortality Rate - Liberia (2000-2027)', fontsize=13, fontweight='normal', pad=10)
    ax1.set_ylabel('MMR (deaths per 100k live births)', fontsize=10)
    ax1.set_xlabel('Year', fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax1.spines[['top','right']].set_visible(False)
    for sp in ['left', 'bottom']:
        ax1.spines[sp].set_linewidth(0.5)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#F8F7F4')
    feat_colors = ['#534AB7' if 'lag' in f or 'delta' in f else '#1D9E75' if 'access' in f or 'time' in f else '#3B8BD4' for f in feat_imp.index]
    feat_imp_plot = feat_imp.head(8)
    ax2.barh(feat_imp_plot.index[::-1], feat_imp_plot.values[::-1], color=feat_colors[:8][::-1], height=0.6)
    ax2.set_title('Feature Importance (Random Forest)', fontsize=11, fontweight='normal')
    ax2.set_xlabel('Importance score', fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    for sp in ['left', 'bottom']:
        ax2.spines[sp].set_linewidth(0.5)
    ax2.grid(axis='x', alpha=0.3, linewidth=0.5)
    ax2.tick_params(labelsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#F8F7F4')
    model_names = list(cv_results.keys())
    mae_means = [np.mean(cv_results[n]['mae']) for n in model_names]
    mae_stds = [np.std(cv_results[n]['mae']) for n in model_names]
    bar_colors = ['#534AB7', '#1D9E75', '#3B8BD4']
    short_names = ['Ridge', 'Random Forest', 'Gradient Boosting']
    bars2 = ax3.bar(short_names, mae_means, yerr=mae_stds, capsize=5, color=bar_colors, width=0.5, error_kw={'linewidth': 1})
    ax3.set_title('Cross-Validation MAE (5-fold time-series)', fontsize=11, fontweight='normal')
    ax3.set_ylabel('Mean Absolute Error', fontsize=9)
    ax3.spines[['top', 'right']].set_visible(False)
    for sp in ['left', 'bottom']:
        ax3.spines[sp].set_linewidth(0.5)
    ax3.grid(axis='y', alpha=0.3, linewidth=0.5)
    for bar, val in zip(bars2, mae_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    ax3.tick_params(labelsize=9)

    plt.suptitle('Liberia MMR Forecasting Model - Full Analysis', fontsize=14, fontweight='normal', y=1.01, color='#2C2C2A')
    fig.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight', facecolor='#F8F7F4')
    print(f"\nPlot saved: {OUTPUT_PLOT}")

    print(f"\n{'=' * 60}")
    print('POLICY SIMULATION RESULTS (2027 projections)')
    print(f"{'=' * 60}")
    for label, grp in all_forecasts.groupby('scenario'):
        mmr_2027 = grp[grp.Year == 2027].MMR_pred.values[0]
        mmr_2021 = df.MMR.iloc[-1]
        reduction = mmr_2021 - mmr_2027
        print(f"\n{label}")
        print(f"  Projected MMR 2027 : {mmr_2027:.0f}")
        print(f"  Reduction from 2021: {reduction:.0f} ({reduction/mmr_2021*100:.1f}%)")

    print(f"\n{'-' * 60}")
    print('KEY INSIGHT: Every +1% SBA coverage correlates with')
    coef_df = pd.Series(final_ridge.coef_, index=FEATURES)
    sba_coef = coef_df['SBA_pct']
    print(f"  ~{abs(sba_coef):.0f} fewer maternal deaths per 100k live births")
    print(f"  (Ridge coeff = {sba_coef:.1f}, scaled features)")
    print('-' * 60)


if __name__ == '__main__':
    main()
