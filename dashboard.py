import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from mmr_forecast_liberia import load_data, engineer_features, forecast_scenario

FEATURES = [
    'SBA_pct', 'SBA_lag1', 'SBA_delta',
    'CHE_GDP', 'CHE_lag1', 'CHE_delta',
    'Malaria_Inc', 'Malaria_lag1', 'Malaria_delta',
    'MMR_lag1', 'access_index', 'time_idx',
]
TARGET = 'MMR'


@st.cache_data
def prepare_data():
    df = load_data()
    fe_df = engineer_features(df).dropna().reset_index(drop=True)
    X = fe_df[FEATURES]
    y = fe_df[TARGET]
    return df, fe_df, X, y


@st.cache_data
def train_models(X, y):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    final_ridge = Ridge(alpha=10.0).fit(X_sc, y)
    final_rf = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(X, y)
    final_gb = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42).fit(X, y)
    return scaler, final_ridge, final_rf, final_gb


@st.cache_data
def compute_cv_results(X, y):
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

    summary = []
    for name, res in cv_results.items():
        summary.append({
            'model': name,
            'mae': np.mean(res['mae']),
            'rmse': np.mean(res['rmse']),
            'r2': np.mean(res['r2']),
        })
    return pd.DataFrame(summary)


def render_chart(df, fe_df, ensemble_fit, forecasts):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.Year,
        y=df.MMR,
        mode='lines+markers',
        name='Actual MMR',
        line=dict(color='#2C2C2A', width=3),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=fe_df.Year,
        y=ensemble_fit,
        mode='lines',
        name='Ensemble fit',
        line=dict(color='#888780', dash='dash', width=2),
    ))

    scenario_colors = {
        'Baseline (status quo)': '#3B8BD4',
        'Accelerated (SBA -> 95%)': '#1D9E75',
        'Low investment': '#D85A30',
        'Custom scenario': '#FF6B9D',
    }

    for label, grp in forecasts.groupby('scenario'):
        color = scenario_colors.get(label, '#666666')
        
        # Add bridge line from last historical point to first forecast
        bridge_x = [df.Year.iloc[-1], grp.Year.iloc[0]]
        bridge_y = [df.MMR.iloc[-1], grp.MMR_pred.iloc[0]]
        fig.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode='lines',
            name=f'{label} (bridge)',
            line=dict(color=color, width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))
        
        # Add main forecast trace
        fig.add_trace(go.Scatter(
            x=grp.Year,
            y=grp.MMR_pred,
            mode='lines+markers',
            name=label,
            line=dict(color=color, width=2.5),
            marker=dict(size=7, color=color),
        ))

    fig.update_layout(
        template='plotly_white',
        title=dict(text='Liberia MMR: Actual, Fit, and Forecast', font=dict(color='#3B8BD4', size=18)),
        xaxis_title='Year',
        yaxis_title='MMR (deaths per 100k live births)',
        legend_title='Series',
        font=dict(family='Arial', size=12, color='#000000'),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        xaxis=dict(tickfont=dict(color='#000000')),
        yaxis=dict(tickfont=dict(color='#000000')),
        hovermode='x unified',
        height=500,
    )
    return fig


def main():
    st.set_page_config(page_title='Liberia MMR Dashboard', layout='wide', page_icon='📊')
    st.markdown(
        """
        <style>
            html, body {
                background-color: #F8F8F8;
                color: #222222;
            }
            .main {
                background-color: #F8F8F8;
            }
            [data-testid="stAppViewContainer"] {
                background-color: #F8F8F8;
            }
            [data-testid="stSidebar"] {
                background-color: #F0F0F0;
            }
            /* Title styling - HIGHEST priority */
            h1, h1 * {
                color: #3B8BD4 !important;
            }
            /* Other headers */
            h2, h3, h4, h5, h6 {
                color: #3B8BD4 !important;
            }
            /* Text elements */
            p, label {
                color: #222222 !important;
            }
            /* Buttons */
            .stButton>button {
                background-color: #3B8BD4 !important;
                color: white !important;
                border-radius: 4px;
            }
            /* Metrics */
            .stMetric {
                background-color: #FFFFFF;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df, fe_df, X, y = prepare_data()
    scaler, final_ridge, final_rf, final_gb = train_models(X, y)
    cv_df = compute_cv_results(X, y)

    ridge_fitted = final_ridge.predict(scaler.transform(X))
    rf_fitted = final_rf.predict(X)
    gb_fitted = final_gb.predict(X)
    ensemble_fit = (ridge_fitted + rf_fitted + gb_fitted) / 3

    scenarios = {
        'Baseline (status quo)': dict(sba_growth=1.0, che_growth=0.3, malaria_reduction=9.0),
        'Accelerated (SBA -> 95%)': dict(sba_growth=2.5, che_growth=0.6, malaria_reduction=12.0),
        'Low investment': dict(sba_growth=0.5, che_growth=0.1, malaria_reduction=4.0),
    }

    st.markdown('<h1 style="color: #3B8BD4 !important; text-align: center;">Liberia Maternal Mortality Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('Explore historical MMR performance, model fit, and scenario forecasts with interactive controls.')

    with st.sidebar:
        st.header('Forecast controls')
        selected_scenario = st.selectbox('Choose scenario', list(scenarios.keys()))
        st.write('Or tweak a custom forecast:')
        custom_sba = st.slider('SBA growth per year (% points)', 0.0, 5.0, 1.0, 0.1)
        custom_che = st.slider('CHE growth per year (% GDP)', 0.0, 1.0, 0.3, 0.05)
        custom_malaria = st.slider('Malaria reduction per year', 0.0, 20.0, 9.0, 0.5)
        n_years = st.slider('Forecast horizon (years)', 1, 10, 6)
        st.write('---')
        st.write('Data source: Liberia health dataset (2000–2021)')

    scenario_params = scenarios[selected_scenario].copy()
    scenario_params.update({
        'sba_growth': custom_sba,
        'che_growth': custom_che,
        'malaria_reduction': custom_malaria,
        'n_years': n_years,
        'label': 'Custom scenario',
    })

    custom_forecast = forecast_scenario(
        last_row=fe_df.iloc[-1].to_dict(),
        final_ridge=final_ridge,
        final_rf=final_rf,
        final_gb=final_gb,
        scaler=scaler,
        df=df,
        FEATURES=FEATURES,
        **scenario_params,
    )

    built_forecasts = []
    for label, params in scenarios.items():
        built_forecasts.append(forecast_scenario(
            last_row=fe_df.iloc[-1].to_dict(),
            final_ridge=final_ridge,
            final_rf=final_rf,
            final_gb=final_gb,
            scaler=scaler,
            df=df,
            FEATURES=FEATURES,
            label=label,
            **params,
        ))
    built_forecasts = pd.concat(built_forecasts)
    all_forecasts = pd.concat([built_forecasts, custom_forecast])

    col1, col2, col3 = st.columns([1.4, 1.0, 1.0])
    col1.metric('Historical years', f"{df.Year.min()}–{df.Year.max()}")
    col2.metric('Current MMR (2021)', f"{int(df.MMR.iloc[-1])}")
    col3.metric('Data points', len(df))

    st.markdown('### Scenario forecast comparison')
    fig = render_chart(df, fe_df, ensemble_fit, all_forecasts)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Forecast table')
    st.dataframe(custom_forecast[['Year', 'MMR_pred']].rename(columns={'MMR_pred': 'Forecast MMR'}).style.format({'Forecast MMR': '{:.0f}'}), height=280)

    st.markdown('### Model validation')
    cv_col1, cv_col2 = st.columns(2)
    cv_col1.table(cv_df.style.format({'mae': '{:.1f}', 'rmse': '{:.1f}', 'r2': '{:.3f}'}))

    feature_importance = pd.Series(final_rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
    fig_imp = px.bar(
        feature_importance,
        orientation='h',
        labels={'index': 'Feature', 'value': 'Importance'},
        title='Random Forest Feature Importance',
        template='plotly_white',
    )
    fig_imp.update_layout(
        plot_bgcolor='#FFFFFF', 
        paper_bgcolor='#FFFFFF',
        title=dict(text='Random Forest Feature Importance', font=dict(color='#3B8BD4')),
        font=dict(color='#000000'),
        xaxis=dict(tickfont=dict(color='#000000')),
        yaxis=dict(tickfont=dict(color='#000000')),
    )
    cv_col2.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('### Custom scenario details')
    st.write(f'*Scenario: {selected_scenario} + custom adjustments*')
    st.write(f'+ SBA growth: {custom_sba:.1f}% points/year')
    st.write(f'+ CHE growth: {custom_che:.2f}% GDP/year')
    st.write(f'+ Malaria reduction: {custom_malaria:.1f} cases/year')
    st.write(f'+ Horizon: {n_years} years')


if __name__ == '__main__':
    main()
