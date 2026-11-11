"""ThreatSense - Unified NIDS + DDoS Detection Dashboard."""
import os, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

ROOT = os.path.dirname(__file__)
NIDS_SRC = os.path.join(ROOT, 'nids', 'src')
if NIDS_SRC not in sys.path: sys.path.insert(0, NIDS_SRC)

from constants import SEVERITY_BY_CATEGORY
from download_data import load_dataset
from train import preprocess
from explain import Explainer
from ddos.predictor import artifacts_available, load_artifacts as load_ddos_artifacts, predict_dataframe

NIDS_MODELS = os.path.join(ROOT, 'nids', 'models')
st.set_page_config(page_title='ThreatSense', page_icon='🛡️', layout='wide')

@st.cache_resource
def load_nids_artifacts():
    return (
        joblib.load(os.path.join(NIDS_MODELS, 'model.joblib')),
        joblib.load(os.path.join(NIDS_MODELS, 'encoders.joblib')),
        joblib.load(os.path.join(NIDS_MODELS, 'feature_columns.joblib')),
        joblib.load(os.path.join(NIDS_MODELS, 'label_encoder.joblib')),
    )

@st.cache_resource
def load_nids_explainer(model, feature_columns):
    return Explainer(model, feature_columns)

@st.cache_data
def load_nids_stream():
    return load_dataset('NSL_KDD_Test.csv').sample(frac=1.0, random_state=42).reset_index(drop=True)

@st.cache_resource
def cached_ddos_artifacts():
    return load_ddos_artifacts()


DDOS_DATASET = os.path.join(
    ROOT,
    'data',
    'ddos',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
)


@st.cache_data
def load_ddos_stream():
    df = pd.read_csv(DDOS_DATASET)
    df.columns = df.columns.str.strip()
    return df.sample(frac=1.0, random_state=42).reset_index(drop=True)


if 'ddos_idx' not in st.session_state:
    st.session_state.ddos_idx = 0

if 'ddos_log' not in st.session_state:
    st.session_state.ddos_log = pd.DataFrame()

if 'ddos_running' not in st.session_state:
    st.session_state.ddos_running = False

if 'nids_idx' not in st.session_state: st.session_state.nids_idx = 0
if 'nids_log' not in st.session_state: st.session_state.nids_log = pd.DataFrame()
if 'nids_running' not in st.session_state: st.session_state.nids_running = False

st.title('🛡️ ThreatSense')
st.caption('Unified AI-powered cyber threat detection and monitoring system')

nids_tab, ddos_tab, about_tab = st.tabs(['🔍 NIDS Monitoring', '⚡ DDoS Detection', 'ℹ️ System Overview'])

with nids_tab:
    model, encoders, feature_columns, label_encoder = load_nids_artifacts()
    stream_df = load_nids_stream()
    st.subheader('Network Intrusion Detection')
    st.caption('NSL-KDD • Random Forest • Normal / DoS / Probe / R2L / U2R')
    with st.sidebar:
        st.header('NIDS Simulation Controls')
        rows_per_tick = st.slider('Connections per tick', 1, 10, 3)
        refresh_seconds = st.slider('Tick interval (seconds)', 1, 10, 2)
        a, b = st.columns(2)
        if a.button('▶ Start', use_container_width=True): st.session_state.nids_running = True
        if b.button('⏸ Pause', use_container_width=True): st.session_state.nids_running = False
        if st.button('🔄 Reset NIDS', use_container_width=True):
            st.session_state.nids_idx = 0; st.session_state.nids_log = pd.DataFrame(); st.session_state.nids_running = False; st.rerun()
    if st.session_state.nids_running:
        st_autorefresh(interval=refresh_seconds*1000, key='nids_tick')
        if st.session_state.nids_idx < len(stream_df):
            end = min(st.session_state.nids_idx + rows_per_tick, len(stream_df))
            batch = stream_df.iloc[st.session_state.nids_idx:end].copy()
            st.session_state.nids_idx = end
            X, _, _, _ = preprocess(batch, encoders=encoders, fit=False)
            preds = model.predict(X); probs = model.predict_proba(X)
            result = batch.reset_index(drop=True).copy()
            result['predicted_category'] = label_encoder.inverse_transform(preds)
            result['confidence'] = probs.max(axis=1)
            result['severity'] = [SEVERITY_BY_CATEGORY.get(x, 'medium') for x in result['predicted_category']]
            result['timestamp'] = pd.Timestamp.now().strftime('%H:%M:%S')
            result['row_id'] = range(len(st.session_state.nids_log), len(st.session_state.nids_log)+len(result))
            st.session_state.nids_log = pd.concat([st.session_state.nids_log, result], ignore_index=True)
    log = st.session_state.nids_log
    cols = st.columns(5)
    counts = log['predicted_category'].value_counts() if len(log) else pd.Series(dtype=int)
    for col, label, value in zip(cols, ['Processed','Normal','DoS','Probe','R2L/U2R'], [len(log), counts.get('Normal',0), counts.get('DoS',0), counts.get('Probe',0), counts.get('R2L',0)+counts.get('U2R',0)]): col.metric(label, int(value))
    if len(log):
        st.dataframe(log[['row_id','timestamp','protocol_type','service','flag','predicted_category','confidence','severity']].tail(50).iloc[::-1], use_container_width=True)
        alerts = log[log.predicted_category != 'Normal']
        if len(alerts):
            st.warning(f'{len(alerts)} intrusion alert(s) detected in processed traffic.')
        else: st.success('No intrusion alerts detected yet.')
    else: st.info('Press Start to begin NSL-KDD traffic simulation.')


with ddos_tab:
    st.subheader('DDoS Network Monitoring')
    st.caption(
        'CICIDS2017 • Hybrid Random Forest + XGBoost '
        '• Benign / DDoS'
    )

    if not artifacts_available():
        st.error('DDoS model artifacts are missing.')

    elif not os.path.exists(DDOS_DATASET):
        st.error(f'DDoS dataset not found: {DDOS_DATASET}')

    else:
        try:
            ddos_model, ddos_preprocessor = cached_ddos_artifacts()
            stream_df = load_ddos_stream()

            st.success(
                f'DDoS model ready • '
                f'{len(ddos_preprocessor["feature_columns"])} features'
            )

            st.caption(
                f'Loaded {len(stream_df):,} CICIDS2017 traffic records '
                'for offline monitoring simulation.'
            )

            c1, c2, c3 = st.columns(3)

            batch_size = c1.slider(
                'Records per tick', 1, 100, 10, key='ddos_batch'
            )

            refresh_seconds = c2.slider(
                'Refresh interval (seconds)',
                1, 10, 2, key='ddos_refresh'
            )

            c3.metric('Dataset records', f'{len(stream_df):,}')

            a, b, c = st.columns(3)

            if a.button('▶ Start DDoS', use_container_width=True):
                if st.session_state.ddos_idx >= len(stream_df):
                    st.session_state.ddos_idx = 0
                    st.session_state.ddos_log = pd.DataFrame()

                st.session_state.ddos_running = True

            if b.button('⏸ Pause DDoS', use_container_width=True):
                st.session_state.ddos_running = False

            if c.button('🔄 Reset DDoS', use_container_width=True):
                st.session_state.ddos_idx = 0
                st.session_state.ddos_log = pd.DataFrame()
                st.session_state.ddos_running = False
                st.rerun()

            if st.session_state.ddos_running:
                st_autorefresh(
                    interval=refresh_seconds * 1000,
                    key='ddos_tick'
                )

                if st.session_state.ddos_idx < len(stream_df):
                    start = st.session_state.ddos_idx
                    end = min(start + batch_size, len(stream_df))

                    batch = stream_df.iloc[start:end].copy()
                    st.session_state.ddos_idx = end

                    with st.spinner('Analysing network traffic...'):
                        predictions = predict_dataframe(
                            batch,
                            ddos_model,
                            ddos_preprocessor
                        )

                    result = batch.reset_index(drop=True).copy()
                    predictions = predictions.reset_index(drop=True)

                    result['prediction'] = predictions['prediction']
                    result['confidence'] = predictions['confidence']
                    result['benign_probability'] = predictions['benign_probability']
                    result['ddos_probability'] = predictions['ddos_probability']
                    result['timestamp'] = pd.Timestamp.now().strftime('%H:%M:%S')
                    result['row_id'] = range(start, end)

                    st.session_state.ddos_log = pd.concat(
                        [st.session_state.ddos_log, result],
                        ignore_index=True
                    ).tail(1000).reset_index(drop=True)

            log = st.session_state.ddos_log
            total = len(log)

            attacks = (
                int((log['prediction'] == 'DDoS Attack').sum())
                if total else 0
            )

            benign = total - attacks

            m1, m2, m3 = st.columns(3)
            m1.metric('Processed Traffic', total)
            m2.metric('DDoS Attacks', attacks)
            m3.metric('Benign Traffic', benign)

            if total:
                st.subheader('Traffic Detection Results')

                display_columns = [
                    'row_id', 'timestamp', 'prediction', 'confidence',
                    'benign_probability', 'ddos_probability'
                ]

                if 'Destination Port' in log.columns:
                    display_columns.insert(2, 'Destination Port')

                st.dataframe(
                    log[display_columns].tail(50).iloc[::-1],
                    use_container_width=True
                )

                alerts = log[log['prediction'] == 'DDoS Attack']

                if len(alerts):
                    st.warning(
                        f'{len(alerts)} potential DDoS attack record(s) detected.'
                    )
                else:
                    st.success('No DDoS attacks detected in processed traffic.')

                st.download_button(
                    'Download DDoS monitoring results',
                    log.to_csv(index=False).encode('utf-8'),
                    'ddos_monitoring_results.csv',
                    'text/csv'
                )

            else:
                st.info(
                    'Press Start DDoS to begin automatic traffic monitoring.'
                )

            if st.session_state.ddos_idx >= len(stream_df):
                st.session_state.ddos_running = False
                st.success('DDoS dataset replay completed.')

        except Exception as e:
            st.error(f'DDoS monitoring error: {e}')


with about_tab:
    st.markdown('''### ThreatSense Architecture

| Module | Dataset | Model | Output |
|---|---|---|---|
| NIDS | NSL-KDD | Random Forest | Normal, DoS, Probe, R2L, U2R |
| DDoS | CICIDS2017 | Random Forest + XGBoost | Benign, DDoS |

Both models keep **their own preprocessing pipelines and feature sets**. Streamlit is only the common dashboard layer.

**Important:** Both NIDS and DDoS models are trained and integrated into the unified dashboard. DDoS monitoring replays CICIDS2017 traffic records for offline detection simulation.
''')
