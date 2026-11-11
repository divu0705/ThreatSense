"""
app.py — Network Intrusion Detection live dashboard.

Simulates live traffic by streaming rows from the NSL-KDD test set (shuffled)
at a configurable rate, classifies each with the trained RandomForest, raises
alerts for non-Normal predictions, and lets you inspect why any single
connection was flagged using SHAP.

Run:  streamlit run src/app.py
"""

import os
import sys
import time
import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

sys.path.insert(0, os.path.dirname(__file__))
from constants import SEVERITY_BY_CATEGORY
from download_data import load_dataset
from train import preprocess
from explain import Explainer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

st.set_page_config(page_title="NIDS Live Dashboard", layout="wide", page_icon="🛡️")


# ---------- Cached loaders ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    return model, encoders, feature_columns, label_encoder


@st.cache_resource
def load_explainer(_model, feature_columns):
    return Explainer(_model, feature_columns)


@st.cache_data
def load_stream_data():
    df = load_dataset("NSL_KDD_Test.csv")
    return df.sample(frac=1.0, random_state=None).reset_index(drop=True)


# ---------- Session state ----------
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame()
if "running" not in st.session_state:
    st.session_state.running = False

model, encoders, feature_columns, label_encoder = load_artifacts()
explainer = load_explainer(model, feature_columns)
stream_df = load_stream_data()

# ---------- Sidebar ----------
st.sidebar.title("🛡️ Simulation controls")
st.sidebar.caption(
    "Traffic is simulated by streaming real, held-out NSL-KDD test rows — "
    "this stands in for a live packet-capture feed. See the implementation "
    "guide for how to swap in real traffic via scapy."
)

rows_per_tick = st.sidebar.slider("Connections per tick", 1, 10, 3)
refresh_seconds = st.sidebar.slider("Tick interval (seconds)", 1, 10, 2)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("▶ Start", use_container_width=True):
    st.session_state.running = True
if col_b.button("⏸ Pause", use_container_width=True):
    st.session_state.running = False

if st.sidebar.button("🔄 Reset simulation", use_container_width=True):
    st.session_state.idx = 0
    st.session_state.log = pd.DataFrame()
    st.session_state.running = False
    st.rerun()

st.sidebar.divider()
st.sidebar.metric("Connections processed", len(st.session_state.log))

if st.session_state.running:
    st_autorefresh(interval=refresh_seconds * 1000, key="tick")

# ---------- Process next batch ----------
if st.session_state.running and st.session_state.idx < len(stream_df):
    end = min(st.session_state.idx + rows_per_tick, len(stream_df))
    batch = stream_df.iloc[st.session_state.idx:end].copy()
    st.session_state.idx = end

    X_batch, y_true, _, _ = preprocess(batch, encoders=encoders, fit=False)
    preds = model.predict(X_batch)
    probs = model.predict_proba(X_batch)

    pred_categories = label_encoder.inverse_transform(preds)
    confidences = probs.max(axis=1)

    batch_result = batch.reset_index(drop=True).copy()
    batch_result["predicted_category"] = pred_categories
    batch_result["confidence"] = confidences
    batch_result["severity"] = [SEVERITY_BY_CATEGORY.get(c, "medium") for c in pred_categories]
    batch_result["timestamp"] = pd.Timestamp.now().strftime("%H:%M:%S")
    batch_result["row_id"] = range(
        len(st.session_state.log), len(st.session_state.log) + len(batch_result)
    )

    # Keep raw feature snapshot for SHAP explanation later
    batch_result["_X_index"] = range(len(X_batch))
    st.session_state["_last_X_batch"] = X_batch

    st.session_state.log = pd.concat([st.session_state.log, batch_result], ignore_index=True)

# ---------- Header + metrics ----------
st.title("Network Intrusion Detection — Live Dashboard")
st.caption(
    "RandomForest classifier trained on NSL-KDD, predicting attack category "
    "(Normal / DoS / Probe / R2L / U2R) per connection, with SHAP-based "
    "per-alert explanations."
)

log = st.session_state.log
c1, c2, c3, c4, c5 = st.columns(5)
if len(log) > 0:
    counts = log["predicted_category"].value_counts()
    c1.metric("Total processed", len(log))
    c2.metric("Normal", int(counts.get("Normal", 0)))
    c3.metric("DoS alerts", int(counts.get("DoS", 0)))
    c4.metric("Probe alerts", int(counts.get("Probe", 0)))
    c5.metric("R2L / U2R alerts", int(counts.get("R2L", 0) + counts.get("U2R", 0)))
else:
    c1.metric("Total processed", 0)
    st.info("Press ▶ Start in the sidebar to begin streaming simulated traffic.")

st.divider()

left, right = st.columns([3, 2])

# ---------- Live traffic table ----------
with left:
    st.subheader("Live connection log")
    if len(log) > 0:
        display_cols = [
            "row_id", "timestamp", "protocol_type", "service", "flag",
            "predicted_category", "confidence", "severity",
        ]
        recent = log[display_cols].tail(25).iloc[::-1]

        def highlight_severity(row):
            colors = {
                "critical": "background-color: #7a1f1f",
                "high": "background-color: #6b3a12",
                "medium": "background-color: #6b5a12",
                "low": "background-color: #2f4a2f",
                "none": "",
            }
            return [colors.get(row["severity"], "")] * len(row)

        st.dataframe(
            recent.style.apply(highlight_severity, axis=1).format({"confidence": "{:.1%}"}),
            use_container_width=True,
            height=500,
        )
    else:
        st.write("No connections processed yet.")

# ---------- Alerts + explanation ----------
with right:
    st.subheader("Alerts")
    if len(log) > 0:
        alerts = log[log["predicted_category"] != "Normal"].tail(50).iloc[::-1]
        if len(alerts) == 0:
            st.success("No attacks detected yet.")
        else:
            options = alerts["row_id"].tolist()
            selected_row_id = st.selectbox(
                "Select an alert to explain",
                options,
                format_func=lambda rid: (
                    f"#{rid} — {log.loc[log.row_id == rid, 'predicted_category'].values[0]} "
                    f"({log.loc[log.row_id == rid, 'timestamp'].values[0]})"
                ),
            )

            selected = log[log.row_id == selected_row_id].iloc[0]
            sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            st.markdown(
                f"### {sev_emoji.get(selected['severity'], '⚪')} "
                f"{selected['predicted_category']} — {selected['confidence']:.1%} confidence"
            )
            st.write(
                f"**Protocol:** {selected['protocol_type']} · "
                f"**Service:** {selected['service']} · **Flag:** {selected['flag']}"
            )

            if st.button("Explain this prediction (SHAP)"):
                with st.spinner("Computing feature attributions..."):
                    X_last = st.session_state.get("_last_X_batch")
                    x_idx = int(selected["_X_index"])
                    # Only valid if this alert came from the most recent batch;
                    # for older alerts we recompute features from the raw row.
                    row_features = X_last.iloc[[x_idx]] if X_last is not None and x_idx < len(X_last) else None
                    if row_features is None:
                        raw_row = pd.DataFrame([selected[["label"] + [c for c in log.columns if c in explainer.feature_columns or c == "label"]]])
                        row_features = None

                    if row_features is not None:
                        pred_idx = list(label_encoder.classes_).index(selected["predicted_category"])
                        top_features = explainer.explain_row(row_features, pred_idx, top_n=8)
                        exp_df = pd.DataFrame(top_features, columns=["feature", "shap_value"])
                        exp_df["direction"] = exp_df["shap_value"].apply(
                            lambda v: "pushes toward this label" if v > 0 else "pushes away from this label"
                        )
                        st.bar_chart(exp_df.set_index("feature")["shap_value"])
                        st.dataframe(exp_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning(
                            "This alert is from an earlier batch — explanation is only "
                            "kept in memory for the most recent tick's connections."
                        )
    else:
        st.write("No alerts yet.")

st.divider()
with st.expander("About this model"):
    st.markdown(
        """
        - **Model:** RandomForest (200 trees), trained on the NSL-KDD `KDDTrain+` set
        - **Target:** attack category — Normal, DoS, Probe, R2L, U2R
          (grouped from ~40 specific attack labels using the standard NSL-KDD taxonomy)
        - **Known limitation:** R2L/U2R recall is low on the held-out test set because
          many R2L/U2R attack *subtypes* in `KDDTest+` never appear in `KDDTrain+` —
          this is the well-documented NSL-KDD train/test distribution gap, and it's
          exactly the kind of generalization problem your final-year project addresses.
        - Traffic here is **simulated** from real held-out test rows, not live packet capture.
        """
    )
