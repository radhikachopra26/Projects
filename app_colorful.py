import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import datetime
import os

MODEL = "EEG_TF_F_T_Stress_MLP.keras"
SCALER = "Scaler.pkl"
SELECTOR = "FeatureSelector.pkl"
ENCODER = "LabelEncoder.pkl"

model = load_model(MODEL)
scaler = joblib.load(SCALER)
selector = joblib.load(SELECTOR)
encoder = joblib.load(ENCODER)

EXPECTED = scaler.n_features_in_
LOG_FILE = "prediction_log.csv"


# ---------- BEAUTIFUL UI THEME ----------
st.set_page_config(
    page_title="EEG Stress Classifier",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>

body {
    background: linear-gradient(135deg,#0f172a,#020617);
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#0f172a,#020617);
    color:#e5e7eb;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.big-title {
    font-size:40px;
    font-weight:900;
    padding:12px;
    color:#38bdf8;
    text-shadow:0 0 12px #0ea5e9;
}

.card {
    padding:18px;
    border-radius:14px;
    background: linear-gradient(145deg,#0b1220,#111c35);
    box-shadow:0 0 18px rgba(56,189,248,.25);
    border:1px solid #38bdf855;
}

.success-box {
    padding:12px;
    border-radius:10px;
    background:linear-gradient(145deg,#052e16,#064e3b);
    border:1px solid #10b98188;
    color:#a7f3d0;
}

.warning-box {
    padding:12px;
    border-radius:10px;
    background:linear-gradient(145deg,#422006,#713f12);
    border:1px solid #fbbf2488;
    color:#fde68a;
}

.tab-style > div {
    border-radius:10px !important;
}

</style>
""", unsafe_allow_html=True)


# ---------- PREPROCESS ----------
def preprocess(x):
    given = x.shape[1]
    if given < EXPECTED:
        x = np.pad(x, ((0,0),(0,EXPECTED-given)), mode="constant")
    elif given > EXPECTED:
        x = x[:, :EXPECTED]
    x = scaler.transform(x)
    x = selector.transform(x)
    return x


# ---------- PREDICT ----------
def predict_array(arr):
    x = preprocess(arr)
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    labels = encoder.inverse_transform(preds)
    return labels, probs


# ---------- LOG ----------
def log_prediction(source, count, label, probs):
    row = {
        "timestamp": datetime.datetime.now(),
        "input_source": source,
        "features": count,
        "prediction": label,
        "prob_Arithmetic": probs[0],
        "prob_Memory": probs[1],
        "prob_Stroop": probs[2],
    }
    df = pd.DataFrame([row])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", index=False, header=False)


# ---------- HEADER ----------
st.markdown('<div class="big-title">🧠 EEG Stress Classification Dashboard</div>', unsafe_allow_html=True)
st.caption("✨ Deep Learning • Auto-Padding • Smart Scaling • Probability Visualization")


tab1, tab2 = st.tabs(["🎨 Single Input Mode", "📂 Batch CSV Mode"])


# ========== TAB 1 — MANUAL INPUT ==========
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Paste EEG Feature Vector")

    raw = st.text_area("Enter comma-separated values", height=140)

    if st.button("🔮 Predict Now", type="primary"):
        try:
            vals = [v for v in raw.replace("\t"," ").split(",") if v!=""]
            arr = np.array(vals, dtype=float).reshape(1,-1)

            labels, probs = predict_array(arr)

            st.markdown(f'<div class="success-box">🎯 Prediction: <b>{labels[0]}</b></div>', unsafe_allow_html=True)

            st.write("📊 Class Probabilities")
            st.bar_chart(pd.DataFrame({
                "Arithmetic":[probs[0][0]],
                "Memory":[probs[0][1]],
                "Stroop":[probs[0][2]]
            }))

            log_prediction("manual_input", arr.shape[1], labels[0], probs[0])
            st.markdown('<div class="warning-box">📝 Saved to prediction_log.csv</div>', unsafe_allow_html=True)

        except:
            st.error("❌ Invalid input — please enter numeric values only")

    st.markdown('</div>', unsafe_allow_html=True)



# ========== TAB 2 — CSV MODE ==========
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📂 Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        x = df.values.astype(float)

        labels, probs = predict_array(x)
        df["Prediction"] = labels

        st.success(f"Processed {len(df)} samples 👌")
        st.dataframe(df.head())

        out = "batch_predictions.csv"
        df.to_csv(out, index=False)

        for i in range(len(df)):
            log_prediction("csv_batch", x.shape[1], labels[i], probs[i])

        st.markdown('<div class="warning-box">📝 Logs Updated</div>', unsafe_allow_html=True)
        st.download_button("📥 Download Results", df.to_csv(index=False), file_name=out)

    st.markdown('</div>', unsafe_allow_html=True)
