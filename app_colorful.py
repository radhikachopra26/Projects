# =========================================================
# 🧠 NEUROSENSE AI PRO
# FINAL CLEAN PROFESSIONAL VERSION
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import datetime
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NeuroSense AI Pro",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# FILES
# =========================================================

STRESS_MLP = "EEG_TF_F_T_Stress_MLP.keras"
RF_MODEL = "EEG_Stress_RF_Model.pkl"
XGB_MODEL = "EEG_Task_XGBoost.pkl"
TASK_MODEL = "Task_Classification_Model.pkl"
SCALER = "Scaler.pkl"
SELECTOR = "FeatureSelector.pkl"
LOG_FILE = "prediction_log.csv"

# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    models = {}

    models["mlp"] = tf.keras.models.load_model(
        STRESS_MLP,
        compile=False
    )

    models["rf"] = joblib.load(
        RF_MODEL
    )

    models["xgb"] = joblib.load(
        XGB_MODEL
    )

    models["task"] = joblib.load(
        TASK_MODEL
    )

    models["scaler"] = joblib.load(
        SCALER
    )

    models["selector"] = joblib.load(
        SELECTOR
    )

    return models

models = load_models()

mlp_model = models["mlp"]
rf_model = models["rf"]
xgb_model = models["xgb"]
task_model = models["task"]
scaler = models["scaler"]
selector = models["selector"]

# =========================================================
# FEATURE COUNT
# =========================================================

EXPECTED = scaler.n_features_in_

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#020617,#0f172a);
color:white;
}

.big-title{
font-size:52px;
font-weight:900;
color:#38bdf8;
text-shadow:0 0 20px #0ea5e9;
}

.card{
padding:28px;
border-radius:22px;
background:rgba(17,24,39,.85);
backdrop-filter:blur(12px);
border:1px solid rgba(56,189,248,.25);
box-shadow:0 0 30px rgba(56,189,248,.18);
margin-bottom:20px;
}

.metric-box{
padding:22px;
border-radius:18px;
background:linear-gradient(
145deg,
#052e16,
#064e3b
);
border:1px solid #10b98188;
text-align:center;
}

.ai-box{
padding:22px;
border-radius:18px;
background:linear-gradient(
145deg,
#1e1b4b,
#312e81
);
border:1px solid #818cf8;
line-height:1.8;
font-size:16px;
}

.stButton>button{
width:100%;
border-radius:14px;
height:52px;
font-size:18px;
font-weight:700;
background:linear-gradient(
90deg,
#0ea5e9,
#2563eb
);
color:white;
border:none;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="big-title">🧠 NeuroSense AI Pro</div>',
    unsafe_allow_html=True
)

st.caption(
    "Hybrid Ensemble EEG Cognitive Stress Intelligence Platform"
)

# =========================================================
# PREPROCESS
# =========================================================

def preprocess(x):

    given = x.shape[1]

    if given < EXPECTED:

        x = np.pad(
            x,
            ((0,0),(0,EXPECTED-given)),
            mode="constant"
        )

    elif given > EXPECTED:

        x = x[:, :EXPECTED]

    x = scaler.transform(x)

    try:
        x = selector.transform(x)
    except:
        pass

    x = np.nan_to_num(x)

    return x.astype(np.float32)

# =========================================================
# RAW FEATURE FIXER
# =========================================================

def fix_raw_features(x, expected):

    current = x.shape[1]

    if current < expected:

        x = np.pad(
            x,
            ((0,0),(0,expected-current)),
            mode="constant"
        )

    elif current > expected:

        x = x[:, :expected]

    return x.astype(np.float32)

# =========================================================
# SAFE PROBA
# =========================================================

def safe_predict_proba(model, x):

    try:
        return model.predict_proba(x)

    except:
        return np.array([
            [0.33,0.33,0.33]
        ])

# =========================================================
# STRESS SCORE
# =========================================================

def compute_stress_score(
    probs,
    model_name=""
):

    probs = np.array(probs)

    if model_name == "TensorFlow Stress MLP":

        weights = np.array([
            20,
            50,
            95
        ])

    elif model_name == "Random Forest":

        weights = np.array([
            30,
            60,
            85
        ])

    elif model_name == "XGBoost":

        weights = np.array([
            25,
            55,
            90
        ])

    elif model_name == "Ensemble AI":

        weights = np.array([
            28,
            58,
            92
        ])

    else:

        return 0

    score = np.sum(
        probs * weights
    )

    confidence = np.max(probs)

    entropy = -np.sum(
        probs * np.log(probs + 1e-9)
    )

    score += confidence * 12

    score -= entropy * 4

    return round(
        np.clip(score,0,100),
        2
    )

# =========================================================
# LEVEL
# =========================================================

def get_stress_level(score):

    if score >= 70:
        return "High Stress 🔴"

    elif score >= 40:
        return "Medium Stress 🟡"

    else:
        return "Low Stress 🟢"

# =========================================================
# AI REPORT
# =========================================================

def generate_ai_analysis(score):

    if score >= 70:

        return """
🚨 High stress detected

• Meditation recommended
• Reduce workload
• Take proper sleep
• Avoid long screen exposure
"""

    elif score >= 40:

        return """
⚠️ Moderate stress detected

• Take regular breaks
• Stay hydrated
• Practice breathing exercises
"""

    else:

        return """
✅ Low stress detected

• Maintain healthy routine
• Continue exercise
"""

# =========================================================
# TASK DETECTION
# =========================================================

def predict_task(arr):

    try:

        expected = getattr(
            task_model,
            "n_features_in_",
            2369
        )

        x = fix_raw_features(
            arr.copy(),
            expected
        )

        pred = task_model.predict(x)

        if isinstance(pred[0], str):
            return pred[0]

        label_map = {

            0: "Arithmetic",

            1: "Memory",

            2: "Stroop"
        }

        pred_val = int(pred[0])

        return label_map.get(
            pred_val,
            f"Task-{pred_val}"
        )

    except Exception as e:

        st.error(f"TASK ERROR: {e}")

        return "Unknown"

# =========================================================
# PREDICTION ENGINE
# =========================================================

def predict_stress(arr, model_name):

    x = preprocess(arr)

    # =====================================================
    # TENSORFLOW
    # =====================================================

    if model_name == "TensorFlow Stress MLP":

        probs = mlp_model.predict(x)

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    elif model_name == "Random Forest":

        probs = safe_predict_proba(
            rf_model,
            x
        )

        if probs.shape[1] == 1:

            val = probs[0][0]

            probs = np.array([
                [
                    1-val,
                    val/2,
                    val/2
                ]
            ])

    # =====================================================
    # XGBOOST
    # =====================================================

    elif model_name == "XGBoost":

        probs = safe_predict_proba(
            xgb_model,
            x
        )

    # =====================================================
    # TASK CLASSIFIER
    # =====================================================

    elif model_name == "Task Classifier":

        try:

            expected = getattr(
                task_model,
                "n_features_in_",
                2369
            )

            x_task = fix_raw_features(
                arr.copy(),
                expected
            )

            pred = task_model.predict(x_task)

            if isinstance(pred[0], str):

                label = pred[0].lower()

                if label == "arithmetic":

                    probs = np.array([
                        [0.8,0.1,0.1]
                    ])

                elif label == "memory":

                    probs = np.array([
                        [0.1,0.8,0.1]
                    ])

                else:

                    probs = np.array([
                        [0.1,0.1,0.8]
                    ])

            else:

                label = int(pred[0])

                if label == 0:

                    probs = np.array([
                        [0.8,0.1,0.1]
                    ])

                elif label == 1:

                    probs = np.array([
                        [0.1,0.8,0.1]
                    ])

                else:

                    probs = np.array([
                        [0.1,0.1,0.8]
                    ])

        except Exception as e:

            st.error(f"TASK MODEL ERROR: {e}")

            probs = np.array([
                [0.33,0.33,0.33]
            ])

    # =====================================================
    # ENSEMBLE
    # =====================================================

    else:

        p1 = safe_predict_proba(
            rf_model,
            x
        )

        if p1.shape[1] == 1:

            val = p1[0][0]

            p1 = np.array([
                [
                    1-val,
                    val/2,
                    val/2
                ]
            ])

        p2 = safe_predict_proba(
            xgb_model,
            x
        )

        p3 = mlp_model.predict(x)

        min_classes = min(
            p1.shape[1],
            p2.shape[1],
            p3.shape[1]
        )

        p1 = p1[:, :min_classes]
        p2 = p2[:, :min_classes]
        p3 = p3[:, :min_classes]

        probs = (
            0.2 * p1 +
            0.4 * p2 +
            0.4 * p3
        )

    scores = []
    levels = []

    for p in probs:

        score = compute_stress_score(
            p,
            model_name
        )

        level = get_stress_level(score)

        scores.append(score)
        levels.append(level)

    return levels, scores, probs

# =========================================================
# LOGGING
# =========================================================

def log_prediction(
    model,
    level,
    score
):

    row = {

        "timestamp": datetime.datetime.now(),

        "input_source": model,

        "stress_level": level,

        "stress_score": score
    }

    df = pd.DataFrame([row])

    if not os.path.exists(LOG_FILE):

        df.to_csv(
            LOG_FILE,
            index=False
        )

    else:

        df.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )

# =========================================================
# MODEL COMPARISON
# =========================================================

def compare_models(arr):

    results = []

    model_list = [

        "TensorFlow Stress MLP",

        "Random Forest",

        "XGBoost",

        "Ensemble AI"
    ]

    for model_name in model_list:

        try:

            levels, scores, probs = predict_stress(
                arr,
                model_name
            )

            results.append({

                "Model": model_name,

                "Stress Score": scores[0],

                "Status": levels[0]
            })

        except:

            results.append({

                "Model": model_name,

                "Stress Score": 0,

                "Status": "Error"
            })

    return pd.DataFrame(results)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([

    "🧠 Live Prediction",

    "📊 Analytics",

    "⚙️ Model Comparison"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    raw = st.text_area(
        "Enter comma-separated EEG values",
        height=250
    )

    selected_model = st.selectbox(
        "Select Prediction Model",
        [
            "TensorFlow Stress MLP",
            "Random Forest",
            "XGBoost",
            "Task Classifier",
            "Ensemble AI"
        ]
    )

    if st.button("🚀 Analyze EEG"):

        try:

            vals = [

                float(v.strip())

                for v in raw.split(",")

                if v.strip() != ""
            ]

            arr = np.array(
                vals,
                dtype=np.float32
            ).reshape(1,-1)

            levels, scores, probs = predict_stress(
                arr,
                selected_model
            )

            detected_task = predict_task(arr)

            # =================================================
            # RESULT UI
            # =================================================

            if selected_model == "Task Classifier":

                st.markdown(f"""
                <div class="metric-box">

                <h1>🧠 Task Detection</h1>

                <h2>Detected Task: {detected_task}</h2>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="metric-box">

                <h1>{levels[0]}</h1>

                <h2>Stress Score: {scores[0]}</h2>

                <h3>Detected Task: {detected_task}</h3>

                </div>
                """, unsafe_allow_html=True)

                st.progress(
                    int(min(scores[0],100))
                )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Model Probabilities"
            )

            st.write(probs)

            if probs.shape[1] >= 3:

                chart_df = pd.DataFrame({

                    "Arithmetic":[probs[0][0]],

                    "Memory":[probs[0][1]],

                    "Stroop":[probs[0][2]]

                })

                st.bar_chart(chart_df)

            # =================================================
            # AI REPORT
            # =================================================

            if selected_model != "Task Classifier":

                st.subheader(
                    "🤖 AI Wellness Assistant"
                )

                report = generate_ai_analysis(
                    scores[0]
                )

                report = report.replace(
                    "\n",
                    "<br>"
                )

                st.markdown(f"""
                <div class="ai-box">

                {report}

                </div>
                """, unsafe_allow_html=True)

            # =================================================
            # SAVE LOG
            # =================================================

            log_prediction(

                selected_model,

                levels[0] if selected_model != "Task Classifier"
                else "Task Detection",

                scores[0] if selected_model != "Task Classifier"
                else 0
            )

            st.success(
                "Prediction Saved Successfully"
            )

            st.session_state["latest_arr"] = arr

        except Exception as e:

            st.error("Prediction Failed")

            st.code(str(e))

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# TAB 2 - ANALYTICS
# =========================================================

with tab2:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "📊 Analytics Dashboard"
    )

    if os.path.exists(LOG_FILE):

        logs = pd.read_csv(LOG_FILE)

        # REMOVE LAST 4 COLUMNS
        clean_logs = logs.iloc[:, :-4]

        st.dataframe(
            clean_logs.tail(),
            use_container_width=True
        )

        st.line_chart(
            logs["stress_score"]
        )

    else:

        st.info(
            "No analytics available yet"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# TAB 3 - MODEL COMPARISON
# =========================================================

with tab3:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "⚙️ Model Comparison"
    )

    if "latest_arr" in st.session_state:

        compare_df = compare_models(
            st.session_state["latest_arr"]
        )

        st.dataframe(
            compare_df,
            use_container_width=True
        )

        st.bar_chart(
            compare_df.set_index("Model")[
                "Stress Score"
            ]
        )

    else:

        st.info(
            "Run a prediction first to compare models."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🧠 NeuroSense AI Pro • Hybrid EEG Cognitive Stress Intelligence Platform"
)
