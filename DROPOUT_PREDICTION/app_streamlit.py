import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from streamlit.components.v1 import html

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Student Dropout Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# LOAD MODEL
# ===============================
artifact = joblib.load("student_dropout_rf_best.pkl")
model = artifact["model"]

# ===============================
# 3D PARTICLE BACKGROUND
# ===============================
html(
    """
    <script src="https://cdn.jsdelivr.net/npm/particles.js@2.0.0/particles.min.js"></script>
    <div id="particles-js"></div>
    <style>
        #particles-js {
            position: fixed;
            width: 100%;
            height: 100%;
            z-index: -1;
            top: 0;
            left: 0;
            background: radial-gradient(circle at top, #0f172a, #020617);
        }
        .block-container {
            padding-top: 3rem;
        }
    </style>
    <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {"value": 70},
        "color": {"value": "#38bdf8"},
        "opacity": {"value": 0.4},
        "size": {"value": 3},
        "line_linked": {"enable": true, "color": "#38bdf8"},
        "move": {"enable": true, "speed": 1}
      }
    });
    </script>
    """,
    height=0,
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
h1, h2, h3, h4 {
    color: #e5e7eb;
    font-family: 'Inter', sans-serif;
}
.card {
    background: rgba(15, 23, 42, 0.85);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
}
.metric {
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
}
.label {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown(
    "<h1>🎓 Student Dropout Prediction Dashboard</h1>"
    "<p style='color:#94a3b8'>AI-powered risk analysis & probability estimation</p>",
    unsafe_allow_html=True
)

# ===============================
# INPUT FORM
# ===============================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📋 Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        study_hours = st.slider("Study Hours Per Week", 0, 40, 10)
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])

    with col2:
        previous_grade = st.slider("Previous Grade (%)", 0, 100, 60)
        attendance = st.slider("Attendance Percentage", 0, 100, 75)
        online_classes = st.selectbox("Online Classes Taken", ["No", "Yes"])

    with col3:
        parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    predict_btn = st.button("🚀 Predict Dropout Risk", key="predict_btn")

    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# PREDICTION LOGIC
# ===============================
if predict_btn:
    encoded_input = {
        "Gender": 1 if gender == "Male" else 0,
        "PreviousGrade": previous_grade,
        "AttendancePercent": attendance,
        "StudyHours": study_hours,
        "ParentalSupport": {"Low": 0, "Medium": 1, "High": 2}[parental_support],
        "ExtracurricularActivities": 1 if extracurricular == "Yes" else 0,
        "OnlineClassesTaken": 1 if online_classes == "Yes" else 0
    }

    df = pd.DataFrame([encoded_input])

    # SAFEST FEATURE ALIGNMENT
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    probability = model.predict_proba(df)[0][1]
    decision = "YES" if probability >= threshold else "NO"

    # ===============================
    # OUTPUT
    # ===============================
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Dropout Decision</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric'>{decision}</div>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='label'>Dropout Probability</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric'>{probability*100:.2f}%</div>",
            unsafe_allow_html=True
        )
        st.progress(probability)
        st.markdown("</div>", unsafe_allow_html=True)

    with colC:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 40], "color": "#22c55e"},
                    {"range": [40, 70], "color": "#facc15"},
                    {"range": [70, 100], "color": "#ef4444"}
                ]
            }
        ))
        fig.update_layout(height=260, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Feature Importance Explanation")

    importance_df = pd.DataFrame({
        "Feature": model.feature_names_in_,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    st.markdown(
        f"<p style='color:#94a3b8'>Decision threshold: {threshold}</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
