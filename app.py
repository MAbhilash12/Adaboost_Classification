import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AdaBoost Classification Dashboard",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/adaboost_classifier.pkl"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/breast_cancer.csv"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background-color: white;
    color: black;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: white;
}

/* Sidebar Text */

[data-testid="stSidebar"] * {
    color: black !important;
}

/* Title */

h1, h2, h3, h4 {
    color: #1E3A8A;
}

/* Slider */

.stSlider > div > div {
    color: #1E3A8A;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    background-color: white;
}

/* Metric Cards */

div[data-testid="metric-container"] {
    background-color: #F3F4F6;
    border: 1px solid #D1D5DB;
    padding: 15px;
    border-radius: 10px;
}

/* Remove dark top padding */

.block-container {
    padding-top: 2rem;
}

/* Button */

.stButton>button {
    background-color: #2563EB;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🎯 AdaBoost Classification Dashboard")

st.markdown("""
### Breast Cancer Prediction using Machine Learning

This application predicts whether a tumor is:

- ✅ Benign
- ⚠️ Malignant
""")

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.title("📌 Enter Tumor Details")

mean_radius = st.sidebar.slider(
    "Mean Radius",
    5.0,
    30.0,
    15.0
)

mean_texture = st.sidebar.slider(
    "Mean Texture",
    5.0,
    40.0,
    20.0
)

mean_perimeter = st.sidebar.slider(
    "Mean Perimeter",
    40.0,
    200.0,
    80.0
)

mean_area = st.sidebar.slider(
    "Mean Area",
    100.0,
    2500.0,
    500.0
)

mean_smoothness = st.sidebar.slider(
    "Mean Smoothness",
    0.05,
    0.2,
    0.1
)

# ==========================================
# CREATE INPUT DATA
# ==========================================

input_data = pd.DataFrame({
    "mean radius": [mean_radius],
    "mean texture": [mean_texture],
    "mean perimeter": [mean_perimeter],
    "mean area": [mean_area],
    "mean smoothness": [mean_smoothness]
})

# ==========================================
# ADD REMAINING FEATURES
# ==========================================

all_columns = model.feature_names_in_

for col in all_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[all_columns]

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(input_data)

probability = model.predict_proba(input_data)

# ==========================================
# RESULT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🧠 Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Benign Tumor")
    else:
        st.error("⚠️ Malignant Tumor")

    st.subheader("📊 Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": ["Malignant", "Benign"],
        "Probability": probability[0]
    })

    st.dataframe(prob_df)

with col2:

    st.subheader("📚 Dataset Head")

    st.dataframe(df.head())

# ==========================================
# PROBABILITY BAR GRAPH
# ==========================================

st.subheader("📈 Prediction Probability Graph")

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(
    prob_df["Class"],
    prob_df["Probability"]
)

ax.set_ylabel("Probability")
ax.set_title("Prediction Confidence")

st.pyplot(fig)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

st.subheader("🔥 Top 10 Important Features")

importance = model.feature_importances_

features = model.feature_names_in_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(10)

fig2, ax2 = plt.subplots(figsize=(10,6))

ax2.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax2.set_title("Feature Importance")

ax2.set_xlabel("Importance Score")

st.pyplot(fig2)

# ==========================================
# PIE CHART
# ==========================================

st.subheader("🥧 Probability Distribution")

fig3, ax3 = plt.subplots(figsize=(5,5))

ax3.pie(
    prob_df["Probability"],
    labels=prob_df["Class"],
    autopct="%1.1f%%"
)

st.pyplot(fig3)

# ==========================================
# MODEL METRICS
# ==========================================

st.subheader("📊 Model Information")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="Algorithm",
        value="AdaBoost"
    )

with m2:
    st.metric(
        label="Estimators",
        value="100"
    )

with m3:
    st.metric(
        label="Learning Rate",
        value="0.5"
    )

# ==========================================
# DATASET SUMMARY
# ==========================================

st.subheader("📚 Dataset Summary")

st.dataframe(df.describe())

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

