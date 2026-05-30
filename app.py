import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG =================
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# ================= STYLES =================
st.markdown("""
<style>
.main {background-color:#0b1220;}

.welcome-box {
    background:#f472b6;
    padding:22px;
    border-radius:14px;
    color:#020617;
    font-size:26px;
    font-weight:700;
    text-align:center;
    margin-bottom:20px;
}

.sub-text {
    text-align:center;
    color:#fbcfe8;
    font-size:15px;
    margin-bottom:20px;
}

.section-title {
    background:#1e293b;
    padding:10px;
    border-radius:8px;
    color:#f472b6;
    font-size:19px;
    font-weight:600;
    margin:18px 0 10px 0;
}

.metric {
    background:#020617;
    padding:16px;
    border-radius:12px;
    border-left:5px solid #ec4899;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "step" not in st.session_state:
    st.session_state.step = 0

def next_step():
    st.session_state.step += 1
    st.rerun()

def prev_step():
    st.session_state.step -= 1
    st.rerun()

# ================= STEP 0 : UPLOAD =================
if st.session_state.step == 0:
    st.markdown("<div class='welcome-box'>Predictive Customer Churn Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Upload a CSV file to begin analysis</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        next_step()

# ================= STEP 1 : PREVIEW =================
elif st.session_state.step == 1:
    df = st.session_state.df
    st.markdown("<div class='section-title'>Dataset Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head())

    c1, c2 = st.columns(2)
    if c1.button("⬅ Back"): prev_step()
    if c2.button("Next ➡"): next_step()

# ================= STEP 2 : MODEL =================
elif st.session_state.step == 2:
    df = st.session_state.df.copy()

    st.markdown("<div class='section-title'>Model Configuration</div>", unsafe_allow_html=True)
    target_col = st.selectbox("Select churn column", df.columns)

    df[target_col] = df[target_col].astype(str).str.lower().str.strip()
    positive = ["yes", "1", "true", "churn", "left", "exit"]
    negative = ["no", "0", "false", "stay", "active"]

    df[target_col] = df[target_col].apply(
        lambda x: 1 if x in positive else (0 if x in negative else np.nan)
    )
    df.dropna(subset=[target_col], inplace=True)
    df[target_col] = df[target_col].astype(int)

    if df[target_col].nunique() != 2:
        st.error("Target column must be binary (e.g. Yes/No, 1/0).")
        st.stop()

    # FIX: Drop non-numeric columns that can't be encoded cleanly
    X_raw = df.drop(columns=[target_col])
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df[target_col]

    if X.shape[1] == 0:
        st.error("No usable feature columns found after encoding.")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # FIX: removed duplicate assignment
    threshold = 0.40
    y_pred = (y_prob >= threshold).astype(int)

    # FIX: Assign session state keys individually (safer than .update() with a dict)
    st.session_state.model = model
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.y_test = y_test
    st.session_state.y_pred = y_pred
    st.session_state.y_prob = y_prob
    st.session_state.accuracy = accuracy_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    if c1.button("⬅ Back"): prev_step()
    if c2.button("Next ➡"): next_step()

# ================= STEP 3 : DASHBOARD =================
elif st.session_state.step == 3:
    # FIX: Guard against missing session keys (e.g. if user refreshes mid-flow)
    required_keys = ["model", "X", "y", "y_test", "y_pred", "y_prob", "accuracy"]
    if not all(k in st.session_state for k in required_keys):
        st.warning("Session data lost. Please restart from the beginning.")
        if st.button("🏠 Go Home"):
            st.session_state.clear()
            st.rerun()
        st.stop()

    model    = st.session_state.model
    X        = st.session_state.X
    y        = st.session_state.y
    y_test   = st.session_state.y_test
    y_pred   = st.session_state.y_pred
    y_prob   = st.session_state.y_prob
    accuracy = st.session_state.accuracy

    st.markdown("<div class='section-title'>Key Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric'><h3>Total Records</h3><h2>{len(y)}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><h3>Churn Rate</h3><h2>{round(y.mean()*100, 2)}%</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><h3>Accuracy</h3><h2>{round(accuracy*100, 2)}%</h2></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Visual Analysis</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # --- Churn Distribution ---
    with col1:
        fig, ax = plt.subplots(figsize=(2.8, 2.2), dpi=160)
        sns.countplot(x=y, ax=ax)
        ax.set_title("Churn Distribution", fontsize=11)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # --- Churn Probability ---
    with col2:
        fig, ax = plt.subplots(figsize=(2.8, 2.2), dpi=160)
        sns.histplot(y_prob, bins=10, kde=True, ax=ax)
        ax.set_title("Churn Probability", fontsize=11)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

    # --- Confusion Matrix ---
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=160)
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="pink",
        cbar=False,
        annot_kws={"size": 10},
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # --- Feature Importance ---
    st.markdown("<div class='section-title'>Top Churn Drivers</div>", unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=160)
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
    ax.set_title("Top Features", fontsize=11)
    ax.tick_params(labelsize=9)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    st.download_button(
        "Download Feature Importance CSV",
        imp_df.to_csv(index=False),
        "churn_feature_report.csv"
    )

    c1, c2 = st.columns(2)
    if c1.button("⬅ Back"): prev_step()
    if c2.button("🏠 Home"):
        st.session_state.clear()
        st.rerun()