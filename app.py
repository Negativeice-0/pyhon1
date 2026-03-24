import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Optional advanced libs
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

try:
    import shap
    shap_available = True
except:
    shap_available = False


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="AI Reality Lab", layout="wide")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Model Performance",
    "📈 Visualization",
    "🧠 Reality + Simulation"
])

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    df = pd.DataFrame(data, columns=cols)
    df['MEDV'] = target
    return df

df = load_data()

X = df.drop(columns=['MEDV'])
y = df['MEDV']

# =========================
# 🔥 CRITICAL FIX: TRAIN/TEST SPLIT
# =========================
# WHY:
# Before, you trained and tested on SAME data → fake accuracy
# Now:
# - train = what model learns from
# - test = unseen data → REAL performance

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# MODELS
# =========================
lin_model = LinearRegression().fit(X_train, y_train)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
).fit(X_train, y_train)

if xgb_available:
    xgb_model = XGBRegressor(
        n_estimators=100,
        random_state=42
    ).fit(X_train, y_train)

# =========================
# PREDICTIONS (ON TEST SET!)
# =========================
y_pred_lin = lin_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

if xgb_available:
    y_pred_xgb = xgb_model.predict(X_test)

# =========================
# METRICS FUNCTION
# =========================
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)  # always supported
    rmse = np.sqrt(mse)  # manual root
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

lin_rmse, lin_r2 = get_metrics(y_test, y_pred_lin)
rf_rmse, rf_r2 = get_metrics(y_test, y_pred_rf)

if xgb_available:
    xgb_rmse, xgb_r2 = get_metrics(y_test, y_pred_xgb)

# =========================
# PAGE 1: OVERVIEW
# =========================
if page == "🏠 Overview":
    st.title("🏠 AI Reality Lab")

    st.markdown("""
This app shows how machine learning behaves in **real-world conditions**.

Key upgrade:
Now uses **train/test split**, meaning:
- Models are evaluated on unseen data
- Accuracy is realistic, not inflated

Next Evolution (better with xgboost):
time-series forecasting (marketing over time)
causal inference (what actually causes sales)
A/B testing simulation
""")

    st.subheader("🏠 Predict House Price")

    rm = st.slider("Rooms", 3.0, 9.0, 6.0)
    lstat = st.slider("Lower Status %", 1.0, 40.0, 10.0)

    if st.button("Predict"):
        input_data = X.mean().values.reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=X.columns)

        input_df["RM"] = rm
        input_df["LSTAT"] = lstat

        st.write("Predictions:")

        st.success(f"Linear: ${lin_model.predict(input_df)[0]:.2f}k")
        st.success(f"Random Forest: ${rf_model.predict(input_df)[0]:.2f}k")

        if xgb_available:
            st.success(f"XGBoost: ${xgb_model.predict(input_df)[0]:.2f}k")


# =========================
# PAGE 2: MODEL PERFORMANCE
# =========================
elif page == "📊 Model Performance":
    st.title("📊 Real Model Performance (Unseen Data)")

    col1, col2, col3 = st.columns(3)

    col1.metric("Linear RMSE", f"{lin_rmse:.2f}")
    col2.metric("Random Forest RMSE", f"{rf_rmse:.2f}")

    if xgb_available:
        col3.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")

    st.markdown("""
### 🔍 Interpretation

- Linear Regression → usually underfits
- Random Forest → captures interactions
- XGBoost → best for structured data

THIS is real performance—not fake training accuracy.
""")


# =========================
# PAGE 3: VISUALIZATION
# =========================
elif page == "📈 Visualization":
    st.title("📈 Feature Importance")

    # Random Forest importance
    importances = rf_model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(X.columns, importances)
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)

    # =========================
    # SHAP EXPLANATIONS
    # =========================
    if shap_available:
        st.subheader("🔍 SHAP Explanation (Why Predictions Happen)")

        explainer = shap.Explainer(rf_model, X_test)
        shap_values = explainer(X_test)

        fig2 = plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig2)

    else:
        st.info("Install SHAP for explainability: pip install shap")


# =========================
# PAGE 4: REALITY + SIMULATION
# =========================
elif page == "🧠 Reality + Simulation":
    st.title("🧠 Reality + Business Simulation")

    st.code("""
FINAL CONCLUSION:

1. WHY TRAIN/TEST SPLIT?
   - Prevents fake accuracy
   - Simulates real-world prediction
   - Without it → model is lying to you

2. WHY RANDOM FOREST / XGBOOST?
   - Real world is NOT linear
   - Captures:
        - interactions
        - non-linear effects
        - outliers

3. RANDOM DATA EXPERIMENT:
   If you generate random data:
   - Linear Regression → fails (no pattern)
   - Random Forest → overfits noise
   - XGBoost → also overfits

   LESSON:
   Models don't "understand"
   They detect patterns—even fake ones.

4. FEATURE IMPORTANCE:
   Shows what model THINKS matters
   Not necessarily what ACTUALLY matters

5. SHAP:
   Explains predictions locally
   Very powerful for interviews

6. MARKETING REALITY:
   More ad spend ≠ more sales (linearly)

   Effects include:
   - diminishing returns
   - timing delays
   - audience saturation

FINAL TRUTH:
All models are approximations.
Reality is always more complex.
""")

    st.subheader("📢 Marketing Simulation")

    # Fake dataset
    np.random.seed(42)
    marketing = np.random.uniform(0, 100, 200)

    # Non-linear reality
    sales = 50 + 10*np.log1p(marketing) + np.random.normal(0, 2, 200)

    sim_df = pd.DataFrame({
        "Marketing Spend": marketing,
        "Sales": sales
    })

    sim_model = RandomForestRegressor().fit(
        sim_df[["Marketing Spend"]],
        sim_df["Sales"]
    )

    spend = st.slider("Marketing Spend", 0, 100, 20)

    pred_sales = sim_model.predict([[spend]])[0]

    st.success(f"Predicted Sales: {pred_sales:.2f}")

    fig, ax = plt.subplots()
    ax.scatter(marketing, sales, alpha=0.4)
    ax.set_xlabel("Marketing Spend")
    ax.set_ylabel("Sales")
    st.pyplot(fig)