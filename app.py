import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# Optional advanced libs
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False


# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="AI Reality Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}

h1, h2, h3 {
    color: #00f5c4;
    font-weight: bold;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #00f5c4;
}

.stButton>button {
    background-color: #00f5c4;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #00d9a3;
}

.stSlider {
    padding: 10px;
}

.success-box {
    background-color: #1a3a1a;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #00f5c4;
}

.warning-box {
    background-color: #3a2a1a;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #ff9800;
}

.error-box {
    background-color: #3a1a1a;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #ff4444;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data():
    """Load Boston Housing dataset"""
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    df = pd.DataFrame(data, columns=cols)
    df['MEDV'] = target
    return df

# =========================
# TRAIN MODELS (CACHED)
# =========================
@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train all models and return predictions"""
    
    # Linear Regression
    lin_model = LinearRegression().fit(X_train, y_train)
    y_train_pred_lin = lin_model.predict(X_train)
    y_test_pred_lin = lin_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ).fit(X_train, y_train)
    y_train_pred_rf = rf_model.predict(X_train)
    y_test_pred_rf = rf_model.predict(X_test)
    
    # XGBoost (if available)
    xgb_model = None
    y_train_pred_xgb = None
    y_test_pred_xgb = None
    
    if xgb_available:
        xgb_model = XGBRegressor(
            n_estimators=100,
            random_state=42,
            verbosity=0
        ).fit(X_train, y_train)
        y_train_pred_xgb = xgb_model.predict(X_train)
        y_test_pred_xgb = xgb_model.predict(X_test)
    
    return {
        'lin_model': lin_model,
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'y_train_pred_lin': y_train_pred_lin,
        'y_test_pred_lin': y_test_pred_lin,
        'y_train_pred_rf': y_train_pred_rf,
        'y_test_pred_rf': y_test_pred_rf,
        'y_train_pred_xgb': y_train_pred_xgb,
        'y_test_pred_xgb': y_test_pred_xgb,
    }

# =========================
# METRICS CALCULATION
# =========================
def calculate_metrics(y_true, y_pred, set_name=""):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'set_name': set_name
    }

def display_metrics_cards(title, y_true, y_pred, model_name=""):
    """Display metrics in Streamlit cards"""
    metrics = calculate_metrics(y_true, y_pred, title)
    
    st.markdown(f"### {title} {model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("MAE", f"${metrics['mae']:.2f}k", help="Mean Absolute Error")
    with col2:
        st.metric("MSE", f"${metrics['mse']:.2f}k²", help="Mean Squared Error")
    with col3:
        st.metric("RMSE", f"${metrics['rmse']:.2f}k", help="Root Mean Squared Error")
    with col4:
        st.metric("R²", f"{metrics['r2']:.4f}", help=f"Explains {metrics['r2']*100:.2f}% of variance")
    
    # Health indicator
    if metrics['r2'] < 0.5:
        st.error("⚠️ Model is weak — missing important variables")
    elif metrics['r2'] < 0.75:
        st.warning("⚠️ Model is decent but improvable")
    else:
        st.success("✅ Strong predictive performance")
    
    return metrics

# =========================
# LOAD & PREPARE DATA
# =========================
df = load_data()

X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all models
models = train_models(X_train, X_test, y_train, y_test)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Model Performance",
    "📈 Visualization & SHAP",
    "🧠 Reality + Simulation",
    "📥 Download Report"
])

# =========================
# PAGE 1: OVERVIEW
# =========================
if page == "🏠 Overview":
    st.title("🏠 AI Reality Lab")
    
    st.markdown("""
    ### Welcome to Production-Grade ML Education
    
    This app demonstrates **real-world machine learning** with honest evaluation.
    
    #### 🔑 Key Concepts
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✅ Train/Test Split
        - Models trained on 80% of data
        - Evaluated on unseen 20%
        - Prevents fake accuracy
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Multiple Models
        - Linear Regression (baseline)
        - Random Forest (captures interactions)
        - XGBoost (best for structured data)
        """)
    
    st.divider()
    
    st.subheader("🏠 Interactive House Price Predictor")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rm = st.slider("Number of Rooms", 3.0, 9.0, 6.0, step=0.1)
    
    with col2:
        lstat = st.slider("Lower Status %", 1.0, 40.0, 10.0, step=0.5)
    
    with col3:
        crim = st.slider("Crime Rate", 0.0, 20.0, 5.0, step=0.5)
    
    if st.button("🔮 Predict House Price", use_container_width=True):
        # Create input DataFrame with mean values
        input_data = X.mean().values.reshape(1, -1)
        input_df = pd.DataFrame(input_data, columns=X.columns)
        
        # Update with user inputs
        input_df["RM"] = rm
        input_df["LSTAT"] = lstat
        input_df["CRIM"] = crim
        
        st.markdown("### 🎯 Predictions Across Models")
        
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            lin_pred = models['lin_model'].predict(input_df)
            st.success(f"**Linear Regression**\n${lin_pred:.2f}k")
        
        with pred_col2:
            rf_pred = models['rf_model'].predict(input_df)
            st.success(f"**Random Forest**\n${rf_pred:.2f}k")
        
        with pred_col3:
            if xgb_available:
                xgb_pred = models['xgb_model'].predict(input_df)
                st.success(f"**XGBoost**\n${xgb_pred:.2f}k")
            else:
                st.info("XGBoost not installed")
        
        st.info(f"📊 **Average Prediction**: ${np.mean([lin_pred, rf_pred]):.2f}k")


# =========================
# PAGE 2: MODEL PERFORMANCE
# =========================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance (Unseen Test Data)")
    
    st.markdown("""
    > **Why Test Set Matters**: These metrics are calculated on data the model has **never seen**.
    > This is your real-world performance indicator.
    """)
    
    # Linear Regression
    st.divider()
    st.subheader("🔵 Linear Regression")
    
    lin_train_metrics = display_metrics_cards(
        "Training Performance",
        y_train,
        models['y_train_pred_lin'],
        "📈"
    )
    
    lin_test_metrics = display_metrics_cards(
        "Test Performance (Real)",
        y_test,
        models['y_test_pred_lin'],
        "🎯"
    )
    
    # Random Forest
    st.divider()
    st.subheader("🟢 Random Forest")
    
    rf_train_metrics = display_metrics_cards(
        "Training Performance",
        y_train,
        models['y_train_pred_rf'],
        "📈"
    )
    
    rf_test_metrics = display_metrics_cards(
        "Test Performance (Real)",
        y_test,
        models['y_test_pred_rf'],
        "🎯"
    )
    
    # XGBoost
    if xgb_available:
        st.divider()
        st.subheader("🟡 XGBoost")
        
        xgb_train_metrics = display_metrics_cards(
            "Training Performance",
            y_train,
            models['y_train_pred_xgb'],
            "📈"
        )
        
        xgb_test_metrics = display_metrics_cards(
            "Test Performance (Real)",
            y_test,
            models['y_test_pred_xgb'],
            "🎯"
        )
    
    # Cross-validation
    st.divider()
    st.subheader("🔄 Cross-Validation (Most Reliable)")
    
    cv_col1, cv_col2, cv_col3 = st.columns(3)
    
    with cv_col1:
        cv_lin = cross_val_score(models['lin_model'], X, y, cv=5, scoring='r2')
        st.metric("Linear CV R²", f"{cv_lin.mean():.4f}", f"±{cv_lin.std():.4f}")
    
    with cv_col2:
        cv_rf = cross_val_score(models['rf_model'], X, y, cv=5, scoring='r2')
        st.metric("RF CV R²", f"{cv_rf.mean():.4f}", f"±{cv_rf.std():.4f}")
    
    with cv_col3:
        if xgb_available:
            cv_xgb = cross_val_score(models['xgb_model'], X, y, cv=5, scoring='r2')
            st.metric("XGBoost CV R²", f"{cv_xgb.mean():.4f}", f"±{cv_xgb.std():.4f}")
    
    st.info("""
    **What is Cross-Validation?**
    - Splits data into 5 random folds
    - Trains 5 different models
    - Averages performance
    - Most realistic estimate of generalization
    """)


# =========================
# PAGE 3: VISUALIZATION & SHAP
# =========================
elif page == "📈 Visualization & SHAP":
    st.title("📈 Model Insights & Explainability")
    
    # Feature Importance
    st.subheader("🎯 Feature Importance (Random Forest)")
    
    importances = models['rf_model'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='#00f5c4')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(importance_df, use_container_width=True)
    
    # Actual vs Predicted
    st.divider()
    st.subheader("📊 Actual vs Predicted (Random Forest)")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set
    axes.scatter(y_train, models['y_train_pred_rf'], alpha=0.6, c=y_train, cmap='viridis')
    axes.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes.set_xlabel('Actual Price ($k)', fontsize=11)
    axes.set_ylabel('Predicted Price ($k)', fontsize=11)
    axes.set_title('Training Set', fontsize=12, fontweight='bold')
    axes.grid(True, alpha=0.3)
    
    # Test set
    scatter = axes.scatter(y_test, models['y_test_pred_rf'], alpha=0.6, c=y_test, cmap='viridis')
    axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes.set_xlabel('Actual Price ($k)', fontsize=11)
    axes.set_ylabel('Predicted Price ($k)', fontsize=11)
    axes.set_title('Test Set (Real Performance)', fontsize=12, fontweight='bold')
    axes.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes, label='Price ($k)')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Residual Analysis
    st.divider()
    st.subheader("📉 Residual Analysis")
    
    residuals_train = y_train - models['y_train_pred_rf']
    residuals_test = y_test - models['y_test_pred_rf']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training residuals
    axes.scatter(models['y_train_pred_rf'], residuals_train, alpha=0.6, color='blue')
    axes.axhline(y=0, color='r', linestyle='--', lw=2)
    axes.set_xlabel('Predicted Price ($k)', fontsize=11)
    axes.set_ylabel('Residuals ($k)', fontsize=11)
    axes.set_title('Training Residuals', fontsize=12, fontweight='bold')
    axes.grid(True, alpha=0.3)
    
    # Test residuals
    axes.scatter(models['y_test_pred_rf'], residuals_test, alpha=0.6, color='green')
    axes.axhline(y=0, color='r', linestyle='--', lw=2)
    axes.set_xlabel('Predicted Price ($k)', fontsize=11)
    axes.set_ylabel('Residuals ($k)', fontsize=11)
    axes.set_title('Test Residuals (Real Performance)', fontsize=12, fontweight='bold')
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # SHAP Explanations
    if shap_available:
        st.divider()
        st.subheader("🔍 SHAP Explainability (Why Predictions Happen)")
        
        st.info("SHAP shows which features push predictions up or down for each sample")
        
        try:
            explainer = shap.Explainer(models['rf_model'], X_test)
            shap_values = explainer(X_test[:100])  # Use subset for performance
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP visualization error: {e}")
    else:
        st.info("📦 Install SHAP for explainability: `pip install shap`")


# =========================
# PAGE 4: REALITY + SIMULATION
# =========================
elif page == "🧠 Reality + Simulation":
    st.title("🧠 Reality + Business Simulation")
    
    st.markdown("""
    ### Why Real-World ML is Different
    
    This section shows how models behave when assumptions break down.
    """)
    
    # Generate synthetic marketing data
    np.random.seed(42)
    marketing_spend = np.random.uniform(0, 100, 200)
    
    # Non-linear reality: diminishing returns
    sales = 50 + 10*np.log1p(marketing_spend) + np.random.normal(0, 2, 200)
    
    sim_df = pd.DataFrame({
        "Marketing Spend": marketing_spend,
        "Sales": sales
    })
    
    # Train simulation model
    sim_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(
        sim_df[["Marketing Spend"]],
        sim_df["Sales"]
    )
    
    st.subheader("📢 Marketing Spend → Sales Prediction")
    
    spend = st.slider("Marketing Spend ($k)", 0, 100, 20)
    
    # Create proper DataFrame
    input_df_sim = pd.DataFrame({"Marketing Spend": [spend]})
    pred_sales = sim_model.predict(input_df_sim)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"**Predicted Sales**: ${pred_sales:.2f}k")
        st.info(f"**Spend**: ${spend}k")
    
    with col2:
        roi = (pred_sales - spend) / spend * 100 if spend > 0 else 0
        st.metric("ROI", f"{roi:.1f}%")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(marketing_spend, sales, alpha=0.5, s=50, label='Actual Data', color='blue')
    
    # Prediction line
    spend_range = np.linspace(0, 100, 100).reshape(-1, 1)
    predictions = sim_model.predict(spend_range)
    ax.plot(spend_range, predictions, 'r-', linewidth=3, label='Model Prediction')
    
    # Highlight current prediction
    ax.scatter([spend], [pred_sales], color='green', s=200, marker='*', 
               label=f'Current Prediction (${spend}k)', zorder=5)
    
    ax.set_xlabel('Marketing Spend ($k)', fontsize=12)
    ax.set_ylabel('Sales ($k)', fontsize=12)
    ax.set_title('Marketing Spend → Sales (Non-Linear Relationship)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    st.subheader("💡 Key Insights")
    
    st.markdown("""
    #### 🔑 Why This Matters
    
    1. **Diminishing Returns**: More spend doesn't always mean proportional sales increase
    2. **Non-Linear Relationships**: Real world rarely follows straight lines
    3. **Noise**: Random variation (market conditions, timing, etc.)
    4. **Model Limitations**: Even RF can't perfectly predict future behavior
    
    #### 🎯 Business Implications
    
    - Don't assume linear scaling
    - Test different spend levels
    - Monitor actual vs predicted
    - Adjust strategy based on real ROI
    """)
    
    st.divider()
    st.subheader("🚀 Future Evolution")
    
    st.markdown("""
    This project can evolve into:
    
    ### 📈 Advanced Modeling
    - Time-series forecasting (sales over weeks/months)
    - Seasonal decomposition
    - Trend analysis
    
    ### 🔍 Causal Inference
    - A/B testing frameworks
    - Uplift modeling (who to target)
    - Causal graphs
    
    ### 🧠 Explainability
    - SHAP interaction plots
    - Feature dependency analysis
    - Counterfactual explanations
    
    ### 🏢 Production Systems
    - Live data pipelines
    - API deployment
    - Real-time predictions
    - Model monitoring & retraining
    """)


# =========================
# PAGE 5: DOWNLOAD REPORT
# =========================
elif page == "📥 Download Report":
    st.title("📥 Generate & Download Report")
    
    st.markdown("### Create a comprehensive PDF report of your model evaluation")
    
    if st.button("📊 Generate Full Report", use_container_width=True):
        with st.spinner("Generating report..."):
            # Create comprehensive figure
            fig = plt.figure(figsize=(16, 20))
            gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle('AI Reality Lab — Complete Model Evaluation Report', 
                        fontsize=20, fontweight='bold', y=0.995)
            
            # 1. Metrics Summary (Text)
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('off')
            
            metrics_text = f"""
LINEAR REGRESSION
  Train R²: {r2_score(y_train, models['y_train_pred_lin']):.4f}  |  Test R²: {r2_score(y_test, models['y_test_pred_lin']):.4f}
  Train RMSE: ${np.sqrt(mean_squared_error(y_train, models['y_train_pred_lin'])):.2f}k  |  Test RMSE: ${np.sqrt(mean_squared_error(y_test, models['y_test_pred_lin'])):.2f}k

RANDOM FOREST
  Train R²: {r2_score(y_train, models['y_train_pred_rf']):.4f}  |  Test R²: {r2_score(y_test, models['y_test_pred_rf']):.4f}
  Train RMSE: ${np.sqrt(mean_squared_error(y_train, models['y_train_pred_rf'])):.2f}k  |  Test RMSE: ${np.sqrt(mean_squared_error(y_test, models['y_test_pred_rf'])):.2f}k
            """
            
            if xgb_available:
                metrics_text += f"""
XGBOOST
  Train R²: {r2_score(y_train, models['y_train_pred_xgb']):.4f}  |  Test R²: {r2_score(y_test, models['y_test_pred_xgb']):.4f}
  Train RMSE: ${np.sqrt(mean_squared_error(y_train, models['y_train_pred_xgb'])):.2f}k  |  Test RMSE: ${np.sqrt(mean_squared_error(y_test, models['y_test_pred_xgb'])):.2f}k
            """
            
            ax1.text(0.05, 0.5, metrics_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 2. Actual vs Predicted (Linear)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.scatter(y_test, models['y_test_pred_lin'], alpha=0.6, c='blue')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual Price ($k)')
            ax2.set_ylabel('Predicted Price ($k)')
            ax2.set_title('Linear Regression: Actual vs Predicted')
            ax2.grid(True, alpha=0.3)
            
            # 3. Actual vs Predicted (RF)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.scatter(y_test, models['y_test_pred_rf'], alpha=0.6, c='green')
            ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax3.set_xlabel('Actual Price ($k)')
            ax3.set_ylabel('Predicted Price ($k)')
            ax3.set_title('Random Forest: Actual vs Predicted')
            ax3.grid(True, alpha=0.3)
            
            # 4. Residuals (Linear)
            ax4 = fig.add_subplot(gs[2, 0])
            residuals_lin = y_test - models['y_test_pred_lin']
            ax4.scatter(models['y_test_pred_lin'], residuals_lin, alpha=0.6, c='blue')
            ax4.axhline(y=0, color='r', linestyle='--', lw=2)
            ax4.set_xlabel('Predicted Price ($k)')
            ax4.set_ylabel('Residuals ($k)')
            ax4.set_title('Linear Regression: Residuals')
            ax4.grid(True, alpha=0.3)
            
            # 5. Residuals (RF)
            ax5 = fig.add_subplot(gs[2, 1])
            residuals_rf = y_test - models['y_test_pred_rf']
            ax5.scatter(models['y_test_pred_rf'], residuals_rf, alpha=0.6, c='green')
            ax5.axhline(y=0, color='r', linestyle='--', lw=2)
            ax5.set_xlabel('Predicted Price ($k)')
            ax5.set_ylabel('Residuals ($k)')
            ax5.set_title('Random Forest: Residuals')
            ax5.grid(True, alpha=0.3)
            
            # 6. Feature Importance
            ax6 = fig.add_subplot(gs[3, :])
            importances = models['rf_model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(10)
            
            ax6.barh(importance_df['Feature'], importance_df['Importance'], color='#00f5c4')
            ax6.set_xlabel('Importance Score')
            ax6.set_title('Top 10 Feature Importance (Random Forest)')
            ax6.invert_yaxis()
            
            # Save to bytes
            pdf_buffer = BytesIO()
            plt.savefig(pdf_buffer, format='pdf', dpi=300, bbox_inches='tight')
            pdf_buffer.seek(0)
            
            st.success("✅ Report generated successfully!")
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_buffer,
                file_name="AI_Reality_Lab_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            plt.close(fig)
    
    st.divider()
    st.subheader("📊 Export Data")
    
    # Export metrics as CSV
    if st.button("📄 Export Metrics as CSV", use_container_width=True):
        metrics_export = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest'] + (['XGBoost'] if xgb_available else []),
            'Train R²': [
                r2_score(y_train, models['y_train_pred_lin']),
                r2_score(y_train, models['y_train_pred_rf']),
            ] + ([r2_score(y_train, models['y_train_pred_xgb'])] if xgb_available else []),
            'Test R²': [
                r2_score(y_test, models['y_test_pred_lin']),
                r2_score(y_test, models['y_test_pred_rf']),
            ] + ([r2_score(y_test, models['y_test_pred_xgb'])] if xgb_available else []),
            'Train RMSE': [
                np.sqrt(mean_squared_error(y_train, models['y_train_pred_lin'])),
                np.sqrt(mean_squared_error(y_train, models['y_train_pred_rf'])),
            ] + ([np.sqrt(mean_squared_error(y_train, models['y_train_pred_xgb']))] if xgb_available else []),
            'Test RMSE': [
                np.sqrt(mean_squared_error(y_test, models['y_test_pred_lin'])),
                np.sqrt(mean_squared_error(y_test, models['y_test_pred_rf'])),
            ] + ([np.sqrt(mean_squared_error(y_test, models['y_test_pred_xgb']))] if xgb_available else []),
        })
        
        csv_buffer = metrics_export.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Metrics CSV",
            data=csv_buffer,
            file_name="model_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.dataframe(metrics_export, use_container_width=True)
