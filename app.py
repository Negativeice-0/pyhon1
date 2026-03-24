import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import hashlib
import json
from datetime import datetime
from pathlib import Path

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
# AUTHENTICATION SYSTEM
# =========================
def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load user database"""
    users_file = Path("users.json")
    if users_file.exists():
        with open(users_file, "r") as f:
            return json.load(f)
    return {
        "admin": {"password": hash_password("admin123"), "role": "admin"},
        "user": {"password": hash_password("user123"), "role": "user"}
    }

def save_users(users):
    """Save user database"""
    with open("users.json", "w") as f:
        json.dump(users, f)

def login_user(username, password, users):
    """Authenticate user"""
    if username in users:
        if users[username] ["password"] == hash_password(password):
            return True, users[username] ["role"]
    return False, None

# =========================
# PAGE SETUP & STYLING
# =========================
st.set_page_config(
    page_title="AI Reality Lab",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': "https://github.com/issues",
        'About': "# AI Reality Lab v2.0\nProduction-Grade ML Education"
    }
)

# Reddit-inspired dark theme
st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #030303;
    color: #d7dadc;
}

[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #1a1a1b 0%, #0d0d0e 100%);
    border-right: 2px solid #343536;
}

[data-testid="stSidebarContent"] {
    padding: 20px 15px;
}

.sidebar-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #343536;
}

.sidebar-header h1 {
    font-size: 24px;
    color: #818384;
    font-weight: 700;
    margin: 0;
}

.sidebar-logo {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #00f5c4, #0099ff);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    font-size: 20px;
}

.nav-section {
    margin-bottom: 30px;
}

.nav-section-title {
    font-size: 12px;
    font-weight: 700;
    color: #818384;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding-left: 10px;
}

[data-testid="stRadio"] {
    gap: 15px;
}

[data-testid="stRadio"] label {
    background-color: transparent !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: #818384 !important;
    font-weight: 500 !important;
    border-left: 3px solid transparent !important;
}

[data-testid="stRadio"] label:hover {
    background-color: #272729 !important;
    color: #d7dadc !important;
    border-left-color: #00f5c4 !important;
}

[data-testid="stRadio"] input:checked + label {
    background-color: #1a1a1b !important;
    color: #00f5c4 !important;
    border-left-color: #00f5c4 !important;
    box-shadow: inset 0 0 0 1px #00f5c4 !important;
}

h1 {
    color: #d7dadc !important;
    font-weight: 700 !important;
    font-size: 32px !important;
    margin-bottom: 20px !important;
}

h2 {
    color: #00f5c4 !important;
    font-weight: 700 !important;
    font-size: 24px !important;
    margin-top: 30px !important;
    margin-bottom: 15px !important;
}

h3 {
    color: #818384 !important;
    font-weight: 600 !important;
    font-size: 18px !important;
}

.stMetric {
    background: linear-gradient(135deg, #1a1a1b 0%, #272729 100%) !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #343536 !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.stMetric [data-testid="metricDeltaContainer"] {
    color: #00f5c4 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5c4 0%, #0099ff 100%) !important;
    color: #030303 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0, 245, 196, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(0, 245, 196, 0.5) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

.stSlider {
    padding: 15px 0 !important;
}

[data-testid="stSlider"] label {
    color: #818384 !important;
    font-weight: 600 !important;
}

.stTextInput > div > div > input,
.stPasswordInput > div > div > input {
    background-color: #272729 !important;
    color: #d7dadc !important;
    border: 1px solid #343536 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.stTextInput > div > div > input:focus,
.stPasswordInput > div > div > input:focus {
    border-color: #00f5c4 !important;
    box-shadow: 0 0 0 2px rgba(0, 245, 196, 0.2) !important;
}

.stSelectbox > div > div > select {
    background-color: #272729 !important;
    color: #d7dadc !important;
    border: 1px solid #343536 !important;
}

.stAlert {
    border-radius: 8px !important;
    padding: 16px !important;
    border: 1px solid #343536 !important;
}

.stSuccess {
    background-color: rgba(0, 245, 196, 0.1) !important;
    border-left: 4px solid #00f5c4 !important;
}

.stError {
    background-color: rgba(255, 68, 68, 0.1) !important;
    border-left: 4px solid #ff4444 !important;
}

.stWarning {
    background-color: rgba(255, 152, 0, 0.1) !important;
    border-left: 4px solid #ff9800 !important;
}

.stInfo {
    background-color: rgba(0, 153, 255, 0.1) !important;
    border-left: 4px solid #0099ff !important;
}

.stDataFrame {
    background-color: #1a1a1b !important;
}

[data-testid="stDataFrameContainer"] {
    background-color: #1a1a1b !important;
}

.stDivider {
    background-color: #343536 !important;
    margin: 30px 0 !important;
}

.stTabs [data-baseweb="tab-list"] button {
    background-color: transparent !important;
    color: #818384 !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #00f5c4 !important;
    border-bottom-color: #00f5c4 !important;
}

.metric-card {
    background: linear-gradient(135deg, #1a1a1b 0%, #272729 100%);
    border: 1px solid #343536;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
}

.user-badge {
    display: inline-block;
    background: linear-gradient(135deg, #00f5c4 0%, #0099ff 100%);
    color: #030303;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-left: 10px;
}

.admin-panel {
    background: linear-gradient(135deg, #1a1a1b 0%, #272729 100%);
    border: 2px solid #00f5c4;
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
}

.login-container {
    max-width: 400px;
    margin: 100px auto;
    background: linear-gradient(135deg, #1a1a1b 0%, #272729 100%);
    border: 1px solid #343536;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

.login-header {
    text-align: center;
    margin-bottom: 30px;
}

.login-header h1 {
    font-size: 28px;
    margin-bottom: 10px;
}

.login-header p {
    color: #818384;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE MANAGEMENT
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

if "model_params" not in st.session_state:
    st.session_state.model_params = {
        'rf_estimators': 100,
        'rf_depth': None,
        'xgb_learning_rate': 0.1,
        'test_size': 0.2
    }

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
def train_models(X_train, X_test, y_train, y_test, params):
    """Train all models with custom parameters"""
    
    # Linear Regression
    lin_model = LinearRegression().fit(X_train, y_train)
    y_train_pred_lin = lin_model.predict(X_train)
    y_test_pred_lin = lin_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=params['rf_estimators'],
        max_depth=params['rf_depth'],
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
            n_estimators=params['rf_estimators'],
            learning_rate=params['xgb_learning_rate'],
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
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

def display_metrics_cards(title, y_true, y_pred, model_name=""):
    """Display metrics in beautiful cards"""
    metrics = calculate_metrics(y_true, y_pred)
    
    st.markdown(f"### {title} {model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 MAE", f"${metrics['mae']:.2f}k", help="Mean Absolute Error")
    with col2:
        st.metric("📈 MSE", f"${metrics['mse']:.2f}k²", help="Mean Squared Error")
    with col3:
        st.metric("📉 RMSE", f"${metrics['rmse']:.2f}k", help="Root Mean Squared Error")
    with col4:
        st.metric("🎯 R²", f"{metrics['r2']:.4f}", help=f"Explains {metrics['r2']*100:.2f}% of variance")
    
    # Health indicator
    if metrics['r2'] < 0.5:
        st.error("⚠️ Model is weak — missing important variables")
    elif metrics['r2'] < 0.75:
        st.warning("⚠️ Model is decent but improvable")
    else:
        st.success("✅ Strong predictive performance")
    
    return metrics

# =========================
# LOGIN PAGE
# =========================
def show_login_page():
    """Display login interface"""
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>🚀 AI Reality Lab</h1>
                <p>Production-Grade ML Education</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔐 Password", type="password", placeholder="Enter password")
        
        col_login, col_demo = st.columns(2)
        
        with col_login:
            if st.button("🔓 Login", width='stretch'):
                users = load_users()
                auth, role = login_user(username, password, users)
                
                if auth:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = role
                    st.success(f"✅ Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        
        with col_demo:
            if st.button("👁️ Demo Mode", width='stretch'):
                st.session_state.authenticated = True
                st.session_state.username = "Demo User"
                st.session_state.role = "user"
                st.info("👁️ Entered demo mode")
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### 📝 Demo Credentials
        
        **Admin Account:**
        - Username: `admin`
        - Password: `admin123`
        
        **User Account:**
        - Username: `user`
        - Password: `user123`
        """)

# =========================
# ADMIN PANEL
# =========================
def show_admin_panel():
    """Display admin control panel"""
    st.markdown('<div class="admin-panel">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Admin Control Panel")
    
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["🎛️ Model Parameters", "👥 User Management", "📊 System Stats"])
    
    with admin_tab1:
        st.markdown("#### Adjust Model Hyperparameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rf_est = st.slider(
                "Random Forest Estimators",
                50, 500, st.session_state.model_params['rf_estimators'],
                step=50
            )
            st.session_state.model_params['rf_estimators'] = rf_est
        
        with col2:
            rf_depth = st.slider(
                "Random Forest Max Depth",
                5, 50, 20,
                step=5
            )
            st.session_state.model_params['rf_depth'] = rf_depth if rf_depth != 20 else None
        
        with col3:
            test_size = st.slider(
                "Test Set Size",
                0.1, 0.5, st.session_state.model_params['test_size'],
                step=0.05
            )
            st.session_state.model_params['test_size'] = test_size
        
        if xgb_available:
            xgb_lr = st.slider(
                "XGBoost Learning Rate",
                0.01, 0.5, st.session_state.model_params['xgb_learning_rate'],
                step=0.01
            )
            st.session_state.model_params['xgb_learning_rate'] = xgb_lr
        
        if st.button("💾 Save Parameters", width='stretch'):
            st.success("✅ Parameters saved!")
    
    with admin_tab2:
        st.markdown("#### User Management")
        
        users = load_users()
        
        st.write("**Current Users:**")
        user_df = pd.DataFrame([
            {"Username": k, "Role": v["role"]}
            for k, v in users.items()
        ])
        st.dataframe(user_df, use_container_width=True)
        
        st.divider()
        
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        new_role = st.selectbox("Role", ["user", "admin"])
        
        if st.button("➕ Add User", width='stretch'):
            if new_user and new_pass:
                if new_user not in users:
                    users[new_user] = {
                        "password": hash_password(new_pass),
                        "role": new_role
                    }
                    save_users(users)
                    st.success(f"✅ User '{new_user}' created!")
                else:
                    st.error("❌ User already exists")
    
    with admin_tab3:
        st.markdown("#### System Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Total Users", len(load_users()))
        
        with col2:
            st.metric("📅 Last Updated", datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        with col3:
            st.metric("🔧 Models Available", "3" if xgb_available else "2")
        
        st.markdown("**System Info:**")
        st.info(f"""
        - **Current User:** {st.session_state.username}
        - **Role:** {st.session_state.role.upper()}
        - **RF Estimators:** {st.session_state.model_params['rf_estimators']}
        - **Test Size:** {st.session_state.model_params['test_size']}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================
if not st.session_state.authenticated:
    show_login_page()
else:
    # Sidebar with beautiful design
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">🚀</div>
            <h1>AI Lab</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # User info
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1a1b 0%, #272729 100%); 
                    border: 1px solid #343536; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <p style="margin: 0; color: #818384; font-size: 12px;">LOGGED IN AS</p>
            <p style="margin: 5px 0 0 0; color: #d7dadc; font-weight: 700; font-size: 14px;">
                {st.session_state.username}
                <span class="user-badge">{st.session_state.role.upper()}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.markdown('<div class="nav-section-title">📍 Navigation</div>', unsafe_allow_html=True)
        
        page = st.radio("", [
            "🏠 Overview",
            "📊 Performance",
            "📈 Visualization",
            "🧠 Simulation",
            "📥 Reports",
        ], label_visibility="collapsed")
        
        st.divider()
        
        # Admin section
        if st.session_state.role == "admin":
            st.markdown('<div class="nav-section-title">⚙️ Admin</div>', unsafe_allow_html=True)
            if st.button("🛠️ Admin Panel", width='stretch'):
                st.session_state.show_admin = True
        
        st.divider()
        
        # Logout
        if st.button("🚪 Logout", width='stretch'):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.role = None
            st.rerun()
    
    # Load data
    df = load_data()
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=st.session_state.model_params['test_size'], random_state=42
    )
    
    models = train_models(X_train, X_test, y_train, y_test, st.session_state.model_params)
    
    # Show admin panel if requested
    if st.session_state.role == "admin" and st.session_state.get("show_admin", False):
        show_admin_panel()
        st.session_state.show_admin = False
    
    # =========================
    # PAGE 1: OVERVIEW
    # =========================
    if page == "🏠 Overview":
        st.title("🏠 AI Reality Lab")
        
        st.markdown("""
        ### Welcome to Production-Grade ML Education
        
        This platform demonstrates **real-world machine learning** with honest evaluation,
        beautiful visualizations, and interactive simulations.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Dataset Size", f"{len(df):,} samples")
        with col2:
            st.metric("🎯 Features", len(X.columns))
        with col3:
            st.metric("🤖 Models", "3" if xgb_available else "2")
        
        st.divider()
        
        st.subheader("🏠 Interactive House Price Predictor")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rm = st.slider("🏘️ Number of Rooms", 3.0, 9.0, 6.0, step=0.1)
        
        with col2:
            lstat = st.slider("📉 Lower Status %", 1.0, 40.0, 10.0, step=0.5)
        
        with col3:
            crim = st.slider("🚨 Crime Rate", 0.0, 20.0, 5.0, step=0.5)
        
        if st.button("🔮 Predict House Price", width='stretch'):
            input_data = X.mean().values.reshape(1, -1)
            input_df = pd.DataFrame(input_data, columns=X.columns)
            
            input_df["RM"] = rm
            input_df["LSTAT"] = lstat
            input_df["CRIM"] = crim
            
            st.markdown("### 🎯 Predictions Across Models")
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                lin_pred = float(models['lin_model'].predict(input_df))
                st.success(f"**Linear Regression**\n${lin_pred:.2f}k")
            
            with pred_col2:
                rf_pred = float(models['rf_model'].predict(input_df))
                st.success(f"**Random Forest**\n${rf_pred:.2f}k")
            
            with pred_col3:
                if xgb_available:
                    xgb_pred = float(models['xgb_model'].predict(input_df))
                    st.success(f"**XGBoost**\n${xgb_pred:.2f}k")
                else:
                    st.info("XGBoost not installed")
            
            avg_pred = np.mean([lin_pred, rf_pred])
            st.info(f"📊 **Average Prediction**: ${avg_pred:.2f}k")
    
    # =========================
    # PAGE 2: PERFORMANCE
    # =========================
    elif page == "📊 Performance":
        st.title("📊 Model Performance (Unseen Test Data)")
        
        st.markdown("""
        > **Why Test Set Matters**: These metrics are calculated on data the model has **never seen**.
        > This is your real-world performance indicator.
        """)
        
        # Linear Regression
        st.divider()
        st.subheader("🔵 Linear Regression")
        
        display_metrics_cards("Training Performance", y_train, models['y_train_pred_lin'], "📈")
        display_metrics_cards("Test Performance (Real)", y_test, models['y_test_pred_lin'], "🎯")
        
        # Random Forest
        st.divider()
        st.subheader("🟢 Random Forest")
        
        display_metrics_cards("Training Performance", y_train, models['y_train_pred_rf'], "📈")
        display_metrics_cards("Test Performance (Real)", y_test, models['y_test_pred_rf'], "🎯")
        
        # XGBoost
        if xgb_available:
            st.divider()
            st.subheader("🟡 XGBoost")
            
            display_metrics_cards("Training Performance", y_train, models['y_train_pred_xgb'], "📈")
            display_metrics_cards("Test Performance (Real)", y_test, models['y_test_pred_xgb'], "🎯")
        
        # Cross-validation
        st.divider()
        st.subheader("🔄 Cross-Validation (Most Reliable)")
        
        cv_col1, cv_col2, cv_col3 = st.columns(3)
        
        with cv_col1:
            cv_lin = cross_val_score(models['lin_model'], X, y, cv=5, scoring='r2')
            st.metric("🔵 Linear CV R²", f"{cv_lin.mean():.4f}", f"±{cv_lin.std():.4f}")
        
        with cv_col2:
            cv_rf = cross_val_score(models['rf_model'], X, y, cv=5, scoring='r2')
            st.metric("🟢 RF CV R²", f"{cv_rf.mean():.4f}", f"±{cv_rf.std():.4f}")
        
        with cv_col3:
            if xgb_available:
                cv_xgb = cross_val_score(models['xgb_model'], X, y, cv=5, scoring='r2')
                st.metric("🟡 XGBoost CV R²", f"{cv_xgb.mean():.4f}", f"±{cv_xgb.std():.4f}")
    
    # =========================
    # PAGE 3: VISUALIZATION
    # =========================
    elif page == "📈 Visualization":
        st.title("📈 Model Insights & Explainability")
        
        # Feature Importance
        st.subheader("🎯 Feature Importance (Random Forest)")
        
        importances = models['rf_model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        fig.patch.set_facecolor('#030303')
        ax.set_facecolor('#1a1a1b')
        ax.spines['bottom'].set_color('#343536')
        ax.spines['left'].set_color('#343536')
        ax.tick_params(colors='#818384')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(importance_df, use_container_width=True)
        
        # Actual vs Predicted
        st.divider()
        st.subheader("📊 Actual vs Predicted (Random Forest)")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#030303')
        
        for ax in axes:
            ax.set_facecolor('#1a1a1b')
            ax.spines['bottom'].set_color('#343536')
            ax.spines['left'].set_color('#343536')
            ax.tick_params(colors='#818384')
        
        # Training set
        axes.scatter(y_train, models['y_train_pred_rf'], alpha=0.6, c=y_train, cmap='viridis')
        axes.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        axes.set_xlabel('Actual Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_ylabel('Predicted Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_title('Training Set', fontsize=12, fontweight='bold', color='#d7dadc')
        axes.grid(True, alpha=0.2, color='#343536')
        
        # Test set
        scatter = axes.scatter(y_test, models['y_test_pred_rf'], alpha=0.6, c=y_test, cmap='viridis')
        axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes.set_xlabel('Actual Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_ylabel('Predicted Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_title('Test Set (Real Performance)', fontsize=12, fontweight='bold', color='#d7dadc')
        axes.grid(True, alpha=0.2, color='#343536')
        plt.colorbar(scatter, ax=axes, label='Price ($k)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Residual Analysis
        st.divider()
        st.subheader("📉 Residual Analysis")
        
        residuals_train = y_train - models['y_train_pred_rf']
        residuals_test = y_test - models['y_test_pred_rf']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor('#030303')
        
        for ax in axes:
            ax.set_facecolor('#1a1a1b')
            ax.spines['bottom'].set_color('#343536')
            ax.spines['left'].set_color('#343536')
            ax.tick_params(colors='#818384')
        
        # Training residuals
        axes.scatter(models['y_train_pred_rf'], residuals_train, alpha=0.6, color='#00f5c4')
        axes.axhline(y=0, color='#ff4444', linestyle='--', lw=2)
        axes.set_xlabel('Predicted Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_ylabel('Residuals ($k)', fontsize=11, color='#d7dadc')
        axes.set_title('Training Residuals', fontsize=12, fontweight='bold', color='#d7dadc')
        axes.grid(True, alpha=0.2, color='#343536')
        
        # Test residuals
        axes.scatter(models['y_test_pred_rf'], residuals_test, alpha=0.6, color='#0099ff')
        axes.axhline(y=0, color='#ff4444', linestyle='--', lw=2)
        axes.set_xlabel('Predicted Price ($k)', fontsize=11, color='#d7dadc')
        axes.set_ylabel('Residuals ($k)', fontsize=11, color='#d7dadc')
        axes.set_title('Test Residuals (Real Performance)', fontsize=12, fontweight='bold', color='#d7dadc')
        axes.grid(True, alpha=0.2, color='#343536')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # SHAP Explanations
        if shap_available:
            st.divider()
            st.subheader("🔍 SHAP Explainability (Why Predictions Happen)")
            
            st.info("SHAP shows which features push predictions up or down for each sample")
            
            try:
                with st.spinner("Generating SHAP explanations..."):
                    explainer = shap.Explainer(models['rf_model'], X_test)
                    shap_values = explainer(X_test[:100])
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    shap.plots.beeswarm(shap_values, show=False)
                    fig.patch.set_facecolor('#030303')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"SHAP visualization error: {e}")
        else:
            st.info("📦 Install SHAP for explainability: `pip install shap`")
    
    # =========================
    # PAGE 4: SIMULATION
    # =========================
    elif page == "🧠 Simulation":
        st.title("🧠 Reality + Business Simulation")
        
        st.markdown("""
        ### Why Real-World ML is Different
        
        This section shows how models behave when assumptions break down.
        """)
        
        # Generate synthetic marketing data
        np.random.seed(42)
        marketing_spend = np.random.uniform(0, 100, 200)
        sales = 50 + 10*np.log1p(marketing_spend) + np.random.normal(0, 2, 200)
        
        sim_df = pd.DataFrame({
            "Marketing Spend": marketing_spend,
            "Sales": sales
        })
        
        sim_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(
            sim_df[["Marketing Spend"]],
            sim_df["Sales"]
        )
        
        st.subheader("📢 Marketing Spend → Sales Prediction")
        
        spend = st.slider("💰 Marketing Spend ($k)", 0, 100, 20)
        
        input_df_sim = pd.DataFrame({"Marketing Spend": [spend]})
        pred_sales_raw = sim_model.predict(input_df_sim)
        pred_sales = float(pred_sales_raw)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("💵 Predicted Sales", f"${pred_sales:.2f}k")
        
        with col2:
            st.metric("💰 Spend", f"${spend}k")
        
        with col3:
            roi = (pred_sales - spend) / spend * 100 if spend > 0 else 0
            st.metric("📈 ROI", f"{roi:.1f}%")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor('#030303')
        ax.set_facecolor('#1a1a1b')
        
        ax.scatter(marketing_spend, sales, alpha=0.5, s=50, label='Actual Data', color='#00f5c4')
        
        spend_range = np.linspace(0, 100, 100).reshape(-1, 1)
        predictions = sim_model.predict(spend_range)
        ax.plot(spend_range, predictions, color='#0099ff', linewidth=3, label='Model Prediction')
        
        ax.scatter([spend], [pred_sales], color='#ff9800', s=300, marker='*', 
                   label=f'Current Prediction (${spend}k)', zorder=5, edgecolors='white', linewidth=2)
        
        ax.set_xlabel('Marketing Spend ($k)', fontsize=12, fontweight='bold', color='#d7dadc')
        ax.set_ylabel('Sales ($k)', fontsize=12, fontweight='bold', color='#d7dadc')
        ax.set_title('Marketing Spend → Sales (Non-Linear Relationship)', fontsize=14, fontweight='bold', color='#d7dadc')
        ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.2, color='#343536')
        ax.spines['bottom'].set_color('#343536')
        ax.spines['left'].set_color('#343536')
        ax.tick_params(colors='#818384')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        st.subheader("💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 🔑 Why This Matters
            
            1. **Diminishing Returns** 📉
               - More spend ≠ proportional sales
            
            2. **Non-Linear Relationships** 🌊
               - Real world rarely follows straight lines
            
            3. **Noise & Variance** 🎲
               - Random variation exists
            
            4. **Model Limitations** 🤖
               - Can't perfectly predict future
            """)
        
        with col2:
            st.markdown("""
            #### 🎯 Business Implications
            
            1. **Test Different Levels** 🧪
               - Don't assume linear scaling
            
            2. **Monitor Performance** 📊
               - Track actual vs predicted
            
            3. **Adjust Strategy** 🔄
               - Based on real ROI
            
            4. **Iterate & Learn** 📚
               - Continuous improvement
            """)
        
        st.divider()
        st.subheader("🚀 Future Evolution")
        
        tabs = st.tabs(["📈 Advanced Modeling", "🔍 Causal Inference", "🧠 Explainability", "🏢 Production"])
        
        with tabs:
            st.markdown("""
            - Time-series forecasting (sales over weeks/months)
            - Seasonal decomposition
            - Trend analysis & forecasting
            - ARIMA & Prophet models
            """)
        
        with tabs:
            st.markdown("""
            - A/B testing frameworks
            - Uplift modeling (who to target)
            - Causal graphs & DAGs
            - Propensity score matching
            """)
        
        with tabs:
            st.markdown("""
            - SHAP interaction plots
            - Feature dependency analysis
            - Counterfactual explanations
            - Local interpretable models
            """)
        
        with tabs:
            st.markdown("""
            - Live data pipelines
            - API deployment
            - Real-time predictions
            - Model monitoring & retraining
            """)
    
    # =========================
    # PAGE 5: REPORTS
    # =========================
    elif page == "📥 Reports":
        st.title("📥 Generate & Download Reports")
        
        st.markdown("### Create comprehensive reports of your model evaluation")
        
        report_type = st.selectbox("Select Report Type", [
            "📊 Full Evaluation Report",
            "📈 Performance Comparison",
            "🎯 Feature Analysis",
            "📉 Residual Analysis"
        ])
        
        if st.button("🔨 Generate Report", width='stretch'):
            with st.spinner("Generating report..."):
                
                if report_type == "📊 Full Evaluation Report":
                    fig = plt.figure(figsize=(16, 20))
                    fig.patch.set_facecolor('#030303')
                    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
                    
                    # Title
                    fig.suptitle('AI Reality Lab — Complete Model Evaluation Report', 
                                fontsize=20, fontweight='bold', y=0.995, color='#d7dadc')
                    
                    # Metrics Summary
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
                    
                    ax1.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
                            verticalalignment='center', 
                            bbox=dict(boxstyle='round', facecolor='#272729', alpha=0.8, edgecolor='#00f5c4', linewidth=2))
                    
                    # Actual vs Predicted (Linear)
                    ax2 = fig.add_subplot(gs[1, 0])
                    ax2.set_facecolor('#1a1a1b')
                    ax2.scatter(y_test, models['y_test_pred_lin'], alpha=0.6, c='#00f5c4')
                    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax2.set_xlabel('Actual Price ($k)', color='#d7dadc')
                    ax2.set_ylabel('Predicted Price ($k)', color='#d7dadc')
                    ax2.set_title('Linear Regression: Actual vs Predicted', color='#d7dadc', fontweight='bold')
                    ax2.grid(True, alpha=0.2, color='#343536')
                    ax2.tick_params(colors='#818384')
                    
                    # Actual vs Predicted (RF)
                    ax3 = fig.add_subplot(gs[1, 1])
                    ax3.set_facecolor('#1a1a1b')
                    ax3.scatter(y_test, models['y_test_pred_rf'], alpha=0.6, c='#0099ff')
                    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax3.set_xlabel('Actual Price ($k)', color='#d7dadc')
                    ax3.set_ylabel('Predicted Price ($k)', color='#d7dadc')
                    ax3.set_title('Random Forest: Actual vs Predicted', color='#d7dadc', fontweight='bold')
                    ax3.grid(True, alpha=0.2, color='#343536')
                    ax3.tick_params(colors='#818384')
                    
                    # Residuals (Linear)
                    ax4 = fig.add_subplot(gs[2, 0])
                    ax4.set_facecolor('#1a1a1b')
                    residuals_lin = y_test - models['y_test_pred_lin']
                    ax4.scatter(models['y_test_pred_lin'], residuals_lin, alpha=0.6, c='#00f5c4')
                    ax4.axhline(y=0, color='#ff4444', linestyle='--', lw=2)
                    ax4.set_xlabel('Predicted Price ($k)', color='#d7dadc')
                    ax4.set_ylabel('Residuals ($k)', color='#d7dadc')
                    ax4.set_title('Linear Regression: Residuals', color='#d7dadc', fontweight='bold')
                    ax4.grid(True, alpha=0.2, color='#343536')
                    ax4.tick_params(colors='#818384')
                    
                    # Residuals (RF)
                    ax5 = fig.add_subplot(gs[2, 1])
                    ax5.set_facecolor('#1a1a1b')
                    residuals_rf = y_test - models['y_test_pred_rf']
                    ax5.scatter(models['y_test_pred_rf'], residuals_rf, alpha=0.6, c='#0099ff')
                    ax5.axhline(y=0, color='#ff4444', linestyle='--', lw=2)
                    ax5.set_xlabel('Predicted Price ($k)', color='#d7dadc')
                    ax5.set_ylabel('Residuals ($k)', color='#d7dadc')
                    ax5.set_title('Random Forest: Residuals', color='#d7dadc', fontweight='bold')
                    ax5.grid(True, alpha=0.2, color='#343536')
                    ax5.tick_params(colors='#818384')
                    
                    # Feature Importance
                    ax6 = fig.add_subplot(gs[3, :])
                    ax6.set_facecolor('#1a1a1b')
                    importances = models['rf_model'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(10)
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
                    ax6.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
                    ax6.set_xlabel('Importance Score', color='#d7dadc', fontweight='bold')
                    ax6.set_title('Top 10 Feature Importance (Random Forest)', color='#d7dadc', fontweight='bold')
                    ax6.invert_yaxis()
                    ax6.tick_params(colors='#818384')
                    
                elif report_type == "📈 Performance Comparison":
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.patch.set_facecolor('#030303')
                    
                    models_list = ['Linear', 'RF']
                    train_r2 = [r2_score(y_train, models['y_train_pred_lin']), 
                               r2_score(y_train, models['y_train_pred_rf'])]
                    test_r2 = [r2_score(y_test, models['y_test_pred_lin']), 
                              r2_score(y_test, models['y_test_pred_rf'])]
                    
                    if xgb_available:
                        models_list.append('XGBoost')
                        train_r2.append(r2_score(y_train, models['y_train_pred_xgb']))
                        test_r2.append(r2_score(y_test, models['y_test_pred_xgb']))
                    
                    # R² Comparison
                    ax = axes[0, 0]
                    ax.set_facecolor('#1a1a1b')
                    x = np.arange(len(models_list))
                    width = 0.35
                    ax.bar(x - width/2, train_r2, width, label='Train', color='#00f5c4')
                    ax.bar(x + width/2, test_r2, width, label='Test', color='#0099ff')
                    ax.set_ylabel('R² Score', color='#d7dadc', fontweight='bold')
                    ax.set_title('R² Score Comparison', color='#d7dadc', fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_list)
                    ax.legend()
                    ax.grid(True, alpha=0.2, color='#343536', axis='y')
                    ax.tick_params(colors='#818384')
                    
                    # RMSE Comparison
                    ax = axes[0, 1]
                    ax.set_facecolor('#1a1a1b')
                    train_rmse = [np.sqrt(mean_squared_error(y_train, models['y_train_pred_lin'])),
                                 np.sqrt(mean_squared_error(y_train, models['y_train_pred_rf']))]
                    test_rmse = [np.sqrt(mean_squared_error(y_test, models['y_test_pred_lin'])),
                                np.sqrt(mean_squared_error(y_test, models['y_test_pred_rf']))]
                    
                    if xgb_available:
                        train_rmse.append(np.sqrt(mean_squared_error(y_train, models['y_train_pred_xgb'])))
                        test_rmse.append(np.sqrt(mean_squared_error(y_test, models['y_test_pred_xgb'])))
                    
                    x = np.arange(len(models_list))
                    ax.bar(x - width/2, train_rmse, width, label='Train', color='#ff9800')
                    ax.bar(x + width/2, test_rmse, width, label='Test', color='#ff4444')
                    ax.set_ylabel('RMSE ($k)', color='#d7dadc', fontweight='bold')
                    ax.set_title('RMSE Comparison', color='#d7dadc', fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_list)
                    ax.legend()
                    ax.grid(True, alpha=0.2, color='#343536', axis='y')
                    ax.tick_params(colors='#818384')
                    
                    # MAE Comparison
                    ax = axes[1, 0]
                    ax.set_facecolor('#1a1a1b')
                    train_mae = [mean_absolute_error(y_train, models['y_train_pred_lin']),
                                mean_absolute_error(y_train, models['y_train_pred_rf'])]
                    test_mae = [mean_absolute_error(y_test, models['y_test_pred_lin']),
                               mean_absolute_error(y_test, models['y_test_pred_rf'])]
                    
                    if xgb_available:
                        train_mae.append(mean_absolute_error(y_train, models['y_train_pred_xgb']))
                        test_mae.append(mean_absolute_error(y_test, models['y_test_pred_xgb']))
                    
                    x = np.arange(len(models_list))
                    ax.bar(x - width/2, train_mae, width, label='Train', color='#00f5c4')
                    ax.bar(x + width/2, test_mae, width, label='Test', color='#0099ff')
                    ax.set_ylabel('MAE ($k)', color='#d7dadc', fontweight='bold')
                    ax.set_title('MAE Comparison', color='#d7dadc', fontweight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(models_list)
                    ax.legend()
                    ax.grid(True, alpha=0.2, color='#343536', axis='y')
                    ax.tick_params(colors='#818384')
                    
                    # Summary Table
                    ax = axes[1, 1]
                    ax.axis('off')
                    
                    summary_data = []
                    for i, model_name in enumerate(models_list):
                        summary_data.append([
                            model_name,
                            f"{train_r2[i]:.4f}",
                            f"{test_r2[i]:.4f}",
                            f"{train_rmse[i]:.2f}",
                            f"{test_rmse[i]:.2f}"
                        ])
                    
                    table = ax.table(cellText=summary_data,
                                    colLabels=['Model', 'Train R²', 'Test R²', 'Train RMSE', 'Test RMSE'],
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0, 0, 1, 1])
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 2)
                    
                    for i in range(len(summary_data) + 1):
                        for j in range(5):
                            cell = table[(i, j)]
                            cell.set_facecolor('#272729')
                            cell.set_text_props(color='#d7dadc')
                            if i == 0:
                                cell.set_facecolor('#00f5c4')
                                cell.set_text_props(color='#030303', weight='bold')
                
                elif report_type == "🎯 Feature Analysis":
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.patch.set_facecolor('#030303')
                    
                    # Feature Importance
                    ax = axes[0, 0]
                    ax.set_facecolor('#1a1a1b')
                    importances = models['rf_model'].feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
                    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
                    ax.set_xlabel('Importance', color='#d7dadc', fontweight='bold')
                    ax.set_title('Feature Importance (All)', color='#d7dadc', fontweight='bold')
                    ax.invert_yaxis()
                    ax.tick_params(colors='#818384')
                    
                    # Top 5 Features
                    ax = axes[0, 1]
                    ax.set_facecolor('#1a1a1b')
                    top5 = importance_df.head(5)
                    colors = plt.cm.plasma(np.linspace(0, 1, 5))
                    ax.pie(top5['Importance'], labels=top5['Feature'], autopct='%1.1f%%',
                          colors=colors, textprops={'color': '#d7dadc'})
                    ax.set_title('Top 5 Features Distribution', color='#d7dadc', fontweight='bold')
                    
                    # Feature Statistics
                    ax = axes[1, 0]
                    ax.axis('off')
                    
                    stats_text = "FEATURE STATISTICS\n\n"
                    for feat in X.columns[:5]:
                        stats_text += f"{feat}:\n"
                        stats_text += f"  Mean: {X[feat].mean():.2f}\n"
                        stats_text += f"  Std: {X[feat].std():.2f}\n"
                        stats_text += f"  Min: {X[feat].min():.2f}\n"
                        stats_text += f"  Max: {X[feat].max():.2f}\n\n"
                    
                    ax.text(0.1, 0.9, stats_text, fontsize=9, family='monospace',
                           verticalalignment='top', color='#d7dadc',
                           bbox=dict(boxstyle='round', facecolor='#272729', alpha=0.8, edgecolor='#00f5c4'))
                    
                    # Correlation Heatmap
                    ax = axes[1, 1]
                    ax.set_facecolor('#1a1a1b')
                    
                    corr_matrix = X.corr()
                    im = ax.imshow(corr_matrix, cmap='viridis', aspect='auto')
                    ax.set_xticks(range(len(X.columns)))
                    ax.set_yticks(range(len(X.columns)))
                    ax.set_xticklabels(X.columns, rotation=45, ha='right', fontsize=8, color='#818384')
                    ax.set_yticklabels(X.columns, fontsize=8, color='#818384')
                    ax.set_title('Feature Correlation Matrix', color='#d7dadc', fontweight='bold')
                    plt.colorbar(im, ax=ax, label='Correlation')
                
                else:  # Residual Analysis
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.patch.set_facecolor('#030303')
                    
                    residuals_lin = y_test - models['y_test_pred_lin']
                    residuals_rf = y_test - models['y_test_pred_rf']
                    
                    # Residuals Distribution (Linear)
                    ax = axes[0, 0]
                    ax.set_facecolor('#1a1a1b')
                    ax.hist(residuals_lin, bins=30, color='#00f5c4', alpha=0.7, edgecolor='white')
                    ax.set_xlabel('Residuals ($k)', color='#d7dadc', fontweight='bold')
                    ax.set_ylabel('Frequency', color='#d7dadc', fontweight='bold')
                    ax.set_title('Linear Regression: Residual Distribution', color='#d7dadc', fontweight='bold')
                    ax.grid(True, alpha=0.2, color='#343536', axis='y')
                    ax.tick_params(colors='#818384')
                    
                    # Residuals Distribution (RF)
                    ax = axes[0, 1]
                    ax.set_facecolor('#1a1a1b')
                    ax.hist(residuals_rf, bins=30, color='#0099ff', alpha=0.7, edgecolor='white')
                    ax.set_xlabel('Residuals ($k)', color='#d7dadc', fontweight='bold')
                    ax.set_ylabel('Frequency', color='#d7dadc', fontweight='bold')
                    ax.set_title('Random Forest: Residual Distribution', color='#d7dadc', fontweight='bold')
                    ax.grid(True, alpha=0.2, color='#343536', axis='y')
                    ax.tick_params(colors='#818384')
                    
                    # Q-Q Plot (Linear)
                    ax = axes[1, 0]
                    ax.set_facecolor('#1a1a1b')
                    from scipy import stats
                    stats.probplot(residuals_lin, dist="norm", plot=ax)
                    ax.set_title('Linear Regression: Q-Q Plot', color='#d7dadc', fontweight='bold')
                    ax.tick_params(colors='#818384')
                    
                    # Q-Q Plot (RF)
                    ax = axes[1, 1]
                    ax.set_facecolor('#1a1a1b')
                    stats.probplot(residuals_rf, dist="norm", plot=ax)
                    ax.set_title('Random Forest: Q-Q Plot', color='#d7dadc', fontweight='bold')
                    ax.tick_params(colors='#818384')
                
                plt.tight_layout()
                
                # Save to bytes
                pdf_buffer = BytesIO()
                plt.savefig(pdf_buffer, format='pdf', dpi=300, bbox_inches='tight', facecolor='#030303')
                pdf_buffer.seek(0)
                
                st.success("✅ Report generated successfully!")
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer,
                    file_name=f"AI_Reality_Lab_{report_type.split()}_Report.pdf",
                    mime="application/pdf",
                    width='stretch'
                )
                
                plt.close(fig)
        
        st.divider()
        st.subheader("📊 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export Metrics as CSV", width='stretch'):
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
                    label="📄 Download CSV",
                    data=csv_buffer,
                    file_name="model_metrics.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
                st.dataframe(metrics_export, use_container_width=True)
        
        with col2:
            if st.button("📋 Export Feature Importance", width='stretch'):
                importances = models['rf_model'].feature_importances_
                importance_export = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                csv_buffer = importance_export.to_csv(index=False)
                
                st.download_button(
                    label="📋 Download CSV",
                    data=csv_buffer,
                    file_name="feature_importance.csv",
                    mime="text/csv",
                    width='stretch'
                )
                
                st.dataframe(importance_export, use_container_width=True)
