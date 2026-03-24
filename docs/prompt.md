# it obvioulsy started elsewhere but that is not important

```bash
You are a senior software engineer about to explain artificial intelligence and have chosen linear regression as an entry point : This code <import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

# Setup Page
st.set_page_config(page_title="Boston Housing Portal", layout="wide")

@st.cache_data
def load_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    df = pd.DataFrame(data, columns=cols)
    df['MEDV'] = target
    return df

# Load data and train model
df = load_data()
model = LinearRegression().fit(df.drop(columns=['MEDV']), df['MEDV'])

# Define y_true and predict
y_true = df['MEDV']
y_pred = model.predict(df.drop(columns=['MEDV']))

# Calculate metrics
sse = ((y_true - y_pred) ** 2).sum()
sst = ((y_true - y_true.mean()) ** 2).sum()
r2 = r2_score(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred, squared=False)

# ======== MAIN UI ========

st.title("🏙️ Boston Housing Price Predictor")
st.markdown("---")

# 📊 Model Performance Card
st.subheader("📊 Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"${rmse:.2f}k", "Average prediction error")
col2.metric("R²", f"{r2:.3f}", "% variance explained")
col3.metric("SSE", f"{sse:.2f}", "Sum of squared errors")

# 📘 Manual Example Card
st.subheader("📘 Manual Example: How Linear Regression Works")
st.info("""
Let’s say we have 3 houses:

| Actual Price (y) | Predicted (ŷ) | Error (y - ŷ) | Squared Error |
|----------------|-------------|-------------|-------------|
| 30             | 28          | +2          | 4           |
| 35             | 37          | -2          | 4           |
| 40             | 36          | +4          | 16          |

→ **SSE = 4 + 4 + 16 = 24**  
→ **SST = (30-35)² + (35-35)² + (40-35)² = 50**  
→ **R² = 1 - (24/50) = 0.52** → “52% of variation explained”  
→ **RMSE = √(24/3) ≈ 2.83** → “Average error = $2.83k”

✅ RMSE is actionable. R² is abstract.
""")

# 📈 Visualization: Actual vs Predicted
st.subheader("📈 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predictions')
ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2, label='Perfect Fit')
ax.set_xlabel("Actual Price ($k)")
ax.set_ylabel("Predicted Price ($k)")
ax.set_title("Linear Regression: Actual vs Predicted")
ax.legend()
st.pyplot(fig)

# 🏠 User Prediction Interface
st.subheader("🏠 Predict Your House Price")
col1, col2 = st.columns(2)

with col1:
    rm = st.slider("Average Rooms (RM)", 3.0, 9.0, 6.0)
    crim = st.slider("Crime Rate (CRIM)", 0.0, 90.0, 3.0)

with col2:
    lstat = st.slider("Lower Status Population %", 1.0, 40.0, 10.0)
    tax = st.slider("Property Tax Rate", 180.0, 711.0, 400.0)

if st.button("Predict House Value"):
    input_data = [df[col].mean() for col in df.columns[:-1]]
    input_df = pd.DataFrame([input_data], columns=df.columns[:-1])
    input_df['RM'] = rm
    input_df['CRIM'] = crim
    input_df['LSTAT'] = lstat
    input_df['TAX'] = tax
    
    prediction = model.predict(input_df)
    st.success(f"### 🏠 Estimated Price: ${prediction:.2f}k")>, these are the requirements <streamlit
pandas
numpy
scikit-learn
matplotlib>, and these are the errors gotten <Matplotlib is building the font cache; this may take a moment.
2026-03-24 12:36:29.980 Uncaught app execution
Traceback (most recent call last):
  File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
             ^^^^^^
  File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lsetga/Projects/pyhon1/app.py", line 34, in <module>
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lsetga/Projects/pyhon1/env/lib/python3.12/site-packages/sklearn/utils/_param_validation.py", line 196, in wrapper
    params = func_sig.bind(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/inspect.py", line 3242, in bind
    return self._bind(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/inspect.py", line 3231, in _bind
    raise TypeError(
TypeError: got an unexpected keyword argument 'squared'
>, i left out the versions so pip would choose the most compatible from the requiements, i was wondering if evrything can stay in one file but still have 4 streamlit ui pages, and was wondering if you could add any other recommendation like how linear regression really only works on a theoratical world of robots where linearity can be achieved but humans are too messy as even salary increase would not normally be based on years worked and education level alone -- connections, softs skills, value derived, etc can also play outlier level roles -- similar to the other types of regression, random forest , etc -- only xdgboost and neural nets come close -- please confirm this first and include as printed out commenst in one of the pages -- be brutally honest.
