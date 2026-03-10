import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Setup Page
st.set_page_config(page_title="Boston Housing Portal")

@st.cache_data
def load_data():
    # Fetching raw data due to sklearn deprecation
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    df = pd.DataFrame(data, columns=cols)
    df['MEDV'] = target
    return df

df = load_data()
model = LinearRegression().fit(df.drop(columns=['MEDV']), df['MEDV'])

st.title("🏙️ Boston Housing Price Predictor")
st.markdown("---")

# User Inputs
st.subheader("Neighborhood Features")
col1, col2 = st.columns(2)

with col1:
    rm = st.slider("Average Rooms (RM)", 3.0, 9.0, 6.0)
    crim = st.slider("Crime Rate (CRIM)", 0.0, 90.0, 3.0)

with col2:
    lstat = st.slider("Lower Status Population %", 1.0, 40.0, 10.0)
    tax = st.slider("Property Tax Rate", 180.0, 711.0, 400.0)

if st.button("Predict House Value"):
    # Build input array using means for other values
    input_data = [df[col].mean() for col in df.columns[:-1]]
    input_df = pd.DataFrame([input_data], columns=df.columns[:-1])
    # Apply user selections
    input_df['RM'] = rm
    input_df['CRIM'] = crim
    input_df['LSTAT'] = lstat
    input_df['TAX'] = tax
    
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Price: ${prediction:.2f}k")