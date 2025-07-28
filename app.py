import streamlit as st
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Load model and scalers
model = joblib.load("rf_model.pkl")
input_scaler = joblib.load("input_scaler.pkl")
output_scaler = joblib.load("output_scaler.pkl")

# Target input section
st.title("🧪 Inverse Design of SCGPC Concrete Mix")
st.markdown("Enter desired target properties:")

cs_target = st.number_input("Compressive Strength (MPa)", min_value=10.0, max_value=80.0, value=40.0)
sf_target = st.number_input("Slump Flow (mm)", min_value=400.0, max_value=800.0, value=700.0)
t500_target = st.number_input("T500 Time (sec)", min_value=1.0, max_value=20.0, value=5.0)

target_values = np.array([[cs_target, sf_target, t500_target]])
scaled_target = output_scaler.transform(target_values)

# Bounds for input mix proportions
bounds = [
    (100, 600),   # Fly Ash
    (50, 250),    # GGBS
    (5, 50),      # NaOH
    (8, 16),      # Molarity
    (50, 300),    # Silicate Solution
    (600, 800),   # Sand
    (800, 1200),  # Coarse Agg
    (120, 220),   # Water
    (0, 10),      # SP
    (20, 80),     # Temperature
]

input_labels = ['Fly Ash', 'GGBS', 'NaOH', 'Molarity', 'Silicate Solution', 'Sand',
                'Coarse Agg', 'Water', 'SP', 'Temperature']

# Define fitness function
def fitness(x):
    x_scaled = input_scaler.transform([x])
    y_pred_scaled = model.predict(x_scaled)
    error = np.mean((y_pred_scaled - scaled_target)**2)
    return error

if st.button("🔍 Optimize Mix Design"):
    result = differential_evolution(fitness, bounds, seed=42, strategy='best1bin', maxiter=1000)
    optimal_mix = result.x
    predicted_scaled = model.predict(input_scaler.transform([optimal_mix]))
    predicted_output = output_scaler.inverse_transform(predicted_scaled)[0]

    mix_df = pd.DataFrame({
        "Component": input_labels,
        "Proportion": np.round(optimal_mix, 2)
    })

    # Display result table
    st.subheader("📊 Suggested Mix Design Proportions")
    st.dataframe(mix_df.set_index("Component"))

    # Pie chart
    st.subheader("🔎 Mix Proportion Visualization")
    fig, ax = plt.subplots()
    ax.pie(mix_df["Proportion"], labels=mix_df["Component"], autopct='%1.1f%%', startangle=140)
    ax.axis("equal")
    st.pyplot(fig)

    # Predicted outputs
    st.subheader("📈 Model-Predicted Properties for Optimized Mix")
    st.markdown(f"""
    - **C Strength**: {predicted_output[0]:.2f} MPa  
    - **S Flow**:     {predicted_output[1]:.2f} mm  
    - **T 500**:      {predicted_output[2]:.2f} sec
    """)
