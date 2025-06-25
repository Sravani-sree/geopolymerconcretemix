import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Simulated dummy model loader (replace with your trained inverse models)
def load_inverse_models():
    models = {}
    for target in ["Molarity", "Alkali/Precursor", "Silicate/Hydroxide", "Water/Solids"]:
        model = RandomForestRegressor(n_estimators=10)
        X_dummy = np.random.rand(20, 3)
        y_dummy = np.random.rand(20)
        model.fit(X_dummy, y_dummy)
        models[target] = model
    return models

models_inverse = load_inverse_models()

# 💡 Conversion logic to material quantities
def calculate_materials(predicted, precursor_kg=450, ggbs_ratio=0.3):
    alkali_ratio = predicted['Alkali/Precursor']
    silicate_ratio = predicted['Silicate/Hydroxide']
    water_solid = predicted['Water/Solids']

    ggbs = precursor_kg * ggbs_ratio
    alkali_total = precursor_kg * alkali_ratio
    na2sio3 = alkali_total * silicate_ratio / (1 + silicate_ratio)
    naoh = alkali_total - na2sio3
    solids = precursor_kg + ggbs + naoh + na2sio3
    water = water_solid * solids
    aggregates = max(0, 2300 - (precursor_kg + ggbs + naoh + na2sio3 + water))

    return {
        "Precursor (Flyash/Metakaolin)": round(precursor_kg, 2),
        "GGBS": round(ggbs, 2),
        "NaOH": round(naoh, 2),
        "Na₂SiO₃": round(na2sio3, 2),
        "Water": round(water, 2),
        "Aggregates (Fine + Coarse)": round(aggregates, 2)
    }

# 💰 Optional cost estimation
def estimate_cost(materials, cost_per_kg):
    total_cost = 0
    cost_breakdown = {}
    for mat, qty in materials.items():
        cost = qty * cost_per_kg.get(mat, 0)
        cost_breakdown[mat] = round(cost, 2)
        total_cost += cost
    cost_breakdown["Total Cost (₹/m³)"] = round(total_cost, 2)
    return cost_breakdown

# 🌐 Streamlit App UI
st.title("Geopolymer Concrete Mix Design Assistant")
st.markdown("Provide target **performance requirements** to predict mix design and cost:")

cs28 = st.number_input("🧱 Compressive Strength (CS28 in MPa)", 10.0, 60.0, 30.0)
sf = st.number_input("💧 Slump Flow (SF in mm)", 300.0, 800.0, 500.0)
t500 = st.number_input("⏱️ Flow Time (T500 in sec)", 1.0, 20.0, 10.0)

if st.button("🔍 Predict Mix Design"):
    input_data = np.array([[cs28, sf, t500]])
    predictions = {}
    for target, model in models_inverse.items():
        pred = model.predict(input_data)[0]
        predictions[target] = round(pred, 3)

    st.subheader("🎯 Predicted Mix Ratios:")
    st.write(pd.DataFrame(predictions.items(), columns=["Parameter", "Predicted Value"]))

    # Estimate material quantities
    materials = calculate_materials(predictions)
    st.subheader("🧱 Estimated Material Quantities (kg/m³):")
    st.write(pd.DataFrame(materials.items(), columns=["Material", "Amount (kg/m³)"]))

    # Add cost breakdown
    cost_per_kg = {
        "Precursor (Flyash/Metakaolin)": 4.5,
        "GGBS": 3.5,
        "NaOH": 35.0,
        "Na₂SiO₃": 20.0,
        "Water": 0.01,
        "Aggregates (Fine + Coarse)": 0.5
    }
    cost_breakdown = estimate_cost(materials, cost_per_kg)
    st.subheader("💰 Estimated Material Cost (₹ per m³):")
    st.write(pd.DataFrame(cost_breakdown.items(), columns=["Material", "Cost (₹)"]))

    # Show pie chart of materials
    st.subheader("📊 Material Composition (Pie Chart)")

    df_pie = pd.DataFrame(materials.items(), columns=["Material", "Amount (kg/m³)"])
    fig = px.pie(df_pie, names="Material", values="Amount (kg/m³)",
                 title="Material Distribution per m³",
                 hole=0.3)
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


