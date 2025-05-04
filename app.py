import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model/improved_xgboost_model.joblib")

# Page configuration
st.set_page_config(page_title="💧 Water Potability Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    # .main {
    #     background-color: #f9fbfc;
    #     padding: 2rem;
    #     border-radius: 15px;
    #     box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    # }
    .title {
        font-size: 38px;
        font-weight: 800;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #004d99;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 🔹 Title
# -------------------------
st.markdown('<div class="title">💧 Water Potability Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="main">', unsafe_allow_html=True)

# -------------------------
# 🔍 Manual Prediction
# -------------------------
st.header("🔍 Predict a Single Sample")

with st.form("manual_input"):
    col1, col2, col3 = st.columns(3)
    ph = col1.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
    hardness = col2.number_input("Hardness", min_value=0.0, value=100.0)
    solids = col3.number_input("Solids (ppm)", min_value=0.0, value=10000.0)

    col4, col5, col6 = st.columns(3)
    chloramines = col4.number_input("Chloramines", min_value=0.0, value=7.0)
    sulfate = col5.number_input("Sulfate", min_value=0.0, value=300.0)
    conductivity = col6.number_input("Conductivity", min_value=0.0, value=400.0)

    col7, col8, col9 = st.columns(3)
    organic_carbon = col7.number_input("Organic Carbon", min_value=0.0, value=10.0)
    trihalomethanes = col8.number_input("Trihalomethanes", min_value=0.0, value=70.0)
    turbidity = col9.number_input("Turbidity", min_value=0.0, value=4.0)

    submitted = st.form_submit_button("Predict Potability")

    if submitted:
        sample = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                organic_carbon, trihalomethanes, turbidity]],
                              columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                                       'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])
        result = model.predict(sample)[0]
        st.markdown("---")
        if result == 1:
            st.success("✅ The water is **POTABLE** (Safe to drink).")
        else:
            st.error("❌ The water is **NOT POTABLE** (Unsafe to drink).")

# -------------------------
# 📂 CSV Batch Prediction
# -------------------------
st.markdown("---")
st.header("📂 Upload CSV for Batch Prediction")

file = st.file_uploader("Upload a CSV File (Must match model features)", type=["csv"])
if file:
    try:
        data = pd.read_csv(file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(data.head())

        # Predictions
        preds = model.predict(data)
        data["Prediction"] = ["✅ Potable" if p == 1 else "❌ Not Potable" for p in preds]

        st.success("🎉 Predictions Completed!")
        st.dataframe(data)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction Results", csv, "potability_results.csv", "text/csv")

    except Exception as e:
        st.error(f"❗ Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)
