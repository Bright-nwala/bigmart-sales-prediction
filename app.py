import os
import requests
import streamlit as st
import pandas as pd
import cloudpickle

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒")

# -------------------------------
# Constants
# -------------------------------
MODEL_URL = "https://github.com/Bright-nwala/bigmart-sales-prediction/releases/download/v2/bigmart_sales_pipeline.pkl"
MODEL_PATH = "bigmart_model.pkl"

# -------------------------------
# Utility: Download & Load Model
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Download model from GitHub release if missing, then load it."""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model file from GitHub Releases…")
        try:
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    try:
        with open(MODEL_PATH, "rb") as f:
            pipeline = cloudpickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load the model: {e}")
        st.stop()

# Load the pipeline
pipeline = load_model()
st.success("✅ Model pipeline loaded")

# -------------------------------
# UI: App Header
# -------------------------------
st.title("🛒 BigMart Sales Prediction App")
st.markdown("Upload a CSV file with the **same features used during training** to get predicted sales.")

# -------------------------------
# File Uploader + Prediction
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(input_df.head())

        # Predict
        preds = pipeline.predict(input_df)
        output = input_df.copy()
        output["Predicted_Item_Outlet_Sales"] = preds

        st.subheader("📈 Predicted Sales Output")
        st.dataframe(output.head(50))

        # Download button
        st.download_button(
            "📥 Download Predictions as CSV",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="predicted_sales.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.markdown("Double-check your CSV columns match the training schema.")
else:
    st.info("👈 Upload a CSV file to begin.")






