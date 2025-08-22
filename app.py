import os
import requests
import streamlit as st
import pandas as pd
import cloudpickle
import urllib.request

MODEL_URL = "https://github.com/your-username/bigmart-sales-prediction/releases/download/v1.0/bigmart_model.pkl"
MODEL_PATH = "bigmart_model.pkl"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load model
with open(MODEL_PATH, "rb") as f:
    model = cloudpickle.load(f)


st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒")

st.title("🛒 BigMart Sales Prediction App")
st.markdown("Upload a CSV file with the **same features used during training** to get predicted sales.")

# Download model if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    try:
        st.info("📥 Downloading model file from GitHub Releases…")
        r = requests.get(MODEL_URL, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load the model with cloudpickle
try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = cloudpickle.load(f)
    st.success("✅ Model pipeline loaded")
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()

# File uploader
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

        # Download
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





