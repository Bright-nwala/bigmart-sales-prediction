import streamlit as st
import pandas as pd
import cloudpickle
import os
st.write("Current files in repo:", os.listdir("."))


# Load trained pipeline
try:
    with open('bigmart_sales_pipeline.pkl', 'rb') as f:
        pipeline = cloudpickle.load(f)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    model_loaded = False

# Streamlit App
st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒")
st.title("🛒 BigMart Sales Prediction App")
st.markdown("Upload a CSV file with the **same features used during training** to get predicted sales.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file and model_loaded:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("📊 Preview of Uploaded Data")
        st.dataframe(input_df.head())

        # Make predictions
        predictions = pipeline.predict(input_df)
        input_df["Predicted_Item_Outlet_Sales"] = predictions

        st.subheader("📈 Predicted Sales Output")
        st.dataframe(input_df)

        # Allow user to download predictions
        csv = input_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predicted_sales.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.markdown("Make sure your CSV has **all required columns** used in training.")
elif not uploaded_file:
    st.info("👈 Upload a CSV file to begin.")



