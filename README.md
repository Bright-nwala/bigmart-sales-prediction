# BigMart Sales Streamlit App

This project is a simple Streamlit app that predicts item outlet sales using a trained machine learning pipeline.

## 📦 Project Structure

```
bigmart_project/
│
├── app.py                      # Streamlit app
├── bigmart_sales_pipeline.pkl # Trained ML pipeline (copy here manually)
└── README.md                  # This file
```

## 🚀 How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the app:

```
streamlit run app.py
```

3. Upload a CSV with the same format used in training and get predictions!

---

## ⚠️ Note
Make sure `bigmart_sales_pipeline.pkl` (downloaded from Colab) is placed in this folder.