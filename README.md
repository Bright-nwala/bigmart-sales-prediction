# 🛒 BigMart Sales Prediction – End-to-End Machine Learning Project

This project predicts the sales of products across various BigMart outlets using historical sales data. It includes data cleaning, feature engineering, model evaluation, pipeline creation, and an interactive Streamlit web app for predictions.

---

## 🎯 Objective

Build a machine learning pipeline that predicts `Item_Outlet_Sales` using product and outlet features. The trained model is reusable and served through a web app to allow predictions on new data.

---

## 📁 Dataset

- **Source**: [Kaggle - BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
- **Target**: `Item_Outlet_Sales`
- **Features**:
  - `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`, `Item_Type`, `Item_MRP`
  - `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`

---

## ⚙️ Tech Stack

- **Language**: Python 3.11+
- **Libraries**:
  - `pandas`, `numpy`, `scikit-learn`, `joblib`, 'cloudpickle' , `matplotlib`, `seaborn`
  - `streamlit` for frontend deployment
- **Dev Tools**: Google Colab, VS Code
- **Model Serving**: Streamlit App (CSV upload → predictions)

---

## 🧪 ML Workflow

1. **Data Cleaning**:
   - Fill missing `Item_Weight` with median
   - Fill missing `Outlet_Size` using mode by `Outlet_Type`

2. **Feature Engineering**:
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features

3. **Model Training**:
   - Trained models:
     - `LinearRegression`
     - `DecisionTreeRegressor`
     - ✅ `RandomForestRegressor` (final model)
   - Evaluated using cross-validation (RMSE)

4. **Pipeline**:
   - Preprocessing + model packed in a single `Pipeline`
   - Saved with `cloudpickle` for production use

5. **Deployment**:
   - Built a Streamlit app that:
     - Accepts CSV input
     - Predicts `Item_Outlet_Sales`
     - Displays and downloads results

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Bright-nwala/bigmart-sales-prediction.git
cd bigmart-sales-prediction

2. Set up virtual environment (optional)
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Launch Streamlit App
streamlit run app.py

📂 Project Structure
bigmart-sales-prediction/
│
├── app.py                        # Streamlit app
├── big_mart_sales.csv           # Dataset (if needed for reference)
├── train_model.py               # Training script (pipeline creation)
├── bigmart_sales_pipeline.pkl   # Saved ML pipeline
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation

💡 Future Enhancements
Hyperparameter tuning (GridSearchCV)

Try advanced models (XGBoost, LightGBM)

Add interactive input form (instead of CSV)

Include SHAP/feature importance explanation

API deployment with FastAPI or Flask

👤 Author
Bright Nwala

💼 GitHub: Bright-nwala

🌐 Portfolio: https://brightnwala.pythonanywhere.com/

📫 LinkedIn: www.linkedin.com/in/bright-nwala-cz77


