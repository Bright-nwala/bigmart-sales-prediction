import pandas as pd
import numpy as np
import cloudpickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('big_mart_sales.csv')

# Select features
features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
            'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
df = data[features + ['Item_Outlet_Sales']].copy()

# Handle missing values
df['Item_Weight'].fillna(df['Item_Weight'].median(), inplace=True)
df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(lambda x: x.fillna(x.mode()[0]))

# Define feature types
num_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
cat_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

# Define preprocessing
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Split data
X = df[features]
y = df['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(max_depth=10, n_estimators=50, random_state=0))
])

# Train pipeline
pipeline.fit(X_train, y_train)
print("✅ Model training complete.")

# Evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📊 Test RMSE: {rmse:.2f}")

# Save pipeline
with open('bigmart_sales_pipeline.pkl', 'wb') as f:
    cloudpickle.dump(pipeline, f)

print("💾 Pipeline saved as 'bigmart_sales_pipeline.pkl'")

