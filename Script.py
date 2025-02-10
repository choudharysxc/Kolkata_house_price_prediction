import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

# Define the file path
file_path = "Copy of cleaned_dataset(1).xlsx"

# Load the dataset
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name='Sheet1')

df = load_data(file_path)

# Encode 'city' column using Label Encoding
label_encoder = LabelEncoder()
if 'city' in df.columns:
    df["city_encoded"] = label_encoder.fit_transform(df["city"]) + 1
    df = df.drop(columns=["city"])

# Define features and target
categorical_columns = ["property_type"]
numerical_columns = [col for col in df.columns if col not in categorical_columns + ["Price in L", "facing"]]

# Fill missing values
df[numerical_columns] = df[numerical_columns].apply(lambda col: col.fillna(col.median()))
df[categorical_columns] = df[categorical_columns].apply(lambda col: col.fillna(col.mode()[0]))

# One-Hot Encode 'property_type'
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Combine processed categorical and numerical features
X = pd.concat([encoded_df, df[numerical_columns]], axis=1)
y = df["Price in L"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting Model
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluate model accuracy
y_pred = gb_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Streamlit UI
st.title("Real Estate Price Prediction")
st.sidebar.header("Enter Property Details")

# User inputs
bedroom_num = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
furnish = st.sidebar.radio("Furnished", [0, 1])
age = st.sidebar.slider("Age of Property (Years)", 0, 50, 5)
total_floor = st.sidebar.number_input("Total Floors", min_value=1, max_value=50, value=10)
price_sqft = st.sidebar.number_input("Price per Sq. Ft.", min_value=1000, max_value=20000, value=4500)
area_sqft = st.sidebar.number_input("Total Area (Sq. Ft.)", min_value=500, max_value=5000, value=1200)
BHK = st.sidebar.number_input("BHK", min_value=1, max_value=10, value=3)
parking = st.sidebar.radio("Parking Available?", [0, 1])
metro_stations = st.sidebar.number_input("Metro Stations Nearby", min_value=0, max_value=10, value=1)
schools = st.sidebar.number_input("Schools Nearby", min_value=0, max_value=10, value=1)
hospitals = st.sidebar.number_input("Hospitals Nearby", min_value=0, max_value=10, value=1)
connectivity = st.sidebar.radio("Good Connectivity?", [0, 1])
shopping = st.sidebar.radio("Nearby Shopping Centers?", [0, 1])
pharmacy = st.sidebar.radio("Nearby Pharmacy?", [0, 1])

# Prepare input data
input_data = {
    "property_type_Residential Apartment": 1,
    "bedroom_num": bedroom_num,
    "furnish": furnish,
    "age": age,
    "total_floor": total_floor,
    "price_sqft": price_sqft,
    "Area sqft": area_sqft,
    "BHK": BHK,
    "parking": parking,
    "metro_stations": metro_stations,
    "schools": schools,
    "hospitals": hospitals,
    "connectivity": connectivity,
    "shopping": shopping,
    "pharmacy": pharmacy
}

def predict_price(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_df.fillna(0, inplace=True)
    input_scaled = scaler.transform(input_df)
    predicted_price = gb_model.predict(input_scaled)
    return np.round(predicted_price[0], 2)

if st.sidebar.button("Predict Price"):
    predicted_price = predict_price(input_data)
    st.success(f"Estimated Property Price: ₹{predicted_price:.2f} Lakhs")

st.write(f"Model Accuracy (R² Score): {r2:.4f}")
st.write(f"Mean Absolute Error (MAE): ₹{mae:.2f} Lakhs")
