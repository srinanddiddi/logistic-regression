import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    return df


df_raw = load_data()
df = df_raw.copy()

# -----------------------------
# REMOVE CUSTOMER ID
# -----------------------------
if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# -----------------------------
# PREPROCESSING
# -----------------------------
# convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())

# binary encoding
binary_cols = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]
for col in binary_cols:
    if col == "gender":
        df[col] = df[col].replace({"Male": 1, "Female": 0})
    else:
        df[col] = df[col].replace({"Yes": 1, "No": 0})

# Force Target Column to integer to resolve "unknown label type"
df["Churn"] = df["Churn"].astype(int)

# one hot encoding
categorical_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# convert boolean columns to int
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

# -----------------------------
# SPLIT DATA
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -----------------------------
# TRAIN MODEL (Cached to run only once)
# -----------------------------
@st.cache_resource
def train_logistic_model(X_tr, y_tr):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    return model


model = train_logistic_model(X_train, y_train)


# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_churn(input_data):
    input_df = pd.DataFrame([input_data])
    # convert TotalCharges
    input_df["TotalCharges"] = pd.to_numeric(
        input_df["TotalCharges"], errors="coerce"
    )
    input_df["TotalCharges"] = input_df["TotalCharges"].fillna(
        df_raw["TotalCharges"].apply(pd.to_numeric, errors="coerce").mean()
    )

    # binary encoding
    input_df["gender"] = input_df["gender"].replace({"Male": 1, "Female": 0})
    binary_input_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_input_cols:
        input_df[col] = input_df[col].replace({"Yes": 1, "No": 0})

    # one hot encoding
    input_df = pd.get_dummies(input_df)
    # match training columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # prediction
    prediction = model.predict(input_df)
    return prediction[0]


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=0)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict"):
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    prediction = predict_churn(input_data)
    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
