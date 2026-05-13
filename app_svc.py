import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------
# TITLE
# ---------------------------------------
st.title("SVM Classification App")

st.write("This app demonstrates Support Vector Machine (SVC) Classification using the Iris dataset.")

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# dataframe
df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

st.subheader("Dataset")
st.dataframe(df.head())

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------
# FEATURE SCALING
# ---------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------
# SIDEBAR PARAMETERS
# ---------------------------------------
st.sidebar.header("SVM Parameters")

kernel = st.sidebar.selectbox(
    "Kernel",
    ["linear", "rbf", "poly", "sigmoid"]
)

C_value = st.sidebar.slider(
    "C Value",
    min_value=0.1,
    max_value=10.0,
    value=1.0
)

gamma_value = st.sidebar.selectbox(
    "Gamma",
    ["scale", "auto"]
)

degree_value = st.sidebar.slider(
    "Degree (for poly kernel)",
    min_value=1,
    max_value=5,
    value=3
)

# ---------------------------------------
# MODEL
# ---------------------------------------
model = SVC(
    kernel=kernel,
    C=C_value,
    gamma=gamma_value,
    degree=degree_value,
    shrinking=True,
    tol=0.001,
    cache_size=200,
    class_weight=None
)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# ---------------------------------------
# EVALUATION
# ---------------------------------------
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Evaluation Metrics")

st.write("Accuracy Score:", accuracy)

st.subheader("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------------------------------
# USER INPUT
# ---------------------------------------
st.subheader("Make Prediction")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_class = target_names[prediction[0]]

    st.success(f"Predicted Class: {predicted_class}")

# ---------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.subheader("Actual vs Predicted")
st.dataframe(results.head(10))