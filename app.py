
import streamlit as st
import pandas as pd
import pickle

st.title("🏠 House Price Prediction - Linear Regression")

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open("linear_model.pkl", "rb"))

model = load_model()

# Input fields - adapt to your actual model features
area = st.number_input("Total Living Area (GrLivArea):", min_value=500, max_value=10000, value=1500)
bedrooms = st.slider("Number of Bedrooms Above Ground:", 1, 10, 3)
garage = st.slider("Garage Cars Capacity:", 0, 4, 2)

# Input dataframe for prediction
input_data = pd.DataFrame([[area, bedrooms, garage]], columns=["GrLivArea", "BedroomAbvGr", "GarageCars"])

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏷️ Predicted Sale Price: ${int(prediction):,}")
