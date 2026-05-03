import streamlit as st
import pandas as pd
import joblib

model = joblib.load("final_model.pkl")

st.title("SuperKart Sales Prediction")

product_weight = st.number_input("Product Weight", 1.0)
product_mrp = st.number_input("Product MRP", 100.0)
allocated_area = st.number_input("Allocated Area", 0.05)
store_year = st.number_input("Store Year", 2005)

product_sugar = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])

product_type = st.selectbox("Product Type", [
    "Fruits and Vegetables","Snack Foods","Household","Frozen Foods",
    "Dairy","Canned","Baking Goods","Health and Hygiene",
    "Meat","Soft Drinks","Breads","Hard Drinks",
    "Others","Starchy Foods","Breakfast","Seafood"
])

store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

input_data = pd.DataFrame({
    "Product_Id": ["FD000"],
    "Product_Weight": [product_weight],
    "Product_Sugar_Content": [product_sugar],
    "Product_Allocated_Area": [allocated_area],
    "Product_Type": [product_type],
    "Product_MRP": [product_mrp],
    "Store_Id": ["OUT000"],
    "Store_Establishment_Year": [store_year],
    "Store_Size": [store_size],
    "Store_Location_City_Type": [city_type],
    "Store_Type": [store_type]
})

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {pred:.2f}")
