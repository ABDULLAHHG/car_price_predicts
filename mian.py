import pandas as pd
import streamlit as st 

df = pd.read_csv('data.csv')

# Columns that need to clean 
columns_to_clean = df.drop('gearbox',axis = 1)
columns_to_clean = columns_to_clean.drop('brand',axis = 1)
columns_to_clean = columns_to_clean.drop('voivodeship',axis = 1)
columns_to_clean = columns_to_clean.drop('city',axis = 1)
columns_to_clean = columns_to_clean.drop('model',axis = 1)
columns_to_clean.mileage = columns_to_clean.mileage.astype(str)

# Cleaned dataframe 
df = df[df.mileage.str.contains(r'km')]

# Show dataframe 
st.dataframe(df)

