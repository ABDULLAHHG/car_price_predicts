import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

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
df = df[df.engine_capacity.str.match(r'[0-9]')]

# prepare dataframe 
df.engine_capacity = df.engine_capacity.apply(lambda x :x.replace('cm3','').replace(' ','')).astype(int)
df.mileage = df.mileage.apply(lambda x :x.replace('km','').replace(' ','')).astype(int)
df.model = df.model.str.split().str[1]

# create dummy variables for categorical variables
df = df.join(pd.get_dummies(df.gearbox))
df = df.join(pd.get_dummies(df.fuel_type))
df = df.join(pd.get_dummies(df.brand))

# create dummy variables for categorical variables
df = df.drop('gearbox',axis = 1)
df = df.drop('fuel_type',axis = 1)
df = df.drop('brand',axis = 1)
df = df.drop('voivodeship',axis = 1)
df = df.drop('city',axis = 1)
df = df.drop('model',axis = 1)

# build model 
X = df.drop('price_in_pln' ,axis = 1)
y = df['price_in_pln']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

# Show dataframe 
st.dataframe(df)
st.text(mean_absolute_error(y_test,y_pred))
