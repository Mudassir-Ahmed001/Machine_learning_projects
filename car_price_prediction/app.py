from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np
import streamlit as st
import pandas as pd
import pickle
#streamlit run app.py


#title
st.title('Car Price Predictor')

#loading the data
company = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/car_price_prediction/company.pkl', 'rb')) #rb=ReadBinary
car_model = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/car_price_prediction/car_model.pkl', 'rb')) #rb=ReadBinary
kms_driven = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/car_price_prediction/kms_driven.pkl', 'rb')) #rb=ReadBinary
fuel_type = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/car_price_prediction/fuel_type.pkl', 'rb')) #rb=ReadBinary
year = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/car_price_prediction/year.pkl', 'rb')) #rb=ReadBinary

#loading the model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))




company_selected = st.selectbox(
    'Choose the Company',
    (company)
)
car_model_selected = st.selectbox(
    'Choose the model',
    (car_model)
)
kms_driven_selected = st.text_input(
    'Kilometers driven',
    ''
)

fuel_type_selected = st.selectbox(
    'Select the Fuel_type',
    ('Petrol', 'Diesel', 'LPG')
)
year_selected = st.selectbox(
    'Select the year',
    (year)
)


#creating a function for predicting
def car_price_prediction():
    final_prediction = loaded_model.predict(pd.DataFrame([[car_model_selected, company_selected, year_selected, kms_driven_selected, fuel_type_selected]],
                                                         columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    st.write(f'The Price Of The Car is:- {final_prediction}')


if st.button('Show the Price'):
    car_price_prediction()
else:
    pass
