import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sklearn as sklearn
import pickle

image = Image.open("img.jpg")
st.image(image,width=100)
st.title('Car Price Prediction')

mlmodel = st.selectbox('Please select the maschine learning model',['LineerRegression','RandomForest','XGBoost'])


brand = st.radio("Select a brand", ("Audi", "Renault", "Opel"))
if brand == 'Audi':
    model = st.radio('Select a model',('A1','A3'))
elif brand == 'Renault':
    model = st.radio('Select a model',('Clio','Duster','Espace'))
elif brand == 'Opel':
    model = st.radio('Select a model',('Astra','Corsa','Insignia'))


km = st.sidebar.number_input('Km:',0,500000,10000,1000)
age = st.sidebar.number_input('Age:',1,4,1,1)
gear = st.sidebar.radio('Select gearing type',('Automatic','Manual','Semi-automatic'))
kw = st.sidebar.number_input('Select kw',40,300,50,1)
my_dict = {
    "hp_kw": kw,
    "age": age,
    "km": km,
    "make_model": brand + '_' + model,
    "gearing_type": gear
}

my_dict = pd.DataFrame([my_dict])


columns=[
 'hp_kw',
 'km',
 'age',
 'make_model_Audi_A1',
 'make_model_Audi_A3',
 'make_model_Opel_Astra',
 'make_model_Opel_Corsa',
 'make_model_Opel_Insignia',
 'make_model_Renault_Clio',
 'make_model_Renault_Duster',
 'make_model_Renault_Espace',
 'gearing_type_Automatic',
 'gearing_type_Manual',
 'gearing_type_Semi-automatic']

from sklearn.preprocessing import OrdinalEncoder            
from sklearn.compose import make_column_transformer


my_dict_dummy = pd.get_dummies(my_dict).reindex(columns=columns, fill_value=0)

if mlmodel == 'XGBoost':
    filename2 = "xgb_final.pkl"
    model = pickle.load(open(filename2, "rb"))
    pred = model.predict(my_dict_dummy)
elif mlmodel == 'RandomForest':
    cat = ['make_model', 'gearing_type']
    enc = OrdinalEncoder()
    my_dict_rf = my_dict
    my_dict_rf[cat] = enc.fit_transform(my_dict_rf[cat])
    filename3 = 'rf1_final.pkl'
    model = pickle.load(open(filename3, "rb"))
    pred = model.predict(my_dict) 
else:
    final_scaler = pickle.load(open('scaler_reg.pkl', "rb"))
    my_dict_scaled = final_scaler.transform(my_dict_dummy)
    filename = "reg_final.pkl"
    model = pickle.load(open(filename, "rb"))
    pred = model.predict(my_dict_scaled)
    

if st.button('Predict'):
    st.success(f'The estimated price of your car is {int(pred[0])}')









