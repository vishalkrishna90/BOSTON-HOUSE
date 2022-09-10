from re import S
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
import warnings as warning
warning.filterwarnings('ignore')

model = pkl.load(open('xgb_model.pkl','rb'))
scaler = pkl.load(open('scaler.pkl','rb'))

df = pd.read_csv('D:\END_TO_END_PROJECT\BOSTON-HOUSE\\boston.csv')

st.title('Boston House Price Prediction')

LSTAT = st.slider('Select Lstat:', 1.00,40.0)
RM = st.slider('Select No Of Rooms', 1,10)
DIS = st.slider('Select Distance',1.0,15.0)
NOX = st.slider('Select Nox',0.0,1.0)
PTRATIO =  st.slider('Teacher Ratio', 0.10,30.0)
CRIM = st.slider('Crime Rate',0.0,100.0)

if st.button('Submit'):
    data = np.array([[LSTAT,RM,DIS,NOX,PTRATIO,CRIM]])
    scale_data = scaler.transform(data)
    res = model.predict(scale_data)
    st.subheader(f'Predicted House Price Is : {(res[0])}')
