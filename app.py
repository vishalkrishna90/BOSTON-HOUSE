from re import S
import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import pickle as pkl
import warnings as warning
warning.filterwarnings('ignore')

model = pkl.load(open('xgb_model.pkl','rb'))
scaler = pkl.load(open('scaler.pkl','rb'))


url = "https://github.com/vishalkrishna90/BOSTON-HOUSE/blob/main/boston.csv" 
download = requests.get(url).content

df = pd.read_csv(io.StringIO(download.decode('utf-8')),sep='\t')


st.title('BOSTON HOUSE PRICE PREDICTION')
st.image('Images/BOSTON_1.jpg')


ls,rm,di = st.columns(3)
with ls:
    LSTAT = st.slider('Select Lstat:', 1.00,40.0)
with rm:
    RM = st.slider('Select No Of Rooms', 1,10)
with di:
    DIS = st.slider('Select Distance',1.0,15.0)

ro,pt,cr = st.columns(3)
with ro:
    NOX = st.slider('Select Nox',0.0,1.0)
with pt:
    PTRATIO =  st.slider('Teacher Ratio', 0.10,30.0)
with cr:
    CRIM = st.slider('Crime Rate',0.0,100.0)

if st.button('Submit'):
    data = np.array([[LSTAT,RM,DIS,NOX,PTRATIO,CRIM]])
    scale_data = scaler.transform(data)
    res = model.predict(scale_data)
    st.subheader(f'Predicted House Price Is : {res[0]}')
