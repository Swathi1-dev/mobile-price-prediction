import pickle
import streamlit as st 
import numpy as np
import pandas as pd
import os 
import tensorflow as tf
from tensorflow.keras.models import load_model


st.title("Mobile Price Prediction App")

#load the model 

model=load_model("mobile_price_model.keras")

with open("scaler.pkl","rb")as f:
    scaler=pickle.load(f)
    
#input features

battery_power=st.number_input("Battery Power(mAh)",min_value=500,max_value=20000,value=1000,step=100)
blue=st.number_input("Bluetooth(0/1)",min_value=0,max_value=1,value=1,step=1)
clock_speed=st.number_input("Clock Speed(GHz)",min_value=0.1,max_value=20.0,value=1.0,step=0.1)
dual_sim =st.number_input("Dual SIM(0/1)",min_value=0,max_value=1,value=1,step=1)
fc=st.number_input("Front Camera(MP)",min_value=0,max_value=50,value=5,step=1)
four_g=st.number_input("4G Support(0/1)",min_value=0,max_value=1,value=1,step=1)
int_memory=st.number_input("Internal Memory(GB)",min_value=1,max_value=512,value=32,step=1)
m_dep=st.number_input("Mobile Depth(cm)",min_value=0.1,max_value=2.0,value=0.5,step=0.1)
mobile_wt=st.number_input("Mobile Weight(g)",min_value=50,max_value=500,value=150,step=1)
n_cores=st.number_input("Number of Cores",min_value=1,max_value=8,value=4,step=1)
pc=st.number_input("Pixel Count",min_value=10000,max_value=100000,value=50000,step=1)
px_height=st.number_input("Pixel Height",min_value=1,max_value=200,value=15,step=1)
px_width=st.number_input("Pixel Width",min_value=1,max_value=200,value=15,step=1)
ram=st.number_input("RAM(GB)",min_value=1,max_value=128,value=4,step=1)
sc_h = st.number_input("Screen Height(cm)",min_value=5,max_value=25,value=15,step=1)
sc_w = st.number_input("Screen Width(cm)",min_value=5,max_value=25,value=15,step=1)
talk_time = st.number_input("Talk Time(hours)",min_value=2,max_value=24,value=8,step=1)
three_g = st.number_input("3G Support(0/1)",min_value=0,max_value=1,value=1,step=1)
touch_screen = st.number_input("Touch Screen(0/1)",min_value=0,max_value = 1,value = 1, step = 1)
wifi = st.number_input("WiFi(0/1)",min_value = 0, max_value = 1, value = 1, step = 1)

input_data=np.array([[battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,
                      m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time,
                        three_g,touch_screen,wifi]])

scaled_data=scaler.transform(input_data)
if st.button("Predict Price Range"):
    prediction=model.predict(scaled_data).argmax(axis=1)
    price_range=["Low","Medium","High","Very High"]
    st.success(f"The predicted price range of the mobile is: {price_range[prediction[0]]}")
