import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open('breastcancer.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

# Giving Title
st.title('BREAST CANCER PREDICTION')

# getting the input data from the user
mean_radius = st.number_input("Enter Mean Radius")
mean_texture = st.number_input('Enter Mean Texture')
mean_perimeter = st.number_input('Enter Mean Perimeter')
mean_area = st.number_input('Enter Mean Area')
mean_smoothness = st.number_input('Enter Mean Smoothness')

# Creating a button for prediction
if st.button('Test Results'):
    classifier = loaded_model.predict(sc.transform([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]]))

    if classifier[0] == 0:
        st.write("YOU ARE NOT AFFECTED BY BREAST CANCER")
        st.write("You are safe 😃 Keep Smiling")
    else:
        st.write("YOU ARE AFFECTED BY BREAST CANCER")
        st.write("Consult the Doctor! Be Strong 💪 ") 

st.write("NOTE: This is only for Educational Purpose")
st.markdown("<h1 style='font-size: 12px;'>Founder: Santhosh Reddy Padala</h1>", unsafe_allow_html=True)
