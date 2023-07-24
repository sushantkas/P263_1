import pandas as pd
import streamlit as st
import numpy as np
from sklearn import *
import pickle


with open("lightgbm_model.pkl", "rb") as file:
        model=pickle.load(file)

#image="https://previews.123rf.com/images/nosua/nosua1609/nosua160900599/63815755-electric-motor-in-disassembled-state-3d-illustration-on-a-white-background.jpg"
image="https://media.istockphoto.com/id/1128669675/vector/3d-engine-contour.jpg?s=612x612&w=0&k=20&c=wwHUGCHgJVd5E3zZ1vG7BTpS3Essmf07MtX2GIejRco="
def add_bg_from_url(link):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url({link});
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url(image)



st.title("Motor Speed Prediction :racing_motorcycle:")

st.header("Enter Input to Predict Motor Speed")
st.subheader("Data must be standardized")

ambient=st.number_input("Enter Ambient temprature of Motor")

coolant=st.number_input("Enter Coolant temprature for Motor")

u_d= st.number_input("Enter Voltage d Component (u_d)")

u_q= st.number_input("Enter Voltage q Component (u_q)")

torque=st.number_input("Enter Torque of Motor")

i_d=st.number_input("Enter Current d Component (i_d)")

i_q=st.number_input("Enter Current q Component (i_q)")

pm=st.number_input("Enter Permanent Magnet value of Motor (PM)")

stator_yoke=st.number_input("Enter Stator yoke Temprature")

stator_tooth=st.number_input("Enter Stator Tooth Temprature")

stator_winding=st.number_input("Enter Stator Winding Temprature")

profile_id=st.selectbox("Select Profile id",(4,  6, 10, 11, 20, 27, 29, 30, 31, 32, 36, 41, 42, 43, 44, 45, 46,
       47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
       64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81,
       72), help="Profile is Unique Measurement session Select Appropriate session for Accuracy")


if st.button("Predict"):
    data=pd.DataFrame({"ambient":ambient,"coolant":coolant,"u_d":u_d,"u_q":u_q,"torque":torque,"i_d":i_d,"i_q":i_q,"pm":pm,"stator_yoke":stator_yoke,"stator_tooth":stator_tooth,"stator_winding":stator_winding, "profile_id":profile_id}, index=[0])
    st.write(data)
    st.write(model.predict(data))

new_d = st.file_uploader("Choose a csv file", help="Columns names must be 'ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id'")

column=('ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id')


if st.button("Read and Predict"):
    try:
        prediction=pd.read_csv(new_d)
        if "Unnamed: 0" in prediction.columns:
            prediction.drop("Unnamed: 0", axis=1, inplace=True)
        if prediction.shape[1]!=12:
            st.write(f"Uploaded File has {prediction.shape[1]} file must have 12 columns ")
            st.write(f"Columns should be 'ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d','i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding','profile_id'")
        else:
            st.write(prediction)
            st.write(f"FIle uploaded Successfully. File has {prediction.shape[0]} Rows and {prediction.shape[1]} Columns")
            prediction["motor_speed"]=model.predict(prediction)
            st.write("Model has Successfully predicted Motor Speed for given input dataset Please download file to view it")
            #st.download_button("Click to download file",prediction,file_name="Predicted Motor Speed.csv",mime='text/csv')
    except:
        #prediction=pd.read_csv(new_d)
        st.write("Unable to read file please check the file format...")

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(prediction)

    st.download_button(
        label="Click to download file",
        data=csv,
        file_name='Predicted Motor Speed.csv',
        mime='text/csv',
    )

#st.download_button("Click to download file",data=prediction,file_name="Predicted Motor Speed.csv")
