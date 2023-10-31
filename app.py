import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

from sklearn.ensemble import RandomForestClassifier

st.title("Medical Diagnostic Web App")
st.subheader("Does the patient have diabetes?")
df=pd.read_csv('diabetes.csv')
if st.sidebar.checkbox('View Data',False):
    st.write(df)

if st.sidebar.checkbox('View Distributions',False):
    df.hist()
    plt.tight_layout()
    st.pyplot()
    
# Step 1: load pickled model
model=open('best_rfc.pickle','rb')
clf=pickle.load(model)
model.close()

# Step 2: get the front end user input
pregs=st.number_input('Pregnancies',0,17,0)
glucose=st.slider('Glucose',44.000,199.00,117.0000)
bp=st.slider('BloodPressure',24,122,72)
skin=st.slider('SkinThickness',7,99,23)
insulin=st.slider('Insulin',14,846,31)
bmi=st.slider('BMI',18.2,67.1,32.0)
diaped=st.slider('DiabetesPedigreeFunction',0.05,2.42,0.372)
age=st.slider('Age',21,81,29)


# Step 3: Get the model input
input_data=[[pregs,glucose,bp,skin,insulin,bmi,diaped,age]]

# Step 4: get the predictions and print the results
prediction=clf.predict(input_data)[0]
if st.button('Predict'):
    if prediction==0:
        st.subheader("Non diabetic")
    else:
        st.subheader("Diabetic")
