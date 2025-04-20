import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the model, scalers, encoders
model = tf.keras.models.load_model('model.keras')
with open('label_encoder_gender.pkl', 'rb') as f:
    gender = pickle.load(f)
with open('one_hot_encoder_geo.pkl', 'rb') as f: 
    geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#streamlit app
st.title('Customer Churn Prediction')

#user input
st.subheader('Customer Information')
age = st.slider('Age', min_value=18, max_value=100)
gendersel = st.selectbox('Gender', gender.classes_)
geography = st.selectbox('Geography', geo.categories_[0])
credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
balance = st.number_input('Balance', min_value=0.0, max_value=1e6)
num_of_products = st.slider('Number of Products', min_value=1, max_value=4)
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, max_value=1e6)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
tenure = st.slider('Tenure', min_value=0, max_value=10)


# prepare tghe input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender':[gendersel],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':    [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]

})

# encode categorical features
geo_encoded = geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_data.drop(columns=['Geography']), geo_encoded_df], axis=1)
input_df_scaled = scaler.transform(input_df)

predictions = model.predict(input_df_scaled)
predictions_prob = predictions[0][0]

st.write("Predicted Probability of Churn: ", predictions_prob)
if predictions_prob > 0.5:
    st.write("Customer is predicted to leave the bank.")   
else:
    st.write("Customer is predicted to stay with the bank.")