# %%
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

# %%
#load the trained model
model=tf.keras.models.load_model('model.h5')

# %%
#load the encoder and scaler
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender =pickle.load(file)

with open('OneHotEncoder.pkl','rb')as file:
    OneHotEncoder=pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler=pickle.load(file)

# %%
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv("Churn_Modelling.csv")

onehot_encoder_geo = OneHotEncoder()
onehot_encoder_geo.fit(data[['Geography']])

label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(data['Gender'])

# %%
st.title('Customer Churn Prediction')

geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# %%
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_data

# %%
# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded

# %%
# Create input data again from scratch

input_data = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_data = pd.DataFrame([input_data])

# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine
input_data = pd.concat(
    [input_data, geo_encoded_df],
    axis=1
)

# Arrange columns exactly like the scaler
input_data = input_data[scaler.feature_names_in_]

# Scale
input_scaled = scaler.transform(input_data)

input_scaled

# %%
prediction = model.predict(input_scaled)

prediction

# %%
if prediction[0][0] > 0.5:
    print("Customer will churn")
else:
    print("Customer will not churn")


