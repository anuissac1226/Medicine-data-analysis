import numpy as np
import streamlit as st
import pandas as pd
import logging
import pickle

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def load_assets():
    with open('random_forest_model_for_rating_prediction.pkl', 'rb') as file:
        model = pickle.load(file) #loading the best model
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file) #loading the std scalar
    with open('tfidf.pkl', 'rb') as file:
        tfidf = pickle.load(file) #loading the tfidf
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file) #loading the label encoder
    return model,scaler,tfidf,label_encoder

model,scaler,tfidf,label_encoder = load_assets()

# Loading dataset
def load_dataset():
    return pd.read_csv('data/Medicine_Details_Updated.csv')

df = load_dataset()

# Streamlit app
st.title('Medicine Analysis')
st.sidebar.header('User Input')
# User Input for Medicine Name
medicine_name = st.text_input('Medicine Name')
logger.debug(f"User input: {medicine_name}")
med_details = df[df['Medicine Name'].str.lower() == medicine_name.lower()]
if not med_details.empty:
    logger.info("medicine name entered...")
    manufacturer = med_details.iloc[0]['Manufacturer']
    composition = med_details.iloc[0]['Composition']
    excellent_review = med_details.iloc[0]['Excellent Review %']
    avg_review = med_details.iloc[0]['Average Review %']
    poor_review = med_details.iloc[0]['Poor Review %']
    logger.info("details fetched...")
    # Show fetched details in dropdown
    manufacturer_selected = st.selectbox('Manufacturer (Fetched)',options=["Select", manufacturer],index=0)
    composition_selected = st.selectbox('Composition (Fetched)', options=["Select", composition], index=0)
    excellent_selected = st.sidebar.slider('Excellent Review %', min_value=0, max_value=100, value=0)
    avg_selected = st.sidebar.slider('Average Review %', min_value=0, max_value=100, value=0)
    poor_selected = st.sidebar.slider('Poor Review %', min_value=0, max_value=100, value=0)
else:
    manufacturer_selected = st.text_input('Manufacturer','')
    composition_selected = st.text_input('Composition', '')
    excellent_selected = st.sidebar.slider('Excellent Review %', min_value=0, max_value=100, value=0)
    avg_selected = st.sidebar.slider('Average Review %', min_value=0, max_value=100, value=0)
    poor_selected = st.sidebar.slider('Poor Review %', min_value=0, max_value=100, value=0)

if st.button('Predict User Rating'):
    logger.info("predict button clicked.")
    # Apply label encoding to Medicine Name and Manufacture
    encoded_medicine_name = label_encoder['Medicine Name'].transform([medicine_name]).reshape(1, -1)
    encoded_manufacturer = label_encoder['Manufacturer'].transform([manufacturer_selected]).reshape(1, -1)
    logger.info("label encodeing...")

    # Apply tfidf to composition
    composition_tfidf = tfidf['Composition'].transform([composition_selected]).toarray()
    excellent_review = float(excellent_selected) if excellent_selected != "Select" else 0
    avg_review = float(avg_selected) if avg_selected != "Select" else 0
    poor_review = float(poor_selected) if poor_selected != "Select" else 0
    reviews = np.array([excellent_review, avg_review, poor_review]).reshape(1, -1)
    logger.debug(f"exe review input: {excellent_review}")
    logger.debug(f"avg review input: {avg_review}")
    logger.debug(f"poor review input: {poor_review}")
    logger.debug(f"reviews : {reviews}")

    # Combine all features into a single input
    X_input = np.hstack([encoded_medicine_name,encoded_manufacturer,reviews,composition_tfidf])

    # Scale the input
    X_input_scaled = scaler.transform(X_input)
    logger.info("scaled...")

    # Predict Rating
    predicted_rating = model.predict(X_input_scaled)
    logger.info("prediction...")
    logger.debug(f"rating: {predicted_rating[0]:.2f}")

    # Display predicted Rating
    st.write(f'Predicted Rating:{predicted_rating[0]:.2f}')
