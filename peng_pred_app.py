import streamlit as st
import pandas as pd
import pickle
from preprocess import preprocess_input

# Load the trained model
model = pickle.load(open('penguin_predict.pkl', 'rb'))

# Function to make predictions
def predict_species(features):
    # Assuming you have a function to preprocess input features
    # Replace this with your actual preprocessing logic
    preprocessed_features = preprocess_input(features)

    # Make predictions
    prediction = model.predict(preprocessed_features)
    return prediction

species_maping = {1:'Gentoo', 0:'Adelie'}
def decode_species(encoded_species):
    return species_maping.get(encoded_species, 'Unkown')


# Main Streamlit app
def main():
    with st.container(border=True):
        st.title("Penguin Species Prediction App")

    # Input form for the user to enter feature values
    st.sidebar.header("User Input")
    culmen_length = st.sidebar.slider("Bill Length (mm)", min_value=30.0, max_value=60.0, value=39.1)
    culmen_depth = st.sidebar.slider("Bill Depth (mm)", min_value=13.0, max_value=21.0, value=18.7)
    flipper_length = st.sidebar.slider("Flipper Length (mm)", min_value=170.0, max_value=240.0, value=181.0)
    body_mass = st.sidebar.slider("Body Mass (g)", min_value=2500.0, max_value=6500.0, value=3750.0)
    island_n = st.sidebar.selectbox('Island: ', ('Biscoe', 'Dream', 'Torgersen'), placeholder='Torgersen')
    year_n = st.sidebar.select_slider('Year: ', ('2007', '2008', '2009'), value='2007')

    # Store user input in a DataFrame
    user_input = pd.DataFrame({
        'island': [island_n],
        'bill_length_mm': [culmen_length],
        'bill_depth_mm': [culmen_depth],
        'flipper_length_mm': [flipper_length],
        'body_mass_g': [body_mass],
        'year': [year_n]
    })

    # Display the user input
    st.subheader("User Input:")
    st.write(user_input)

    # Make predictions
    prediction = predict_species(user_input)

    # Display the prediction
    final_species = decode_species(prediction[0])
    with st.container(border=True):
        st.header(f"The predicted species is: {final_species}")

    st.image('dataset-cover.jpg')
    st.markdown('''# Penguin Classification Project

## Overview
This project focuses on classifying penguins into two species (Adelie and Gentoo) based on various physical measurements. The dataset used is the Palmer Penguins dataset, which includes features like bill length, bill depth, flipper length, and body mass.

## Goal
The main goal of this project is to build a classification model that accurately predicts the species of a penguin given its physical measurements. The ultimate aim is to get a model that is able to predict the penguins with a 100% accuracy.

## Tools and Libraries
- Python
- Notebooks (iPythonNotebook)
- Scikit-learn
- Matplotlib
- Streamlit (to deploy the app)

 ***Project undertaken by Digant Singh, 2024***
''')
    

if __name__ == "__main__":
    main()
