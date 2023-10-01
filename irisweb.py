# irisweb.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
import streamlit as st
from PIL import Image

# Function to predict iris variety
def predict_iris_variety(model, sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return prediction

# Streamlit UI function
def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)
    st.markdown("You are using Streamlit...", unsafe_allow_html=True)
    
    sepal_length = st.text_input("Enter Sepal Length", "5.1")
    sepal_width = st.text_input("Enter Sepal Width", "3.5")
    petal_length = st.text_input("Enter Petal Length", "1.4")
    petal_width = st.text_input("Enter Petal Width", "0.2")
    
    if st.button("Click here to Predict"):
        try:
            result = predict_iris_variety(classifier, float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))
            st.balloons()
            st.success(f'The output is {result[0]}')
        except ValueError:
            st.error("Please enter valid numerical values for all the inputs.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Load the pre-trained model
try:
    with open("iris_model.pkl", "rb") as pickle_in:
        classifier = pickle.load(pickle_in)
except Exception as e:
    st.error(f"An error occurred while loading the model: {str(e)}")
    classifier = None

# Run the Streamlit app
if __name__ == "__main__":
    Input_Output()
