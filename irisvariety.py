import pickle
import warnings
import streamlit as st
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the pre-trained model
pickle_in = open("C:/Users/Ez-Studio/streamlitproject/knn_model.pkl", "rb")
classifier = pickle.load(pickle_in)

# Function to predict iris variety
def predict_iris_variety(sepal_length, sepal_width, petal_length, petal_width):
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

# Streamlit UI function
def Input_Output():
    st.title("Iris Variety Prediction")
    st.image("https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png", width=600)
    st.markdown("You are using Streamlit...", unsafe_allow_html=True)
    
    sepal_length = st.text_input("Enter Sepal Length", ".")
    sepal_width = st.text_input("Enter Sepal Width", ".")
    petal_length = st.text_input("Enter Petal Length", ".")
    petal_width = st.text_input("Enter Petal Width", ".")
    
    result = ""
    if st.button("Click here to Predict"):
        result = predict_iris_variety(float(sepal_length), float(sepal_width), float(petal_length), float(petal_width))
        st.balloons()
        st.success('The output is {}'.format(result))

# Run the Streamlit app
if __name__ == "__main__":
    Input_Output()
