import streamlit as st
import tensorflow as tf
import numpy as np
from joblib import load as joblib_load
import os

st.title("Customer Feedback Analysis")

# Map display names to the actual filenames in ./models/
model_map = {
    "Naive Bayes": os.path.join("models", "model_0.joblib"),
    "Simple Dense": os.path.join("models", "model_1.keras"),
    "LSTM": os.path.join("models", "model_2.keras"),
    "GRU": os.path.join("models", "model_3.keras"),
    "BiLSTM": os.path.join("models", "model_4.keras"),
    "Conv1D": os.path.join("models", "model_5.keras"),
}

@st.cache_resource
def load_model(model_path):
    """
    Loads a saved model, either scikit-learn (.joblib)
    or Keras (.keras).
    """
    if model_path.endswith(".joblib"):
        return joblib_load(model_path)  # scikit-learn model
    else:
        return tf.keras.models.load_model(model_path)  # tf.keras model

# Let user pick a model by descriptive name
chosen_display_name = st.selectbox("Select a model:", list(model_map.keys()))
selected_model_file = model_map[chosen_display_name]

# Load the selected model
model = load_model(selected_model_file)

# Text input for the user review
user_input = st.text_area("Enter a review to analyze:")

# Button to trigger classification
if st.button("Analyze"):
    # Check if the input is not empty
    if user_input.strip() == "":
        st.write("Please enter a review for sentiment analysis.")
    else:
        # Determine the prediction probabilities for 3 classes: 0=Negative, 1=Neutral, 2=Positive
        if selected_model_file.endswith(".joblib"):
            # scikit-learn model => we can use predict_proba on the raw text
            # Output is something like [p_negative, p_neutral, p_positive]
            probabilities = model.predict_proba([user_input])[0]
        else:
            # Keras model => expects a tensor of shape (batch_size, 1) dtype=string
            input_text = tf.constant([[user_input]], dtype=tf.string)
            probabilities = model.predict(input_text)[0]

        # Convert probabilities to a class index
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        confidence_percent = confidence * 100

        # Map the integer label to a string label
        label_map = ["Negative", "Neutral", "Positive"]
        predicted_label = label_map[predicted_class]

        # Display the results
        st.write(f"### Prediction: {predicted_label}")

        # Color-code confidence
        if confidence_percent > 80:
            st.success(f"High confidence ({confidence_percent:.2f}%)")
        elif 40 <= confidence_percent <= 80:
            st.warning(f"Moderate confidence ({confidence_percent:.2f}%)")
        else:
            st.error(f"Low confidence ({confidence_percent:.2f}%)")