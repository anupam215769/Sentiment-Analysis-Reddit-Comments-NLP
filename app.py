import streamlit as st
import tensorflow as tf

# Set the title of the app
st.title("Reddit Comments Sentiment Analysis")

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_1.keras')
    return model

model = load_model()

# Text input for the user comment
user_input = st.text_area("Enter a comment to identify:")

# Button to trigger classification
if st.button("Identify"):
    # Check if the input is not empty
    if user_input.strip() == "":
        st.write("Please enter a comment for identification.")
    else:
        # Prepare the input as a TensorFlow constant with the correct shape and dtype
        input_text = tf.constant([[user_input]], dtype=tf.string)  # Shape (1,1), dtype tf.string

        # Get the prediction probability
        prediction_prob = model.predict(input_text)[0][0]

        confidence_percentage = prediction_prob * 100

        # Determine the label based on the probability
        prediction_label = "Hateful" if prediction_prob >= 0.5 else "Normal"

        # Display the results
        st.write(f"### Prediction: {prediction_label}")

        if confidence_percentage > 80:
            st.success(f"High confidence ({confidence_percentage:.2f}%)")
        elif 40 <= confidence_percentage <= 80:
            st.warning(f"Moderate confidence ({confidence_percentage:.2f}%)")
        else:
            st.error(f"Low confidence ({confidence_percentage:.2f}%)")