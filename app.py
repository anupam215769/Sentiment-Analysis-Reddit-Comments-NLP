import streamlit as st
import tensorflow as tf

# Set the title of the app
st.title("Hateful or Normal Comment Classifier")

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_1.keras')
    return model

model = load_model()

# Text input for the user comment
user_input = st.text_area("Enter a comment to classify:")

# Button to trigger classification
if st.button("Classify"):
    # Check if the input is not empty
    if user_input.strip() == "":
        st.write("Please enter a comment for classification.")
    else:
        # Prepare the input as a list (the model expects a batch of inputs)
        input_text = [user_input]

        # Get the prediction probability
        prediction_prob = model.predict(input_text)[0][0]

        # Determine the label based on the probability
        prediction_label = "Hateful" if prediction_prob >= 0.5 else "Normal"

        # Display the results
        st.markdown(f"**Prediction:** {prediction_label}")
        st.markdown(f"**Probability:** {prediction_prob:.4f}")