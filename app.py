
import re
import numpy as np
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec


# Project Directory
BASE_DIR = Path(__file__).resolve().parent

# Model File Paths
ann_model_path = BASE_DIR / "final_ann_model.keras"
word2vec_model_path = BASE_DIR / "word2vec_model.model"


# Load Trained Models
@st.cache_resource
def load_models():

    ann_model = load_model(ann_model_path)

    word2vec_model = Word2Vec.load(
        str(word2vec_model_path)
    )

    return ann_model, word2vec_model


model, word2vec_model = load_models()


# Text Cleaning Function
def clean_text(text):

    # Convert text into lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z'\s]", ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Convert Review into Word2Vec Vector
def get_review_vector(review, model):

    vectors = [
        model.wv[word]
        for word in review
        if word in model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# Streamlit Application
st.set_page_config(
    page_title="McDonald's Review Rating Prediction",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 McDonald's Customer Review Rating Prediction")

st.write(
    "Enter a customer review to predict its rating "
    "using Word2Vec and Artificial Neural Network."
)

st.markdown("---")


# User Input
review = st.text_area(
    "Enter your customer review:",
    placeholder="Example: The food was delicious and the service was excellent.",
    height=150
)


# Prediction Button
if st.button("Predict Rating", type="primary"):

    if review.strip() == "":

        st.warning("Please enter a customer review.")

    else:

        # Clean Review
        cleaned_review = clean_text(review)

        # Tokenize Review
        review_tokens = cleaned_review.split()

        # Generate Word2Vec Vector
        review_vector = get_review_vector(
            review_tokens,
            word2vec_model
        )

        # Reshape Input for ANN
        review_vector = np.array(
            [review_vector],
            dtype=np.float32
        )

        # Predict Rating
        prediction = model.predict(
            review_vector,
            verbose=0
        )

        # Get Predicted Class
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Convert Class into Star Rating
        predicted_rating = predicted_class + 1

        # Calculate Confidence
        confidence = np.max(prediction) * 100

        # Display Results
        st.success(
            f"Predicted Rating: {predicted_rating} ⭐"
        )

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display Probability Distribution
        st.subheader("Rating Probability Distribution")

        for i, probability in enumerate(prediction[0]):

            rating = i + 1

            st.write(
                f"{rating} Star: {probability * 100:.2f}%"
            )

            st.progress(float(probability))


st.markdown("---")

st.caption(
    "Machine Learning Project | Word2Vec + Optuna Tuned ANN"
)
