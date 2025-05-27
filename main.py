import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
# word_index
rev_index = {v: k for k, v in word_index.items()}

model = load_model('imdb_rnn_model.keras')

def decode_review(encoded_review):
    decoded_review = ' '.join([rev_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

def preprocess_review(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_review(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    return sentiment, prediction[0][0]

import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment.")

review_input = st.text_area("This movie was fantastic!")

if st.button("Classify"):

    preprocess_input = preprocess_review(review_input)

    prediction = model.predict(preprocess_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]:.4f}")

else:
    st.write('plese enter a review and click "Classify"')
