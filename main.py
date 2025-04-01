import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Step 1 : Load Imdb datset word index
word_index = imdb.get_word_index()
# Reverse the word index to get words from indices
reversed_index = dict((value,key) for (key,value) in word_index.items())

# Step 2 :Load the pre-trained model
model = load_model('imdb_rnn_model.h5')

# Step 3 : Helper function to decode reviews
def decode_review(text):
    # Decode the review from indices to words
    return ' '.join([reversed_index.get(i - 3, '?') for i in text])

# Step 4 : Function to preprocess user input
def preprocess_text(text):
    # Convert the text to lowercase and split into words
    words = text.lower().split()
    # Convert words to indices using the word index
    indices = [word_index.get(word, 2) +3 for word in words]  # +3 to account for padding
    padded_indices = sequence.pad_sequences([indices], maxlen=500)  # Pad to max length of 500
    return padded_indices

# Step 5 : Function to predict sentiment of a review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    # Make prediction
    prediction = model.predict(preprocessed_input)

    # Convert prediction to sentiment label
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    #print("Preprocessed Input:", preprocess_text(review_text))
    #print("Prediction:", prediction[0][0])
    return sentiment, prediction[0][0]


# Step 6 : streamlit app

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to get its sentiment prediction.")
st.write("Positive reviews are rated above 0.5 and negative reviews below 0.5.")

# Input text area for user to enter a review
review_text = st.text_area("Movie Review")

if st.button("Predict Sentiment"):
    sentiment, prediction = predict_sentiment(review_text)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction}")
    