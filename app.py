import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
model = joblib.load('spam_classifier_model.joblib')

# Load the TF-IDF vectorizer (we need this to process new emails the same way as the training data)
# This part assumes you saved the vectorizer as well during training.
# If not, we might need to recreate it. Let me know if you didn't save it.
try:
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.warning("TF-IDF vectorizer file not found. The app might not work correctly.")
    tfidf_vectorizer = None

# Create a title for our app
st.title('Spam Email Detector')

# Create a text input box for the user to enter an email
email_text = st.text_area("Enter email here:")

# Create a button that will trigger the prediction
if st.button('Check Email'):
    if email_text:
        if tfidf_vectorizer is not None:
            # Transform the input email using the same TF-IDF vectorizer
            email_vectorized = tfidf_vectorizer.transform([email_text])

            # Make the prediction
            prediction = model.predict(email_vectorized)[0]

            # Display the result
            if prediction == 1:
                st.error('This email is likely SPAM!')
            else:
                st.success('This email is NOT spam.')
        else:
            st.error("Error: TF-IDF vectorizer not loaded.")
    else:
        st.warning('Please enter an email to check.')