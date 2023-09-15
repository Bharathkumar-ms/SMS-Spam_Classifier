import streamlit as st
import joblib

# Load the saved CountVectorizer
with open("count_vectorizer.pkl", "rb") as vectorizer_file:
    loaded_vectorizer = joblib.load(vectorizer_file)

# Load the saved XGBoost classifier
with open("best_nb_classifier.pkl", "rb") as nb_file:
    loaded_nb_classifier = joblib.load(nb_file)

# Create a Streamlit web app
st.title("Email Spam Classifier")

# Add a text input field for entering an email
email_text = st.text_area("Enter an email text:")

# Create a function to classify the email
def classify_email(email_text):
    if email_text:
        email_count = loaded_vectorizer.transform([email_text])
        prediction = loaded_nb_classifier.predict(email_count)
        return "Spam" if prediction[0] == 1 else "Not Spam"
    else:
        return ""

# Add a button to classify the email
if st.button("Classify"):
    result = classify_email(email_text)
    st.write(f"Classification: {result}")

