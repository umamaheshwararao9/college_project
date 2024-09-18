import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

# Load your data and models
datasets = {
    "English": ("../csv_file_data/english_dataset.csv", "../model/english.h5"),
    "Spanish": ("../csv_file_data/spanish_dataset.csv", "../model/spanish.h5"),
    "French": ("../csv_file_data/french_dataset.csv", "../model/french.h5"),
    "Latin": ("../csv_file_data/latin_dataset.csv", "../model/latin.h5"),
    "German": ("../csv_file_data/german_dataset.csv", "../model/german.h5"),
}

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])  # Remove special characters
    else:
        text = str(text)  # Handle NaN values by converting to string
    return text

# Function to load dataset and model
def load_data_and_model(language):
    df = pd.read_csv(datasets[language][0], encoding='latin-1')
    df['Text'] = df['Text'].apply(clean_text)
    df = df[df['Text'] != '']  # Remove rows with empty strings after cleaning

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['Text'])
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['Emotion'])
    
    model = load_model(datasets[language][1])
    
    return tokenizer, label_encoder, model

# Load all models and tokenizers
models = {}
tokenizers = {}
label_encoders = {}

for language in datasets:
    tokenizers[language], label_encoders[language], models[language] = load_data_and_model(language)

# Initialize translator
translator = Translator()

# Streamlit app
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions in English Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        if st.form_submit_button(label='Submit'):
            if raw_text.strip() == '':
                st.error("Please enter some text.")
            else:
                cleaned_text = clean_text(raw_text)

                for language in datasets:
                    # Tokenize the text
                    tokenized_text = tokenizers[language].texts_to_sequences([cleaned_text])
                    padded_text = pad_sequences(tokenized_text, maxlen=100)

                    # Predict the emotion
                    prediction = models[language].predict(padded_text)
                    predicted_class = prediction.argmax(axis=-1)[0]
                    predicted_emotion = label_encoders[language].inverse_transform([predicted_class])[0]

                    # Translate emotion to English
                    translated_emotion = translator.translate(predicted_emotion, dest='en').text

                    # Display original and translated emotions
                    st.success(f"Prediction in {language}")
                    st.write(f"Original Emotion: {predicted_emotion}")
                    st.write(f"Translated Emotion: {translated_emotion}")

if __name__ == '__main__':
    main()
