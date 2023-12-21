import streamlit as st
from google_play_scraper import Sort, reviews
from google_play_scraper import app
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(data):
    # Convert text to lowercase
    data['reviews_text_new'] = data['ulasan'].str.lower()
    
    # Remove special characters
    data['reviews_text_new'] = data['reviews_text_new'].str.replace(r'[^A-Za-z0-9 ]+', ' ')
    
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    data['reviews_text_nonstop'] = data['reviews_text_new'].apply(lambda row: [word for word in nltk.word_tokenize(row) if word not in stop_words])
    
    # Load normalized words from file
    normalized_word = pd.read_excel("kamus perbaikan kata.xlsx")
    normalized_word_dict = {}

    for index, row in normalized_word.iterrows():
        if row[0] not in normalized_word_dict:
            normalized_word_dict[row[0]] = row[1]

    # Normalize slang words
    def normalized_term(document):
        return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]
    
    data['reviews_slang'] = data['reviews_text_nonstop'].apply(normalized_term)
    
    # Initialize Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    # Stemming process
    def stemm_terms(row):
        return [stemmer.stem(term) for term in row]
    
    data['reviews_text_stemm'] = data['reviews_slang'].apply(lambda row: stemm_terms(row))
    
    # Join stemmed words
    final_string = [' '.join(text) for text in data['reviews_text_stemm'].values]
    data["processed_text"] = final_string
    
    return data

# Function to predict sentiment
def predict_sentiment(review_input):
    # Load Naive Bayes model from pickle file
    model_nb = joblib.load("model_naive_bayes.pkl")

    # Predict sentiment using the loaded model
    predicted_sentiment = model_nb.predict(review_input)
    return predicted_sentiment[0]


# Streamlit app
def main():
    st.title('Text Preprocessing App')
    
    app_link = st.text_input("Masukkan link Google Play Store:")
    
    if st.button("Proses Link"):
        app_id = app_link.split('id=')[1].split('&')[0]
        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=2000,
            filter_score_with=None
        )

        data = pd.DataFrame(np.array(result), columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))

        data = data[['content', 'score']]

        data = data.rename(columns={'content': 'ulasan', 'score': 'label'})

        st.write("Data awal:")
        st.write(data.head())

        processed_data = preprocess_text(data)
        st.write("Data setelah preprocessing:")
        st.write(processed_data.head())
        
        # Prediction part
        st.write("Prediksi sentimen:")
        
        # Mapping the ratings
        data['sentiment_rating'] = np.where(data.label > 3,1,0)
        
        ## Removing neutral reviews
        data = data[data.label != 3]
        
        # Printing the counts of each class
        #st.write(data['sentiment_rating'].value_counts())
        
        processed_data = processed_data[processed_data.label != 3]
        for idx, row in processed_data.iterrows():
            predicted_sentiment = predict_sentiment([row['processed_text']])
            if predicted_sentiment == 1:
                st.write(f"Review {idx+1}: Positif")
            else:
                st.write(f"Review {idx+1}: Negatif")


if __name__ == "__main__":
    main()
