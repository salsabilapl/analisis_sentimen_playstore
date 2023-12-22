import streamlit as st
from google_play_scraper import Sort, reviews
from google_play_scraper import app
import pandas as pd
import numpy as np
import joblib
from plotly import express as px
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

        #st.write("Data awal:")
        #st.write(data.head())

        processed_data = preprocess_text(data)
        #st.write("Data setelah preprocessing:")
        #st.write(processed_data.head())
        
        # Prediction part
        #st.write("Prediksi sentimen:")
        
        # Mapping the ratings
        processed_data['sentiment_rating'] = np.where(processed_data.label > 3, 1, 0)
        
        ## Removing neutral reviews
        processed_data = processed_data[processed_data.label != 3]
        
        # Get the counts of positive and negative sentiments
        positive_sentiments = processed_data[processed_data['sentiment_rating'] == 1]
        negative_sentiments = processed_data[processed_data['sentiment_rating'] == 0]
        
        # Get the counts
        positive_count = len(positive_sentiments)
        negative_count = len(negative_sentiments)
        
        # Creating Pie Chart with custom colors
        fig = px.pie(values=[positive_count, negative_count], names=['Positif', 'Negatif'], title='Perbandingan Sentimen')
        fig.update_traces(marker=dict(colors=['#0000BB', '#748BFB']))  # Ubah kode warna sesuai keinginan Anda
        st.plotly_chart(fig, use_container_width=True)

        # Display 3 examples of positive and negative comments in a table format
        st.write("Contoh Komentar Positif:")
        positive_samples = positive_sentiments.head(3)[['processed_text']]
        st.table(positive_samples)
        
        st.write("Contoh Komentar Negatif:")
        negative_samples = negative_sentiments.head(3)[['processed_text']]
        st.table(negative_samples)


if __name__ == "__main__":
    main()
