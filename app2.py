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
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

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


# Streamlit app
def main():
    st.title('Analisis Sentimen pada Ulasan Aplikasi Playstore')

    st.sidebar.title("Analisis Sentimen Ulasan Pada Google Play Store")
    st.sidebar.write("Selamat datang di Aplikasi Analisis Sentimen versi 2.0!")
    st.sidebar.title('Menu')
    menu = st.sidebar.selectbox('-Pilih Menu-', ["🏠 Home","🤖 Sentiment Analysis"])

    if menu == '🏠 Home':
        st.image('sentiment-icon.png', use_column_width=10)
        st.info('Selamat Datang', icon="👋")
        st.warning('Aplikasi ini menganalisis ribuan ulasan dari Play Store secara real-time untuk memberikan wawasan yang mendalam mengenai sentimen pengguna terhadap aplikasi yang Anda pilih.', icon="❓")
        st.info('Silakan pilih Sentiment Analysis di sidebar', icon="⬅️")
        
    elif menu=='🤖 Sentiment Analysis':
        st.image('playstore.png', use_column_width=100)
        
        app_link = st.text_input("Masukkan link Google Play Store:")
        
        if st.button("Proses Link"):
            app_id = app_link.split('id=')[1].split('&')[0]
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.MOST_RELEVANT,
                count=3000,
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
    
            # Load Naive Bayes model from pickle file
            model_nb = joblib.load("model_naive_bayes.pkl")
        
            # Predict sentiment using the loaded model
            data['predicted_sentiment'] = model_nb.predict(data['processed_text'])
    
            #Get the counts of positive and negative sentiments
            positive_sentiments = data[data['predicted_sentiment'] == 1]
            negative_sentiments = data[data['predicted_sentiment'] == 0]
            
            # Get the counts
            positive_count = len(positive_sentiments)
            negative_count = len(negative_sentiments)
            
            # Creating Pie Chart with custom colors
            st.markdown("<h3 style='text-align: center;'>Perbandingan Sentimen</h3>", unsafe_allow_html=True)
            fig = px.pie(values=[positive_count, negative_count], names=['Positif', 'Negatif'])
            fig.update_traces(marker=dict(colors=['#0000BB', '#748BFB']))  # Ubah kode warna sesuai keinginan Anda
            st.plotly_chart(fig, use_container_width=True)
    
            st.markdown("<h3 style='text-align: center;'>Jumlah Sentimen</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Sentimen Positif:** {positive_count}")
            with col2:
                st.info(f"**Sentimen Negatif:** {negative_count}")
    
    
            # Display 3 examples of positive and negative comments in a table format
            st.markdown("<h3 style='text-align: center;'>Contoh Komentar</h3>", unsafe_allow_html=True)
            st.write("Contoh Komentar Positif:")
            positive_samples = positive_sentiments.head(5)[['ulasan']]
            st.table(positive_samples)
            
            st.write("Contoh Komentar Negatif:")
            negative_samples = negative_sentiments.head(5)[['ulasan']]
            st.table(negative_samples)
    
            word_cloud_text = ''.join(data['ulasan'])
    
            # Show Word Cloud
            st.subheader('Word Cloud dari Ulasan')
            word_cloud_text = ''.join(data['processed_text'])
            wordcloud = WordCloud(max_font_size=100, max_words=100, background_color="white",
                                 scale=10, width=800, height=400).generate(word_cloud_text)
            
            st.image(wordcloud.to_array())

if __name__ == "__main__":
    main()
