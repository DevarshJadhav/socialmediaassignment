import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
import json
import re
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# News API Key
api_key = '7a3ff37653f347c5a234987df2fecfd7'

# Stopwords
stop_words = set(stopwords.words('english'))

# Fetch and clean news data
def fetch_news(query, language='en', page_size=100):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    news_data = response.json()
    return news_data

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(cleaned_tokens)

# Generate LDA model
def generate_lda(cleaned_articles):
    dictionary = corpora.Dictionary([article.split() for article in cleaned_articles])
    corpus = [dictionary.doc2bow(article.split()) for article in cleaned_articles]
    lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
    return lda_model, dictionary

# Generate word cloud
def generate_wordcloud(lda_model):
    topics = lda_model.show_topics(num_topics=10, num_words=10)
    topic_strings = [topic[1] for topic in topics]
    all_topics = ' '.join([' '.join(word.split('*')[1].replace('"', '') for word in topic.split('+')) for topic in topic_strings])

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_topics)
    return wordcloud

# Streamlit App
st.title('News Topic Modeling and Word Cloud')

query = st.text_input('Enter a topic to search for news:', 'technology')
if st.button('Fetch News and Generate Word Cloud'):
    news_data = fetch_news(query)
    articles = [article['content'] for article in news_data['articles'] if article['content']]
    cleaned_articles = [clean_text(article) for article in articles]

    lda_model, dictionary = generate_lda(cleaned_articles)
    wordcloud = generate_wordcloud(lda_model)

    st.image(wordcloud.to_array())

    st.write('Click on a word to display related news articles.')
    selected_word = st.text_input('Enter a word to search for related articles:')
    
    if selected_word:
        related_articles = [article for article in articles if article and selected_word in article]
        for article in related_articles:
            st.write(article)

if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
