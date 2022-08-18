import nltk
import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

ps = PorterStemmer()

def convert_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    x = []

    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            x.append(i)

    text = x[:]
    x.clear()

    for i in text:
        x.append(ps.stem(i))

    return " ".join(x)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Classifier')

input_msg = st.text_area("Enter the message")

if st.button('Predict'):

    converted_sms = convert_text(input_msg)
    vector_input = tfidf.transform([converted_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam !!!')
    else:
        st.header('Not a Spam')