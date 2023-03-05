import streamlit as st
import sklearn
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def prerpocessing(text):
    ps = PorterStemmer()
    # convert it to lower case
    text = text.lower()
    
    # breaking text into words (tokenizations)
    text = nltk.word_tokenize(text)
    
    new_text = []
    for i in text:
        if i.isalnum():
            new_text.append(i)
    
    # shallow copy
    text = new_text[:]
    new_text.clear()
    
    # removing all stopwords (lang building words)
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            new_text.append(i)
      
    # shallow copy
    text = new_text[:]
    new_text.clear()
    
    # stemming (convert all the word to its root form or base form)
    for i in text:
        new_text.append(ps.stem(i))
        
    
    return " ".join(new_text)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Sms Spam Classifier')
html_temp = """
<p style="color: red;">
            This model gives "97%" accuracy,
            sometime its gives false output as well.
</p>
<span>Author: @Anirudh</span>
"""
st.markdown(html_temp,unsafe_allow_html=True)
input_sms = st.text_area('Enter your message')

if st.button('Predict'):
    transformed_sms = prerpocessing(input_sms)

    vector_input = tfidf.transform([transformed_sms])


    result = model.predict(vector_input)[0]


    if result == 0:
        st.header("Not Spam")

    else:
        st.header("Spam")
