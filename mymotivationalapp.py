#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:00:53 2020

@author: ananyabanerjee

Motivational Quotes Display App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

from time import time
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import recall_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import joblib
import altair as alt

DATA_URL_TRAIN= (
#"quotesdrivedb.csv"
"quotes.json"
)

#load Individual models
#model=joblib.load("Models/Individual/model_toxic.sav")

#load the individual vectorizer from disk
#vectorizer=joblib.load("Vectorizers/vectorizer_toxic.sav")

st.title("Welcome to your daily dose of motivation!")
st.markdown("This application is a Streamlit dashboard that can be used to display a motivational quote!")
#st.markdown("### My first streammlit dashboard")

st.sidebar.title("About")

st.sidebar.info("This is a demo application written for exploring motivational quotes!")

#display image
from PIL import Image
im = Image.open('m.jpeg')
#image=im.resize((int(im.size[0]/3),int(im.size[1]/3)), 0)
image=im
st.image(image) #use_column_width=True)


#@st.cache(persist=True)
def load_data(nrows, col):
    #data=pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE',"CRASH_TIME"]])
    #Read files
    #data=pd.read_csv(DATA_URL_TRAIN)
    data=pd.read_json(DATA_URL_TRAIN)

    # Dropping duplicates and creating a list containing all the quotes
    #quotes = data[col].drop_duplicates()
    #data.drop_duplicates(inplace=True, keep = False)
    data=data.drop_duplicates(subset=['Quote'], keep=False)

    st.write("columns", data.columns)

    #drop nan values
    data.dropna(subset=[col], inplace=True)
    #data_test_labels.dropna(subset=[col], inplace=True)

    #convert all to lowercase
    #lowercase = lambda x: str(x).lower()
    #data.rename(lowercase, axis='columns', inplace=True)
    #data_test.rename(lowercase, axis='columns', inplace=True)

    #rename columns if necessary
    #data.rename(columns={'crash_date_crash_time':'date/time'}, inplace=True)

    return data


#input data
data = load_data(10000, 'Quote')
original_data= data


#enter comment_text
input=st.text_input(label='Enter text')
#display it
st.write(input)

"""
## QUESTION:
st.header("How would you like to select your daily dost of motivation?")
#slider for asking type of Toxicity
_slider= st.selectbox('Categories', ['Quotes','Author', 'Popularity','Category'])

#query data based on query toxic_slider


if _slider=='Quotes':
    #st.write(original_data.query("injured_pedestrians >= 1")[["on_street_name", "injured_pedestrians"]].sort_values(by=['injured_pedestrians'], ascending=False).dropna(how='any')[:5])
    #use toxic model
    #bow=toxic_vectorizer.transform([input]).toarray()
    #pred=toxic_model.predict(bow)

    #bow=multi_vectorizer.transform([input]).toarray()
    #pred=multi_toxic_model.predict(bow)

    #st.write(pred)
    #if pred[0]==1:
    #    st.write("Yes! It is toxic!")
    #else:
    #    st.write("No! It is not toxic!")

elif _slider=='Author':
    #use severe toxic model

    #bow=severe_toxic_vectorizer.transform([input]).toarray()
    #pred=severe_toxic_model.predict(bow)


    #bow=multi_vectorizer.transform([input]).toarray()
    #pred=multi_severe_toxic_model.predict(bow)
    #pred=multi_model.predict(bow)

    #st.write(pred)
    #if pred[0]==1:
    #    st.write("Yes! It is severely toxic!")
    #else:
    #    st.write("No! It is not severely toxic!")

elif _slider=='Popularity':
    #use severe toxic model

    #bow=obscene_vectorizer.transform([input]).toarray()
    #pred=obscene_model.predict(bow)


    #bow=multi_vectorizer.transform([input]).toarray()
    #pred=multi_severe_toxic_model.predict(bow)
    #pred=multi_model.predict(bow)

    #st.write(pred)
    #if pred[0]==1:
    #    st.write("Yes! It is obscene!")
    #else:
    #    st.write("No! It is not obscene!")

elif _slider=='Category':
    #use severe toxic model

    #bow=threat_vectorizer.transform([input]).toarray()
    #pred=threat_model.predict(bow)


    #bow=multi_vectorizer.transform([input]).toarray()
    #pred=multi_severe_toxic_model.predict(bow)
    #pred=multi_model.predict(bow)

    #st.write(pred)
    #if pred[0]==1:
    #    st.write("Yes! It is threatning!")
    #else:
    #        st.write("No! It is not threatning!")


"""

#another ## QUESTION:
st.header("View training data according to category")
#
#st.markdown("")
num_datapoints = st.selectbox(
    'How many data points would you like to view?',[0, 5, 10, 50, 100, 500, 1000]
     )


st.header("Five Motivational Thoughts!")
#sample 5 motivational comments
cl = data['Quote'].sample(num_datapoints).values
#a=list(cl['comment_text'])
#st.write(a)
st.table(cl)

#Ques
st.header("How many categories are there?")

st.write(np.unique(data['Category']))

#another chart: # Pie chart
st.header("Pie chart showing percentage of each toxicity type")
#find counts
counts=list(np.unique(data['Category'].value_counts()))
labels=list(np.unique(data['Category']))

fig = go.Figure(data=[go.Pie(labels=labels, values=counts)])
#fig.show()
st.plotly_chart(fig)

#another questions
## QUESTION:
st.header("How many datapoints exist for each category")
#slider for asking type of Toxicity
st.write(data['Category'].value_counts())


#another # QUESTION:
st.header("Choice of graph as per category!")
#
column = st.selectbox('Categories', list(np.unique(data['Category'])))



st.area_chart(data.loc[data['Category']==column, 'Quote'].head(100))





if st.checkbox("Show Raw Data", False):
    st.subheader('Raw_Data')
    st.write(data.head(100))

    #heading
    st.header("Graph showing training data!")
    #make a chart_data
    #st.line_chart(data[['toxic','severe_toxic','insult','obscene','identity_hate','threat']].head(10000))
    #st.bar_chart(data[['toxic','severe_toxic','insult','obscene','identity_hate','threat']].head(10000))
    #st.line_chart(data['Category'].head(10000))
    #st.area_chart(data['Category'].head(10000))
    #st.area_chart(data.head(10000))
    c = alt.Chart(data).mark_circle().encode(x='Quote', y='Category')#, size='c',
                                       #color='c')
    st.altair_chart(c, width=-1)
