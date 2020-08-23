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

st.title("Welcome to your daily dose of motivation!😊")
#st.write("𝕄𝕠𝕥𝕚𝕧𝕒𝕥𝕖❕ 𝕀𝕕𝕖𝕒𝕥𝕖❕ 𝕀𝕟𝕟𝕠𝕧𝕒𝕥𝕖❕ 𝕀𝕟𝕤𝕡𝕚𝕣𝕖❕")
st.markdown("𝑴𝒐𝒕𝒊𝒗𝒂𝒕𝒆❗ 𝑰𝒅𝒆𝒂𝒕𝒆❗ 𝑰𝒏𝒏𝒐𝒗𝒂𝒕𝒆❗ 𝑰𝒏𝒔𝒑𝒊𝒓𝒆❗")
#st.markdown("🅼🅾🆃🅸🆅🅰🆃🅴❗ 🅸🅳🅴🅰🆃🅴❗ 🅸🅽🅽🅾🆅🅰🆃🅴❗ 🅸🅽🆂🅿🅸🆁🅴❗")
#st.markdown("Mᴏᴛɪᴠᴀᴛᴇ! Iᴅᴇᴀᴛᴇ! Iɴɴᴏᴠᴀᴛᴇ! Iɴsᴘɪʀᴇ!")
st.markdown("> This application is a Streamlit dashboard that can be used to display a motivational quote!")
#st.markdown("### My first streammlit dashboard")

st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/badge/MADE%20WITH-PYTHON%20-red?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/SERVED%20WITH-Heroku-blue?style=for-the-badge'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/DASHBOARDING%20WITH-Streamlit-green?style=for-the-badge'"
                "alt='API stability' height='25'/></div>", unsafe_allow_html=True)


st.sidebar.title("\n About")

st.sidebar.info("This is a demo application written for exploring motivational quotes!")

st.write("\n")
st.write("\n")
st.write("\n")

#display image
from PIL import Image
im = Image.open('m5.jpg')
image=im.resize((int(im.size[0]/2),int(im.size[1]/2)), 0)
#image=im
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

    #st.write("columns", data.columns)

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
#input=st.text_input(label='Enter text')
#display it
#st.write(input)


#another ## QUESTION:
st.header("\n How many motivational quotes would you like to view?🤔💭")
#
#st.markdown("")
num_datapoints = st.selectbox(
    'How many data points would you like to view?',[1, 5, 10, 50, 100, 500, 1000]
     )


#st.header("Your customized motivational thoughts!")
st.header("𝒞𝓊𝓈𝓉ℴ𝓂𝒾𝓏ℯ𝒹 ℳℴ𝓉𝒾𝓋𝒶𝓉𝒾ℴ𝓃!")
#sample 5 motivational comments
cl = list(data['Quote'].sample(num_datapoints).values)
#a=list(cl['comment_text'])
#st.write(a)
st.table(cl)


re=st.selectbox("Would you like to know the author of this quote?", ['N/A','No','Yes'])
if re=='Yes':
    for i in range(len(cl)):
        st.write("\n For the quote '",cl[i],"', the author is",list(data.loc[data['Quote']==cl[i]]['Author'])[0])

elif re=='No':
    st.markdown("Cool! ")


#Ques
st.header("The motivational thoughts belong to these categories!")
v=list(np.unique(data['Category']))
v.remove("")
st.write(v)

#another chart: # Pie chart
st.header("Pie chart showing percentage of each category of thoughts in our database!")
#find counts
counts=list(np.unique(data['Category'].value_counts()))
labels=list(np.unique(data['Category']))

fig = go.Figure(data=[go.Pie(labels=labels, values=counts)])
#fig.show()
st.plotly_chart(fig)

#another questions
## QUESTION:
st.header("Datapoints existing in our databse belonging to aforementioned categories!")
#slider for asking type of Toxicity
st.write(data['Category'].value_counts())


#
st.header("Would you like to choose the category to which your thought of the data should belong to?")

ans=st.selectbox("Choice", ['Yes','No'])

if ans=='Yes':
   d=st.selectbox("Which category would you like to see?", labels)

   d1=list(data[data['Category']==d]['Quote'].sample(1))[0]
   st.write("Your motivational quote: ")
   st.markdown(d1)
else:
    st.markdown("Your wish is my command! 😊")







if st.checkbox("Show Raw Data", False):
    st.subheader('Raw_Data')
    st.write(data.head(100))

    #heading
    #st.header("Graph showing training data!")
    #make a chart_data
    #st.line_chart(data[['toxic','severe_toxic','insult','obscene','identity_hate','threat']].head(10000))
    #st.bar_chart(data[['toxic','severe_toxic','insult','obscene','identity_hate','threat']].head(10000))
    #st.line_chart(data['Category'].head(10000))
    #st.area_chart(data['Category'].head(10000))
    #st.area_chart(data.head(10000))
    #c = alt.Chart(data).mark_circle().encode(x='Quote', y='Category')#, size='c',
                                       #color='c')
    #st.altair_chart(c, width=-1)
    #another # QUESTION:
    st.header("Choice of graph as per category!")
    column = st.selectbox('Categories', list(np.unique(data['Category'])))
    st.area_chart(data.loc[data['Category']==column, 'Quote'].head(100))
