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
import random
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

#st.write("𝓑𝓮𝓵𝓲𝓮𝓿𝓮 𝓲𝓷 𝔂𝓸𝓾𝓻𝓼𝓮𝓵𝓯❗")

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
#st.header("\n How many motivational quotes would you like to view?🤔💭")
#
st.header("\n 𝐇𝐨𝐰 𝐦𝐚𝐧𝐲 𝐦𝐨𝐭𝐢𝐯𝐚𝐭𝐢𝐨𝐧𝐚𝐥 𝐪𝐮𝐨𝐭𝐞𝐬 𝐰𝐨𝐮𝐥𝐝 𝐲𝐨𝐮 𝐥𝐢𝐤𝐞 𝐭𝐨 𝐯𝐢𝐞𝐰❓ 🤔💭")

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

colors=['red','yellow','green','blue','informational','important','blueviolet','ff69b4']

for i in range(len(cl)):
    col=random.choice(colors)
    bv=cl[i].replace("'","")
    #st.write(col)
    st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/static/v1?label=*&message="+bv+"&color="+col+"'"
                "alt='API stability' height='25'/></div>", unsafe_allow_html=True)
    st.write("\n")


re=st.selectbox("Would you like to know the author of this quote?", ['N/A','No','Yes'])
if re=='Yes':
    for i in range(len(cl)):
        st.write("\n For the quote '",cl[i],"', the author is",list(data.loc[data['Quote']==cl[i]]['Author'])[0])

elif re=='No':
    st.markdown("Cool! ")


#Ques
#st.header("The motivational thoughts belong to these categories!")
st.header("🆃🅷🅴 🅼🅾🆃🅸🆅🅰🆃🅸🅾🅽🅰🅻 🆃🅷🅾🆄🅶🅷🆃🆂 🅱🅴🅻🅾🅽🅶 🆃🅾 🆃🅷🅴🆂🅴 🅲🅰🆃🅴🅶🅾🆁🅸🅴🆂")
v=list(np.unique(data['Category']))
v.remove("")
st.write(v)

#another chart: # Pie chart
#st.header("Pie chart showing percentage of each category of thoughts in our database!")
st.header("𝐏𝐢𝐞 𝐜𝐡𝐚𝐫𝐭 𝐬𝐡𝐨𝐰𝐢𝐧𝐠 𝐩𝐞𝐫𝐜𝐞𝐧𝐭𝐚𝐠𝐞 𝐨𝐟 𝐞𝐚𝐜𝐡 𝐜𝐚𝐭𝐞𝐠𝐨𝐫𝐲 𝐨𝐟 𝐭𝐡𝐨𝐮𝐠𝐡𝐭𝐬 𝐢𝐧 𝐨𝐮𝐫 𝐝𝐚𝐭𝐚𝐛𝐚𝐬𝐞❗")
#find counts
counts=list(np.unique(data['Category'].value_counts()))
labels=list(np.unique(data['Category']))

fig = go.Figure(data=[go.Pie(labels=labels, values=counts)])
#fig.show()
st.plotly_chart(fig)

#another questions
## QUESTION:
#st.header("Datapoints existing in our databse belonging to aforementioned categories!")
st.header("𝔇𝔞𝔱𝔞𝔭𝔬𝔦𝔫𝔱𝔰 𝔢𝔵𝔦𝔰𝔱𝔦𝔫𝔤 𝔦𝔫 𝔬𝔲𝔯 𝔡𝔞𝔱𝔞𝔟𝔞𝔰𝔢 𝔟𝔢𝔩𝔬𝔫𝔤𝔦𝔫𝔤 𝔱𝔬 𝔞𝔣𝔬𝔯𝔢𝔪𝔢𝔫𝔱𝔦𝔬𝔫𝔢𝔡 𝔠𝔞𝔱𝔢𝔤𝔬𝔯𝔦𝔢𝔰!")
#slider for asking type of Toxicity
st.write(data['Category'].value_counts())


#
#st.header("Would you like to choose the category to which your thought of the data should belong to?")
st.header("𝐖𝐨𝐮𝐥𝐝 𝐲𝐨𝐮 𝐥𝐢𝐤𝐞 𝐭𝐨 𝐜𝐡𝐨𝐨𝐬𝐞 𝐭𝐡𝐞 𝐜𝐚𝐭𝐞𝐠𝐨𝐫𝐲 𝐭𝐨 𝐰𝐡𝐢𝐜𝐡 𝐲𝐨𝐮𝐫 𝐭𝐡𝐨𝐮𝐠𝐡𝐭 𝐨𝐟 𝐭𝐡𝐞 𝐝𝐚𝐭𝐚 𝐬𝐡𝐨𝐮𝐥𝐝 𝐛𝐞𝐥𝐨𝐧𝐠 𝐭𝐨❓")
ans=st.selectbox("Choice", ['Yes','No'])

if ans=='Yes':
   d=st.selectbox("Which category would you like to see?", labels)

   d1=list(data[data['Category']==d]['Quote'].sample(1))[0]
   st.write("​​​​​​​​​​𝒴ℴ𝓊𝓇 𝓂ℴ𝓉𝒾𝓋𝒶𝓉𝒾ℴ𝓃𝒶𝓁 𝓆𝓊ℴ𝓉ℯ:")
   st.markdown(d1)
else:
    st.markdown("Your wish is my command! 😊")








if st.checkbox("Show Raw Data", False):
    st.subheader('🆁🅰🆆 🅳🅰🆃🅰')
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
    st.header("𝕮𝖍𝖔𝖎𝖈𝖊 𝖔𝖋 𝖌𝖗𝖆𝖕𝖍 𝖆𝖘 𝖕𝖊𝖗 𝖈𝖆𝖙𝖊𝖌𝖔𝖗𝖞❗")
    column = st.selectbox('Categories', list(np.unique(data['Category'])))
    st.area_chart(data.loc[data['Category']==column, 'Quote'].head(100))




st.markdown("<div align='center'><br>"
                "<img src='https://img.shields.io/badge/Self%20Motivate-red?style=for-the-badge&logo=appveyor'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/-Ideate-green?style=for-the-badge&logo=appveyor'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/-Innovate-yellow?style=for-the-badge&logo=appveyor'"
                "alt='API stability' height='25'/>"
                "<img src='https://img.shields.io/badge/-Inspire-informational?style=for-the-badge&logo=appveyor'"
                "alt='API stability' height='25'/></div>", unsafe_allow_html=True)


#Stying options
#st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)
#st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)
