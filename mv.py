#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:10:17 2020

@author: ananyabanerjee

Motivational Quotes

"""
import pandas as pd
import numpy as np

DATA_URL_TRAIN= (
#"quotesdrivedb.csv"
"quotes.json"
)




#@st.cache(persist=True)
def load_data(nrows, col):
    #data=pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['CRASH_DATE',"CRASH_TIME"]])
    #Read files
    #data=pd.read_csv(DATA_URL_TRAIN)
    data=pd.read_json(DATA_URL_TRAIN)

    # Dropping duplicates and creating a list containing all the quotes
    #quotes = data[col].drop_duplicates()

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


