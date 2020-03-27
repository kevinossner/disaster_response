#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:57:31 2020

@author: kevinossner
"""

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


# load data function
def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath: path to file with disaster response messages
        categories_filepath: path to file with categories of messages
    OUTPUT:
        pandas.DataFrame: messages and categories merged
    '''
    # load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets on id
    df = pd.merge(messages, categories, on='id')
    return df


# clean data function
def clean_data(df):
    '''
    INPUT:
        df: output of 'load_data()' function
    OUTPUT:
        pandas.DataFrame:
            - splitted 'categories' column to get one column for every
            category
            - drop columns with more than 50% of missings
    '''
    # split categories column
    cat = df['categories'].str.split(';', expand=True)
    # first row to extract column headers
    row = cat[:1]
    categories_colnames = row.apply(lambda x: x[0][:-2])
    # rename the columns
    cat.columns = categories_colnames
    # transform column values
    for column in cat:
        # column value equals last char of the string
        cat[column] = cat[column].str.slice(-1)
        # convert column from string to numeric
        cat[column] = pd.to_numeric(cat[column])
    # create new categories dataset
    df = df.drop('categories', axis=1)
    df = pd.concat([df, cat], axis=1)
    # drop columns with more than 50% of missings
    df = df.loc[:, (df.isnull().sum()/len(df)>0.5)==False]
    # drop duplicated rows
    df = df.drop_duplicates()
    return df


# save data
def save_data(df, database_filename):
    '''
    INPUT:
        df: dataframe to move to sqlite
        database_filename: string of desired database name with '.db'
    OUTPUT:
        None
    '''
    # create sqlite engine
    engine = create_engine('sqlite:///' + str/(database_filename))
    # df to sqlite file
    df.to_sql('disaster_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()


# define file paths
messages_filepath = './data/disaster_messages.csv'
categories_filepath = './data/disaster_categories.csv'


# load data
df = load_data(messages_filepath, categories_filepath)


# clean data
df_cleaned = clean_data(df)

