#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:45:41 2020

@author: kevinossner
"""

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle


# load preprocessed data
def load_data(database_filepath):
    '''
    INPUT: 
        database_filepath: path used for importing the database     
    OUTPUT:
        X: features (message)
        Y: labels (categorie of the message)
        y.keys: label names
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df =  pd.read_sql_table('DisasterResponse', engine)
    X = df.message.values
    y = df.iloc[:,5:]
    category_names = df.columns[4:]
    return X, y, category_names


# tokenize text data
def tokenize(text):
    '''
    INPUT: 
        text: column with text  
    OUTPUT:
        clean_tokens: processed text variable (tokenized, lower case, stripped
                                               and lemmatized)
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# build ML-pipeline
def build_model(X_train,y_train):
    '''
    INPUT: 
        X_train: training features
        y_train: training labels
    OUTPUT:
        cv: pipeline model (tokenization, count vectorization, 
        TFIDTransformation and random forest)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    cv.fit(X_train,y_train)
    return cv


# test the pipeline
def evaluate_model(pipeline, X_test, Y_test, category_names):
    '''
    INPUT: 
        pipeline: ML-pipeline
        X_test: test features
        y_test: test labels
        category_names: list of the categories 
    OUTPUT:
        classification_report: performance of the pipeline
    '''
    y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names)

          
# export the model
def save_model(model, model_filepath):
    '''
    INPUT: 
        model: model to be exported
        model_filepath: path where the model will be saved
    OUTPUT:
        exported model as a pickle file
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()