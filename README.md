# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Required libraries
- nltk
- numpy
- pandas
- scikit-learn
- sqlalchemy

### Motivation

This project provides an analyzis of disaster responses data from [Figure Eight](https://www.figure-eight.com/) and builds a model for an API that classifies disaster messages.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


### Files

- data/process_data.py: A data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
- model/train_classifier.py: A machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

## Acknowledgements

Thank you to [Figure Eight](https://www.figure-eight.com/) for providing the data, and thanks to [Udacity](https://www.udacity.com/) for the learning process and review.
