# Disaster Response Pipeline Project

### Repo Url

https://github.com/roysakura/DS-DisasterResponse


### Projects Description

This is a machine learning web application project. Consist of three main modules.

1. data module. The main feature for this module is to process the raw data from front end, clean and preprocess the data for machine learning model, then save these clean data to database

2. models module. This module is machine learning module, it used the clean data from database to train a model for predicting disaster message category.

3. app module. This app module use the trained model to predict new message for usage. 

4. In this project, you need to follow the following instrucionts for implementation.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database and the table name
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db disaster_table`
    - To run ML pipeline that trains classifier and savespython data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        `python models/train_classifier.py data/DisasterResponse.db disaster_table models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
