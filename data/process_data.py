# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load data from raw data source

    Arg:
    messages_filepath: the raw message source
    categories_filepath: the category item source

    Return:

    a dataframe object
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories,on='id')
    
    return df


def clean_data(df):
    '''
    Clean up the dataframe for later processing
    '''

    #Add categories columns as target
    categories = df['categories'].str.split(';',expand=True)
    #Use first row for new category column name
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    #change category value as needed
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
        
    #merge back to original dataset    
    df = pd.concat([df.drop(['categories'],axis=1),categories],axis=1)
    
    #drop duplicate items
    df.drop_duplicates(inplace=True)
    
    return df
    
def save_data(df, database_filename,table_name):

    '''
    Save clean data to database

    Arg:
    df: clean dataframe
    database_filename: database location
    table_name: table for storing


    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('{}'.format(table_name), engine, index=False)  


def main():

    '''
    Data Processing procedure.
    
    '''
    if len(sys.argv) == 5:

        messages_filepath, categories_filepath, database_filepath, table_name = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath,table_name)
        
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