import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Ingest 'messages' and 'categores' files,  merge on id, return df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id',how='inner')

    return df


def clean_data(df):
    """
    input df: merged messages and categories data
    cleaning: 
        1. separate 'categories' column into each col per category
        2. strip category value into int
        3. remove duplicates
    output df: clean df
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)

    # Use the first row of categories dataframe to create column names for the categories data.
    category_colnames =  categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`, and concat categories
    df = df.drop('categories', axis=1)
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df

def save_data(df, database_filename):
    """
    Push dataframe into SQLAlchemy's table
    No output
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('main_table', engine, index=False, if_exists = 'replace')

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