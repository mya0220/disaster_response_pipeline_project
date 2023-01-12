import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB

def load_data(database_filepath):
    """Load data from database"""
    engine =  create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('main_table', con = engine)
    X = df.iloc[:, 1].values
    Y = df.iloc[:, 4:-1].values
    
    return df, X, Y


def tokenize(text):
    """
    input: one message
    output: a list of clean tokens for the msg
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """Build pipeline CV+PipeLine"""
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        
            ('start_verb', StartingVerbExtractor())
        ])),
    
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'clf__estimator__alpha':[0.1,1,10],
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, cate_list):
    """
    Generate classification report
    """
    y_pred = model.predict(X_test)

    for i in range(len(Y_test[0])):
        print("category: ", cate_list[i])
        print(classification_report(Y_test[i], y_pred[i]))


def save_model(model, model_filepath):
    """Export model as pickle"""
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        df, X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        cate_list = df.iloc[:,4:-1].columns.tolist()

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, cate_list)
        
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