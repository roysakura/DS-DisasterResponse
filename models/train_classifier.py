# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import FunctionTransformer,LabelEncoder,OneHotEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from P2.models.transformation import ColumnSelector,MyLabelBinarizer,tokenize

def load_data(database_filepath,table):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * from {};'.format(table),engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'],axis=1)
    
    category_names = y.columns.tolist()
    
    return X,y,category_names


def build_model():
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),('tfidf', TfidfTransformer(norm='l1')),('clf',MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline

def optimize_model(model,X_train,Y_train,X_test,Y_test,category_names):
    parameters = {
              'features__text_pileline__tfidf__norm':['l1'],
              'features__text_pileline__tfidf__sublinear_tf':[True],
              'clf__estimator__max_depth':[3],
              'clf__estimator__max_features':[3],
              'clf__estimator__n_estimators':[100]
              }

    cv = GridSearchCV(estimator=model, param_grid=parameters,cv=5)
    be = cv.fit(X_train,Y_train.values).best_estimator_
    print(cv.best_params_)
    print(cv.best_score_)
    pred = be.predict(X_test)
    for i,column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column].values,pred[:,i]))

    model = be

def evaluate_model(model, X_test, Y_test, category_names):
    pred = model.predict(X_test)
    for i,column in enumerate(category_names):
        print(column)
        print(classification_report(Y_test[column].values,pred[:,i]))

def save_model(model, model_filepath):
    filename = '{}'.format(model_filepath)
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 4:
        database_filepath, table, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath,table)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        #print('Optimizing model...')
        #optimize_model(model, X_train, Y_train, X_test, Y_test,category_names)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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