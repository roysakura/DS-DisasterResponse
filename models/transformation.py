from sklearn.preprocessing import FunctionTransformer,LabelEncoder,OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [t.lower().strip() for t in tokens]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
            
class MyLabelBinarizer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return pd.DataFrame(self.encoder.transform(x))