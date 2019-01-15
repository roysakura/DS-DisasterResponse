import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from collections import Counter

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3

nltk.download("stopwords")
stopwords_list = stopwords.words('english')
app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords_list:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)
        else:
            continue

    return clean_tokens

# load data
#engine = create_engine('sqlite:///../data/DisasterResponse.db')
#engine = create_engine('sqlite:///Users/roy/Documents/workplace/MLND-projects/DataScience/P2/data/DisasterResponse.db')
engine = sqlite3.connect('/Users/roy/Documents/workplace/MLND-projects/DataScience/P2/data/DisasterResponse.db')
#df = pd.read_sql_table('disaster_messages', engine)
df = pd.read_sql('select * from disaster_table',engine)
# load model
model = joblib.load("/Users/roy/Documents/workplace/MLND-projects/DataScience/P2/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    ## Add Category Pie Chart

    categories = df.columns[4:]
    cat_dict = {}
    for cat in categories:
        cat_dict[cat] = df[cat].sum()

    cat_df = pd.DataFrame.from_dict(cat_dict,orient='index').reset_index()
    cat_df.columns = ['label','number']

    cat_graph = {}
    cat_graph['data'] = [
        Pie(labels=cat_df.label.values.tolist(),values=cat_df.number.values,textfont=dict(size=8),pull=.3,hole=.1)
    ]
    cat_graph['layout'] = dict(title='Message Categories')
    graphs.append(cat_graph)


    ## Add Top 10 word Pie Chart

    full_text = (' ').join(df['message'].values.tolist())
    full_text = re.sub(r'[^\w]', ' ', full_text)
    tokenize_words = tokenize(full_text)
    word_freq = Counter(tokenize_words)
    word_df = pd.DataFrame.from_dict(dict(word_freq),orient='index').reset_index()
    word_df.columns = ['word','number']

    word_graph = {}
    word_graph['data'] = [
        Pie(labels=word_df[:10].word.values.tolist(),values=word_df[:10].number.values,textfont=dict(size=8),pull=.3,hole=.1)
    ]
    word_graph['layout'] = dict(title='Top 10 Message Word')
    graphs.append(word_graph)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
