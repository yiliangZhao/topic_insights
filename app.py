import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from utils import clean, DB, TABLE_NAME, MODEL_PATH, DICTIONARY_PATH, TOPICS_PATH, TOPIC_POPULARITY_PATH, TOPIC_SENTIMENT_PATH
import sqlite3

app = Flask(__name__)

# Load the topics dictionary
with open(TOPICS_PATH, 'rb') as f:
    topics_dict = pickle.load(f)

# Load the topic popularity dictionary
with open(TOPIC_POPULARITY_PATH, 'rb') as f:
    topic_popularity_dict = pickle.load(f)

# Load the topic sentiment dictionary
with open(TOPIC_SENTIMENT_PATH, 'rb') as f:
    topic_sentiment_dict = pickle.load(f)

# Load the LDA model and the dictionary
ldamodel = LdaModel.load(MODEL_PATH)
dictionary = Dictionary.load(DICTIONARY_PATH)


@app.route("/", methods=['GET'])
def read_root() -> dict:
    """Returns a status message."""
    return {"Status": "OK"}


@app.route("/topics", methods=['GET'])
def return_topics() -> jsonify:
    """Returns a JSON object of topics."""
    return jsonify(topics_dict)


@app.route("/topic_popularity/", methods=['GET'])
def return_popularity() -> jsonify:
    """Returns the popularity of a specific topic."""
    topic_id = request.args.get('topic_id')
    if topic_id is None:
        return {'response':f'Error: No topic_id field provided. Please specify a topic_id from {list(topics_dict.keys())}'}
    topic_id = int(topic_id)
    if topic_id not in topics_dict.keys():
        return {'response':f'topic: {topic_id} is not found.\n Please pick a topic number from {list(topics_dict.keys())}'}
    return jsonify(topic_popularity_dict[topic_id])


@app.route("/topic_sentiment/", methods=['GET'])
def return_sentiment() -> jsonify:
    """Returns the sentiment of a specific topic."""
    topic_id = request.args.get('topic_id')
    if topic_id is None:
        return {'response':f'Error: No topic_id field provided. Please specify a topic_id from {list(topics_dict.keys())}'}
    topic_id = int(topic_id)
    if topic_id not in topics_dict.keys():
        return {'response':f'topic: {topic_id} is not found.\n Please pick a topic number from {list(topics_dict.keys())}'}
    return jsonify(topic_sentiment_dict[topic_id])


@app.route("/predict_topic/", methods=['POST'])
def predict_topic() -> jsonify:
    """Predicts the topic of a given text."""
    sentence = request.json['text']
    processed_sentence = clean(sentence)
    new_corpus = dictionary.doc2bow(processed_sentence.split())
    topics = ldamodel.get_document_topics(new_corpus)

    response = {}
    for topic in topics:
        response[topics_dict[topic[0]]] = round(float(topic[1]), 4)
    
    return jsonify(dict(sorted(response.items(), key=lambda item: item[1], reverse=True)))


@app.route("/get_documents/", methods=['GET'])
def get_documents() -> jsonify:
    """Returns documents related to a specific topic."""
    topic_id = request.args.get('topic_id')
    if topic_id is None:
        return {'response':f'Error: No topic_id field provided. Please specify a topic_id from {list(topics_dict.keys())}'}
    topic_id = int(topic_id)
    if topic_id not in topics_dict.keys():
        return {'response':f'topic: {topic_id} is not found.\n Please pick a topic number from {list(topics_dict.keys())}'}
    confidence_score = request.args.get('threshold')
    if confidence_score is None:
        return {'response':'Error: No threshold field provided. Please specify a threshold between 0 and 1.'}
    confidence_score = float(confidence_score)
    conn = sqlite3.connect(DB)
    query = f'''SELECT companyName, filedAt, section, topic{topic_id} 
              FROM {TABLE_NAME} 
              WHERE topic{topic_id} >= {confidence_score}'''
    response = conn.execute(query)
    docs = []
    for item in response:    
        obj = {'name':item[0], 'filedAt':item[1], 'section':item[2], 'score':float(item[3])}
        docs.append(obj)
    conn.close()
    result = {'topic':topics_dict[topic_id], 'documents':docs}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5002)))
    