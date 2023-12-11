import unittest
import os
import numpy as np
import pickle
import pandas as pd
from typing import List, Tuple
from gensim.models import LdaModel
from gensim import corpora
from datetime import datetime
from textblob import TextBlob
import sqlite3
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from string import punctuation
from nltk.corpus import words
from setup import load_data, topic_modeling, getTopics, compute_metrics, get_topic_trend
from setup import save_topic_trend, get_sentiment, get_sentiment_trend, save_sentiment_trend
from setup import topic_to_doc_mapping, write_to_db
from utils import clean

class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.df_pos = pd.DataFrame({'topic1': [0.1, 0.2, 0.3], 'topic2': [0.4, 0.5, 0.6]})
        self.df_neg = pd.DataFrame({'topic1': [0.7, 0.8, 0.9], 'topic2': [1.0, 1.1, 1.2]})
        self.topic_dict = {'topic1': 'Topic 1', 'topic2': 'Topic 2'}
        self.file = 'test.pkl'
        self.df = pd.DataFrame({'companyName': ['Company1', 'Company2'], 'filedAt': ['2020-01-01', '2020-02-01'], 
                                'topic1': [{'0': 0.1, '1': 0.2}, {'0': 0.3, '1': 0.4}], 
                                'topic1A': [{'0': 0.5, '1': 0.6}, {'0': 0.7, '1': 0.8}], 
                                'topic7': [{'0': 0.9, '1': 1.0}, {'0': 1.1, '1': 1.2}]})
        self.db = 'test.db'
        self.table_name = 'test_table'


    def test_load_data(self):
        df, doc_clean = load_data('tests/test.csv')

        # Check if the function returns the correct types
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(doc_clean, List)

        # Check if the DataFrame has the expected columns
        expected_columns = ['ticker', 'companyName', 'formType', 'description', 'filedAt', 'linkToFilingDetails', 
        'Section1', 'Section1A', 'Section7', 'filedAt_dt', 'year', 'month', 'Section1_clean', 'Section1A_clean', 'Section7_clean']        
        self.assertListEqual(list(df.columns), expected_columns)

        # Check if the length of the list of cleaned documents matches the total number of documents in the DataFrame
        self.assertEqual(len(doc_clean), len(df['Section1_clean']) + len(df['Section1A_clean']) + len(df['Section7_clean']))


    def test_topic_modeling(self):
        data = [['hello', 'world'], ['goodbye', 'world']]
        n_topics = 2
        ldamodel, dictionary = topic_modeling(data, n_topics)

        # Check if the returned objects are of correct type
        self.assertIsInstance(ldamodel, LdaModel)
        self.assertIsInstance(dictionary, corpora.Dictionary)

        # Check if the number of topics in the model is correct
        self.assertEqual(len(ldamodel.get_topics()), n_topics)

        # Check if the dictionary is correctly formed
        self.assertEqual(len(dictionary.token2id), 3)  # 3 unique words in data
        self.assertIn('hello', dictionary.token2id)
        self.assertIn('world', dictionary.token2id)
        self.assertIn('goodbye', dictionary.token2id)

    def test_getTopics(self):
        # Create a dummy LDA model and dictionary
        dictionary = corpora.Dictionary([['test', 'doc']])
        model = LdaModel(corpus=[dictionary.doc2bow(['test', 'doc'])], id2word=dictionary, num_topics=1)
        
        # Test the function
        topics = getTopics(model, dictionary, 'test doc')
        self.assertEqual(len(topics), 1)
        self.assertEqual(type(topics[0]), tuple)
        self.assertEqual(len(topics[0]), 2)
        self.assertEqual(type(topics[0][0]), int)
        self.assertEqual(type(topics[0][1]), np.float32)


    def test_compute_metrics(self):
        # Create a dummy time series
        time_series = pd.Series([1, 2, 3, 4, 5])
        
        # Test the function
        slope = compute_metrics(time_series)
        self.assertEqual(slope, 1.0)


    def test_get_topic_trend(self):
        # Create a dummy DataFrame
        df = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'topic1': [[(0, 0.5)], [(0, 0.3)], [(0, 0.2)]],
            'topic1A': [[(1, 0.5)], [(1, 0.7)], [(1, 0.8)]],
            'topic7': [[(2, 0.5)], [(2, 0.3)], [(2, 0.2)]]
        })
        
        # Test the function
        new_df = get_topic_trend(df)
        self.assertEqual(new_df.shape, (3, 3))
        self.assertEqual(new_df.sum().sum(), 4.0)
        self.assertEqual(new_df.index.tolist(), [2000, 2001, 2002])


    def test_save_topic_trend(self):
        # Mocking the required global variables and functions
        file = 'data/test.pkl'
        topic_dict = {0: 'topic1', 1: 'topic2'}

        global compute_metrics
        compute_metrics = lambda x: 0.5

        df = pd.DataFrame({
            0: [0.1, 0.2, 0.3],
            1: [0.2, 0.3, 0.4]
        }, index=[2000, 2001, 2002])

        save_topic_trend(df, topic_dict, file)

        with open(file, 'rb') as f:
            result = pickle.load(f)

        expected_result = {
            0: {'popularity': {2000: 0.1, 2001: 0.2, 2002: 0.3}, 'trend': 0.1, 'name': 'topic1'},
            1: {'popularity': {2000: 0.2, 2001: 0.3, 2002: 0.4}, 'trend': 0.1, 'name': 'topic2'}
        }
        self.assertEqual(result, expected_result)

        # Clean up
        os.remove(file)


    def test_get_sentiment(self):
        text = "I love this product"
        result = get_sentiment(text)
        expected_result = TextBlob(text).sentiment.polarity
        self.assertEqual(result, expected_result)


    def test_get_sentiment_trend(self):
        df = pd.DataFrame({
            'year': [2000, 2001, 2002],
            'sentiment1': [0.1, -0.2, 0.3],
            'topic1': [[(0, 0.5)], [(1, 0.5)], [(0, 0.5)]],
            'sentiment1A': [-0.1, 0.2, -0.3],
            'topic1A': [[(1, 0.5)], [(0, 0.5)], [(1, 0.5)]],
            'sentiment7': [0.1, -0.2, 0.3],
            'topic7': [[(0, 0.5)], [(1, 0.5)], [(0, 0.5)]]
        })

        result_pos, result_neg = get_sentiment_trend(df)
        expected_result_pos = pd.DataFrame({
            0: [0.1, 0.2, 0.3],
            1: [np.nan, np.nan, np.nan]
        }, index=pd.Index([2000, 2001, 2002], name='year'))

        expected_result_neg = pd.DataFrame({
            0: [np.nan, np.nan, np.nan],
            1: [-0.1, -0.2, -0.3]
        }, index=pd.Index([2000, 2001, 2002], name='year'))
        pd.testing.assert_frame_equal(result_pos, expected_result_pos)
        pd.testing.assert_frame_equal(result_neg, expected_result_neg)


    def test_save_sentiment_trend(self):
        save_sentiment_trend(self.df_pos, self.df_neg, self.topic_dict, self.file)
        with open(self.file, 'rb') as f:
            topic_sentiment = pickle.load(f)
        self.assertEqual(len(topic_sentiment), 2)
        self.assertEqual(set(topic_sentiment.keys()), set(self.topic_dict.keys()))
        os.remove(self.file)


    def test_topic_to_doc_mapping(self):
        df_transformed = topic_to_doc_mapping(self.df)
        self.assertEqual(df_transformed.shape, (6, 15))
        self.assertEqual(set(df_transformed.columns), set(['companyName', 'filedAt', 'section'] + ['topic' + str(i) for i in range(12)]))


    def test_write_to_db(self):
        df_transformed = topic_to_doc_mapping(self.df)
        write_to_db(df_transformed, self.db, self.table_name)
        conn = sqlite3.connect(self.db)
        df_from_db = pd.read_sql(f'SELECT * FROM {self.table_name}', conn)
        self.assertEqual(df_transformed.shape, df_from_db.shape)
        self.assertEqual(set(df_transformed.columns), set(df_from_db.columns))
        conn.close()
        os.remove(self.db)


class TestClean(unittest.TestCase):
    def setUp(self):
        self.doc = 'Item 1. This is a test document. Item 1A. It contains English and non-English words. Item 7. Lemmatization is needed.'
        self.stop = set(stopwords.words('english'))
        self.exclude = set(punctuation)
        self.english_words = set(words.words())
        self.lemma = WordNetLemmatizer()

    def test_clean(self):
        cleaned_doc = clean(self.doc)
        self.assertIsInstance(cleaned_doc, str)
        self.assertNotIn('Item 1.', cleaned_doc)
        self.assertNotIn('Item 1A.', cleaned_doc)
        self.assertNotIn('Item 7.', cleaned_doc)
        self.assertNotIn('a', cleaned_doc)
        self.assertNotIn('is', cleaned_doc)
        self.assertNotIn('.', cleaned_doc)
        self.assertNotIn('lemmatization', cleaned_doc)
        self.assertIn('test', cleaned_doc)
        self.assertIn('document', cleaned_doc) 

if __name__ == '__main__':
    unittest.main()