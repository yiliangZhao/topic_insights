import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('wordnet')

import string
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
from gensim import corpora

DB = 'mydb.sqlite'
TABLE_NAME = 'document_topics'
MODEL_PATH = 'models/lda.model'
DICTIONARY_PATH = 'models/dictionary.model'
TOPICS_PATH = 'data/topics.pkl'
TOPIC_POPULARITY_PATH = 'data/topic_popularity.pkl'
TOPIC_SENTIMENT_PATH = 'data/topic_sentiment.pkl'

stop = set(stopwords.words('english'))
stop.add('also')
stop.add('could')

english_words = set(words.words())
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()


def clean(doc: str) -> str:
    """
    This function cleans the input document by removing specific strings, 
    stop words, punctuation, non-English words, and lemmatizing the words.
    
    Parameters:
    doc (str): The input document to be cleaned.
    
    Returns:
    str: The cleaned document.
    """
    # Check if the input is a string
    if not isinstance(doc, str):
        return None
    
    # Remove specific strings from the document
    doc = doc.replace('Item 1.', '')
    doc = doc.replace('Item 1A.', '')
    doc = doc.replace('Item 7.', '')
    
    # Remove stop words from the document
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    
    # Remove punctuation from the document
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    
    # Keep only English words in the document
    english = " ".join(word for word in punc_free.split() if word in english_words)
    
    # Lemmatize the words in the document and remove words with length less than or equal to 3
    normalized = " ".join(lemma.lemmatize(word) for word in english.split() if len(word) > 3)
    
    return normalized
