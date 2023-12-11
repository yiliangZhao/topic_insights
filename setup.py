import numpy as np
import pandas as pd
import pickle
import time
from gensim.models import LdaModel
from gensim import corpora
from datetime import datetime
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from textblob import TextBlob
from typing import List, Tuple
import sqlite3
from utils import clean, DB, TABLE_NAME, MODEL_PATH, DICTIONARY_PATH, TOPICS_PATH, TOPIC_POPULARITY_PATH, TOPIC_SENTIMENT_PATH


def load_data(file: str = 'data/all_filings_and_sections.csv') -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Load data from a CSV file, preprocess it and return a DataFrame and a list of cleaned documents.

    This function reads a CSV file into a DataFrame, performs several preprocessing steps such as 
    converting date strings to datetime objects, extracting year and month from the date, applying 
    a cleaning function to several columns, and splitting the cleaned text into words. It then 
    returns the preprocessed DataFrame and a list of cleaned documents.

    Parameters:
    file (str): The path to the CSV file to load. Defaults to 'data/all_filings_and_sections.csv'.

    Returns:
    df (pd.DataFrame): The preprocessed DataFrame.
    doc_clean (List[List[str]]): A list of cleaned documents, where each document is a list of words.
    """
    df = pd.read_csv(file).drop('Unnamed: 0', axis=1)

    df['filedAt_dt'] = df['filedAt'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z"))
    df['year'] = df['filedAt_dt'].apply(lambda x: x.year)
    df['month'] = df['filedAt_dt'].apply(lambda x: x.month)

    for item in ['1', '1A', '7']:
        df[f'Section{item}_clean'] = df[f'Section{item}'].apply(clean)

    df = df.dropna()
    documents1, documents2, documents3 = df['Section1_clean'], df['Section1A_clean'], df['Section7_clean']

    doc_clean1 = [doc.split() for doc in documents1] 
    doc_clean2 = [doc.split() for doc in documents2] 
    doc_clean3 = [doc.split() for doc in documents3]

    doc_clean = doc_clean1 + doc_clean2 + doc_clean3
    return df, doc_clean


def topic_modeling(data: List[List[str]], n_topics: int = 12) -> Tuple[LdaModel, corpora.Dictionary]:
    """
    Perform topic modeling on the given data using Latent Dirichlet Allocation (LDA).

    This function creates a term dictionary of the corpus, converts the list of documents into a 
    Document Term Matrix using the dictionary, and then trains an LDA model on the document term matrix.

    Parameters:
    data (List[List[str]]): A list of documents, where each document is a list of words.
    n_topics (int): The number of topics to be extracted from the LDA model. Defaults to 12.

    Returns:
    ldamodel (LdaModel): The trained LDA model.
    dictionary (corpora.Dictionary): The term dictionary of the corpus.
    """
    # Creating the term dictionary of our corpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(data)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in data]

    # Creating the object for LDA model using gensim library
    Lda = LdaModel

    # Running and Training LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word = dictionary, passes=30)
    return ldamodel, dictionary


def getTopics(model: LdaModel, dictionary: corpora.Dictionary, doc: str) -> List[Tuple[int, float]]:
    """
    Get the topic distribution for a new document using a trained LDA model and a term dictionary.

    This function preprocesses the new document, creates a new corpus using the term dictionary, 
    and then uses the LDA model to get the topic distribution for the new document.

    Parameters:
    model (LdaModel): The trained LDA model.
    dictionary (corpora.Dictionary): The term dictionary of the corpus.
    doc (str): The new document.

    Returns:
    topics (List[Tuple[int, float]]): The topic distribution for the new document, 
    where each element is a tuple containing a topic ID and the corresponding probability.
    """
    # Preprocess the new document
    new_text = clean(doc).split()
    
    # Create a new corpus
    new_corpus = dictionary.doc2bow(new_text)
    
    # Use the LDA model to get the topic
    topics = model.get_document_topics(new_corpus)
    return topics


def compute_metrics(time_series: pd.Series) -> float:
    """
    Compute the slope of the given time series using linear regression.

    This function reshapes the time series into a 2D array, fits a linear regression model to the data, 
    and then computes the slope of the fitted line. The slope is rounded to 8 decimal places.

    Parameters:
    time_series (Series): The time series data.

    Returns:
    slope (float): The slope of the fitted line, rounded to 8 decimal places.
    """
    # Compute slope
    x = np.array(range(len(time_series))).reshape(-1, 1)
    y = time_series.values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    return np.round(slope, 8)[0]


def get_topic_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a DataFrame with topics and their proportions for each document and returns a DataFrame 
    with years as rows and topics as columns. The values in the DataFrame represent the sum of proportions for 
    each topic in a given year.
    
    The input DataFrame is expected to have the following columns:
    - 'year': the year the document was published
    - 'topic1', 'topic1A', 'topic7': each of these columns contains a list of tuples. Each tuple consists of a 
    topic and its proportion in the document.
    
    The output DataFrame is sorted by year in ascending order and the year is set as the index.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with topics and their proportions for each document.
    
    Returns:
    new_df (pd.DataFrame): A DataFrame with years as rows and topics as columns. The values in the DataFrame 
    represent the sum of proportions for each topic in a given year.
    """
    # defaultdict is used to avoid KeyError when a new key is accessed
    data = defaultdict(lambda: defaultdict(int))

    # Iterate over the rows in the DataFrame
    # Each row represents a document with topics and their proportions
    for _, row in df.iterrows():
        # For each topic in 'topic1', 'topic1A', 'topic7' columns, 
        # add its proportion to the corresponding year in the data dictionary
        for topic, proportion in row['topic1']:
            data[row['year']][topic] += proportion
        for topic, proportion in row['topic1A']:
            data[row['year']][topic] += proportion
        for topic, proportion in row['topic7']:
            data[row['year']][topic] += proportion

    # Convert the dictionary to a DataFrame
    # Transpose the DataFrame to make years as rows and topics as columns
    new_df = pd.DataFrame(data).T.reset_index()
    
    # Rename the columns
    new_df.columns = ['year'] + new_df.columns[1:].tolist()

    # Sort the DataFrame by year in ascending order
    new_df.sort_values('year', ascending=True, inplace=True)
    # Set the year as the index
    new_df.set_index('year', inplace=True)
    return new_df


def save_topic_trend(df: pd.DataFrame, topic_dict: dict, file: str) -> None:
    """
    This function takes a DataFrame with topics as columns and years as rows, computes the popularity and trend 
    for each topic, and saves the results in a pickle file. The popularity is the sum of proportions for each 
    topic in a given year, and the trend is computed using the 'compute_metrics' function (not defined in this 
    code snippet). 
    
    The output is a dictionary where each key is a topic and the value is another dictionary with the following 
    keys:
    - 'popularity': a dictionary where each key is a year and the value is the popularity of the topic in that 
    year.
    - 'trend': the trend of the topic.
    - 'name': the name of the topic.
    
    The dictionary is saved in a pickle file at the path specified by the 'TOPIC_POPULARITY_PATH' constant 
    (not defined in this code snippet).
    
    Parameters:
    df (pd.DataFrame): A DataFrame with topics as columns and years as rows.
    topic_dict (dict): A dictionary where each key is a topic id and the value is the name of the topic.

    Returns:
    None
    """
    topic_popularity = {}

    for col in df.columns:
        value_obj = {}
        temp_dict = df[col].to_dict()
        value_obj['popularity'] = {k: round(v, 5) for k, v in temp_dict.items()}
        slope = compute_metrics(df[col])
        value_obj['trend'] = slope
        value_obj['name'] = topic_dict[col]
        topic_popularity[col] = value_obj

    # Save dictionary to a pickle file
    with open(file, 'wb') as f:
        pickle.dump(topic_popularity, f)


def get_sentiment(text: str) -> float:
    """
    This function takes a string as input and uses the TextBlob library to determine the sentiment polarity of the text.
    The sentiment polarity is a float within the range [-1.0, 1.0] where -1.0 is very negative, 0 is neutral and 1.0 is very positive.

    Args:
        text (str): The text to analyze.

    Returns:
        float: The sentiment polarity of the text.
    """
    blob = TextBlob(text)
    return blob.sentiment.polarity


def get_sentiment_trend(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function calculates the sentiment trend for different topics over the years. 
    It takes a DataFrame as input where each row represents a document with its sentiment and topics. 
    The sentiment is calculated for each topic and normalized by the total proportion of the topic in each year.

    Args:
        df (pd.DataFrame): The input DataFrame. Each row represents a document with columns for sentiment and topics.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames. The first DataFrame contains the positive sentiment trend for each topic over the years. 
        The second DataFrame contains the negative sentiment trend for each topic over the years.
    """
    # Initialize a dictionary to store the sum of proportions for each topic in each year and month
    data_pos = defaultdict(lambda: defaultdict(int))
    data_neg = defaultdict(lambda: defaultdict(int))
    data_count = defaultdict(lambda: defaultdict(int))

    # Iterate over the rows in the DataFrame
    for _, row in df.iterrows():
        for sentiment_key, topic_key in [('sentiment1', 'topic1'), ('sentiment1A', 'topic1A'), ('sentiment7', 'topic7')]:
            sentiment = row[sentiment_key]
            for topic, proportion in row[topic_key]:
                if sentiment > 0:
                    data_pos[row['year']][topic] += (proportion * sentiment)
                else:
                    data_neg[row['year']][topic] += (proportion * sentiment)
                data_count[row['year']][topic] += proportion

    # Convert the dictionary to a DataFrame
    new_df_pos = pd.DataFrame(data_pos).T.reset_index()
    new_df_pos.columns = ['year'] + new_df_pos.columns[1:].tolist()

    # Convert year and month to datetime for plotting
    new_df_pos.sort_values('year', ascending=True, inplace=True)
    # Set the date as the index
    new_df_pos.set_index('year', inplace=True)

    # Convert the dictionary to a DataFrame
    new_df_neg = pd.DataFrame(data_neg).T.reset_index()
    new_df_neg.columns = ['year'] + new_df_neg.columns[1:].tolist()

    # Convert year and month to datetime for plotting
    new_df_neg.sort_values('year', ascending=True, inplace=True)
    # Set the date as the index
    new_df_neg.set_index('year', inplace=True)

    # Convert the dictionary to a DataFrame
    new_df_cnt = pd.DataFrame(data_count).T.reset_index()
    new_df_cnt.columns = ['year'] + new_df_cnt.columns[1:].tolist()

    # Convert year and month to datetime for plotting
    new_df_cnt.sort_values('year', ascending=True, inplace=True)
    # Set the date as the index
    new_df_cnt.set_index('year', inplace=True)

    normalized_df_pos = new_df_pos / new_df_cnt
    normalized_df_neg = new_df_neg / new_df_cnt

    return normalized_df_pos, normalized_df_neg


def save_sentiment_trend(df_pos: pd.DataFrame, df_neg: pd.DataFrame, topic_dict: dict, file: str) -> None:
    """
    This function saves the sentiment trend for different topics into a pickle file. 
    It takes two DataFrames as input where each DataFrame represents the positive and negative sentiment trend for each topic over the years.

    Args:
        df_pos (pd.DataFrame): The DataFrame containing the positive sentiment trend for each topic over the years.
        df_neg (pd.DataFrame): The DataFrame containing the negative sentiment trend for each topic over the years.
        topic_dict (dict): A dictionary where each key is a topic id and the value is the name of the topic.

    Returns:
        None
    """
    topic_sentiment = {}

    for col in df_pos.columns:
        value_obj = {}
        temp_dict = df_pos[col].to_dict()
        value_obj['positive sentiment'] = {}
        value_obj['positive sentiment']['scores'] = {k: round(v, 5) for k, v in temp_dict.items()}
        slope = compute_metrics(df_pos[col])
        value_obj['positive sentiment']['trend'] = slope

        temp_dict = df_neg[col].to_dict()
        value_obj['negative sentiment'] = {}
        value_obj['negative sentiment']['scores'] = {k: round(v, 5) for k, v in temp_dict.items()}
        slope = compute_metrics(df_neg[col])
        value_obj['negative sentiment']['trend'] = slope

        value_obj['name'] = topic_dict[col]
        topic_sentiment[col] = value_obj

    # Save dictionary to a pickle file
    with open(file, 'wb') as f:
        pickle.dump(topic_sentiment, f)


def topic_to_doc_mapping(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """
    This function takes a DataFrame and an integer as input, and returns a new DataFrame with specific transformations.
    The transformations include selecting specific columns, renaming them, adding a new column with a fixed value, 
    and creating a dictionary with specific columns and a range of 'topic' columns. The function then creates a new 
    DataFrame from this dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame. It should contain 'companyName', 'filedAt', 'topic1', 'topic1A', and 'topic7' columns.
    n (int, optional): The range for 'topic' columns. Default is 12.

    Returns:
    pd.DataFrame: The transformed DataFrame. It contains 'companyName', 'filedAt', 'section', and 'topic' + str(i) columns.
    """
    df1 = df[['companyName', 'filedAt', 'topic1']].copy()
    df1.columns = ['companyName','filedAt', 'topics']
    df1['section'] = 'Section1'

    df2 = df[['companyName', 'filedAt', 'topic1A']].copy()
    df2.columns = ['companyName','filedAt', 'topics']
    df2['section'] = 'Section1A'

    df3 = df[['companyName', 'filedAt', 'topic7']].copy()
    df3.columns = ['companyName','filedAt', 'topics']
    df3['section'] = 'Section7'

    df_combined = pd.concat([df1, df2, df3])

    # Create a dictionary with 'companyName', 'filedAt', 'section' and all 'topic' + str(i) columns
    data = {**{'companyName': df_combined['companyName'], 'filedAt': df_combined['filedAt'], 'section': df_combined['section']}, 
            **{'topic' + str(i): df_combined['topics'].apply(lambda x: dict(x).get(i, 0)) for i in range(n)}}

    # Create a new DataFrame new_df from the dictionary
    return pd.DataFrame(data)


def write_to_db(df: pd.DataFrame, db: str, table_name: str) -> None:
    """
    Writes the contents of a DataFrame to a SQLite database table. If the table does not exist, it will be created.
    If the table already exists, its contents will be replaced.

    Args:
        df (pd.DataFrame): The DataFrame to be written to the SQLite database.
        db (str): The name of the SQLite database.
        table_name (str): The name of the table in the SQLite database.

    Returns:
        None
    """
    # Establish a connection to the SQLite database
    # If the database does not exist, it will be created
    conn = sqlite3.connect(db)

    # Define a SQL query to create a new table in the SQLite database
    # The table will have columns for 'companyName', 'FiledAt', 'section', and 'topic0' to 'topic11'
    # If the table already exists, the query will do nothing
    query = f'''CREATE TABLE IF NOT EXISTS {table_name} (companyName text, FiledAt text, section text, topic0 real, topic1 real, topic2 real, 
                topic3 real, topic4 real, topic5 real, topic6 real, \
                topic7 real, topic8 real, topic9 real, topic10 real, topic11 real)'''

    # Execute the SQL query
    conn.execute(query)

    # Write the contents of the DataFrame 'new_df' to the SQLite table
    # If the table already exists, its contents will be replaced
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # Commit the current transaction
    conn.commit()

    # Close the connection to the SQLite database
    conn.close()


if __name__ == '__main__':
    print('Running setup.py')
    start = time.time()
    df, doc_clean = load_data()
    ldamodel, dictionary = topic_modeling(doc_clean)

    # Print the topics
    topics = ldamodel.print_topics(num_topics=12, num_words=5)
    for topic in topics:
        print(topic)

    # Save the model and dictionary
    ldamodel.save(MODEL_PATH)
    dictionary.save(DICTIONARY_PATH)

    # Save the mapping of topic number to topic name
    topic_dict = {x[0]:x[1] for x in topics}

    # Save dictionary to a pickle file
    with open(TOPICS_PATH, 'wb') as f:
        pickle.dump(topic_dict, f)

    # Obtain the topics for each document
    for item in ['1', '1A', '7']:
        df[f'topic{item}'] = df[f'Section{item}_clean'].apply(lambda x: getTopics(ldamodel, dictionary, x))

    # Obtain the topic proportions for each year
    # new_df has years and topics as columns
    new_df = get_topic_trend(df)
    save_topic_trend(new_df, topic_dict, TOPIC_POPULARITY_PATH)

    # Obtain the sentiment for each document
    for item in ['1', '1A', '7']:
        df[f'sentiment{item}'] = df[f'Section{item}_clean'].apply(get_sentiment)

    # Obtain the sentiment trend for each topic
    normalized_df_pos, normalized_df_neg = get_sentiment_trend(df)

    # Save the sentiment trend for each topic
    save_sentiment_trend(normalized_df_pos, normalized_df_neg, topic_dict, TOPIC_SENTIMENT_PATH)

    # Create a new DataFrame with 'companyName', 'filedAt', 'section', and 'topic' + str(i) columns
    df_topic_doc = topic_to_doc_mapping(df)

    # Write the DataFrame to the SQLite database
    write_to_db(df_topic_doc, DB, TABLE_NAME)

    print('Done!')
    print('Time taken:', time.time() - start, 'seconds')
