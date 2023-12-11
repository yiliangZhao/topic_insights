import os
import requests
import json

# Define the HOST URL
HOST = os.getenv('HOST', 'https://topicinsights-a6elquvi7a-uc.a.run.app') 

def test_predict_topic() -> None:
    """
    Sends a POST request to the '/predict_topic/' endpoint with a text document
    and prints the JSON response.
    """
    # Define the header
    headers = {'Content-Type': 'application/json'}
    with open('testing_doc1.txt', 'r') as f:
        text = f.read()
    # Define the data to send
    data = {
        "text": text
    }

    # Convert the data to JSON format
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(HOST + '/predict_topic/', data=data_json, headers=headers)

    # Print the response
    print(response.json())


def get_topic_popularity(topic_id: int) -> None:
    """
    Sends a GET request to the '/topic_popularity/' endpoint with a topic_id
    and prints the JSON response.

    Args:
        topic_id (int): The ID of the topic.
    """
    response = requests.get(HOST + f"/topic_popularity/?topic_id={topic_id}")
    print(response.json())


def get_topic_sentiment(topic_id: int) -> None:
    """
    Sends a GET request to the '/topic_sentiment/' endpoint with a topic_id
    and prints the JSON response.

    Args:
        topic_id (int): The ID of the topic.
    """
    response = requests.get(HOST + f"/topic_sentiment/?topic_id={topic_id}")
    print(response.json())


def get_topics() -> None:
    """
    Sends a GET request to the '/topics' endpoint and prints the JSON response.
    """
    response = requests.get(HOST + "/topics")
    print(response.json())


if __name__ == "__main__":
    test_predict_topic()
    get_topic_popularity(2)
    get_topic_sentiment(2)
    get_topics()
