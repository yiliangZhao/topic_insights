import requests
import json

# Define the URL
url = "https://topicinsights-a6elquvi7a-uc.a.run.app/predict_topic/"

# Define the header
headers = {'Content-Type': 'application/json'}

with open('testing_doc1.txt', 'r') as f:
    text = f.read()
# Define the data you want to send
data = {
    "text": text
}

# Convert the data to JSON format
data_json = json.dumps(data)

# Send the POST request
response = requests.post(url, data=data_json, headers=headers)

# Print the response
print(response.json())