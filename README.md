# Cloud Run Microservice for Topic Insights

## Prerequisite
* Set the following environment variables:
```
export PROJECT_ID=<>
export HOST=<>
export IMAGE_NAME=topicinsights
export REGION=us-central1 
export IMAGE_URL=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest
export SERVICE_NAME=topicinsights
```
* [Optional] Run `python setup.py` to create the LDA models and the necessary python objects, which will be used by the web servers.
  
## Unit testing
Run `python -m unittest discover tests` 

## Local Deployment
* Run 
```
python3.9 -m venv env
source env/bin/activate
pip install -r requirements.txt
python app.py
```
* Update the `HOST` environment variable
* Run `python test_web_services.py`
  
## Cloud Deployment
* Run
  ```
  make build_image
  make push_image
  deploy.sh
  ```
* Note down the URL of the service after the deployment is done.

## API Documentation
### Overview
This API provides services related to topics. It supports the following queries:
- Given any input sentence or paragraph, return the extracted themes from the text document.
- Given a theme, return a list of relevant sentences or paragraphs under the same theme.
- Given a theme, return statistics to show the change in theme popularity across time. 
- Given a theme, return statistics to show the change in sentiment across time. 

### Endpoints
#### GET /
Returns a status message.
**Response**
A JSON object containing a status message.
```
json
{
    "Status": "OK"
}
```
#### GET /topics
Returns a JSON object of topics.
**Response**
A JSON object containing topics.
json
```
{
    {
    0: "0.014*"operating" + 0.013*"property" + 0.012*"real" + 0.012*"debt" + 0.011*"estate"",
    1: "0.014*"company" + 0.012*"business" + 0.008*"president" + 0.008*"global" + 0.007*"information"",
    2: "0.016*"revenue" + 0.013*"development" + 0.013*"customer" + 0.011*"technology" + 0.011*"data"",
    3: "0.046*"million" + 0.024*"income" + 0.016*"interest" + 0.015*"total" + 0.015*"asset"",
    ...
}
}
```
#### GET /topic_popularity/
Returns the popularity of a specific topic.
**Parameters**
- topic_id: The ID of the topic.
**Response**
A JSON object containing the popularity of a specific topic. The name of the topic is a mixture of word tokens with related weights. Popularity contains the score for each year and trend is the slope of the linear regression fit. 
json
```
{
name: "0.016*"revenue" + 0.013*"development" + 0.013*"customer" + 0.011*"technology" + 0.011*"data"",
popularity: {
    2019: 91.65511,
    2020: 92.5218,
    2021: 85.95182,
    2022: 88.25761,
    2023: 70.97163
    },
trend: -4.56311624
}
```
#### GET /topic_sentiment/
Returns the sentiment of a specific topic.
**Parameters**
- topic_id: The ID of the topic.
**Response**
A JSON object containing the sentiment of a specific topic. The name of the topic is a mixture of word tokens with related weights. Positive/negative sentiment contains the sentiment scores for each year and the trend is the slope of the linear regression fit.
json
```
{
name: "0.016*"revenue" + 0.013*"development" + 0.013*"customer" + 0.011*"technology" + 0.011*"data"",
negative sentiment: {
    scores: {
        2019: -0.00003,
        2020: 0,
        2021: 0,
        2022: 0,
        2023: -0.00001
    },
    trend: 0.00000462
    },
positive sentiment: {
    scores: {
        2019: 0.08939,
        2020: 0.09064,
        2021: 0.09082,
        2022: 0.09235,
        2023: 0.0908
    },
    trend: 0.00045193
    }
}
```
#### POST /predict_topic/
Predicts the topic of a given text.
**Parameters**
- text: The text to predict the topic of.
**Response**
A JSON object containing the predicted topic of the given text. The key is the name of the topic, which is a mixture of word tokens with related weights and the value is the proportion of that topic.
json
```
{
   {'0.014*"company" + 0.012*"business" + 0.008*"president" + 0.008*"global" + 0.007*"information"': 0.3679, '0.016*"revenue" + 0.013*"development" + 0.013*"customer" + 0.011*"technology" + 0.011*"data"': 0.0691, 
   '0.020*"content" + 0.016*"business" + 0.015*"data" + 0.012*"service" + 0.011*"advertising"': 0.0615, '0.024*"business" + 0.018*"financial" + 0.013*"adversely" + 0.012*"result" + 0.011*"affect"': 0.0509, 
   '0.033*"risk" + 0.021*"capital" + 0.019*"financial" + 0.018*"management" + 0.015*"credit"': 0.2009, '0.042*"insurance" + 0.019*"loss" + 0.016*"business" + 0.015*"company" + 0.014*"reinsurance"': 0.2419}
}
```
#### GET /get_documents/
Returns documents related to a specific topic.
**Parameters**
- topic_id: The ID of the topic.
- threshold: The threshold for the confidence score.
**Response**
A JSON object containing documents related to a specific topic. Each entry in the list contains the name of the company, filing date and filing document with the strength of the topic requested. It also contains the name of the topic in the form of a mixture of word tokens.
json
```
{
    documents: [
        {
            filedAt: "2023-02-13T16:16:13-05:00",
            name: "CADENCE DESIGN SYSTEMS INC",
            score: 0.6480688452720642,
            section: "Section1"
        },
        {
            filedAt: "2022-02-22T16:16:05-05:00",
            name: "CADENCE DESIGN SYSTEMS INC",
            score: 0.6471912860870361,
            section: "Section1A"
        },
        {
            filedAt: "2021-02-22T17:01:48-05:00",
            name: "CADENCE DESIGN SYSTEMS INC",
            score: 0.6726188063621521,
            section: "Section1"
        }],
    topic: "0.016*"revenue" + 0.013*"development" + 0.013*"customer" + 0.011*"technology" + 0.011*"data""
}
```
## License

This library is licensed under Apache 2.0. Full license text is available in [LICENSE](LICENSE).
