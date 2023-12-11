# Cloud Run Template Microservice

A template repository for a Cloud Run microservice, written in Python

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)

## Prerequisite

* Enable the Cloud Run API via the [console](https://console.cloud.google.com/apis/library/run.googleapis.com?_ga=2.124941642.1555267850.1615248624-203055525.1615245957) or CLI:

```bash
gcloud services enable run.googleapis.com
```

## Features

* **Flask**: Web server framework
* **Buildpack support** Tooling to build production-ready container images from source code and without a Dockerfile
* **Dockerfile**: Container build instructions, if needed to replace buildpack for custom build
* **SIGTERM handler**: Catch termination signal for cleanup before Cloud Run stops the container
* **Service metadata**: Access service metadata, project ID and region, at runtime
* **Local development utilities**: Auto-restart with changes and prettify logs
* **Structured logging w/ Log Correlation** JSON formatted logger, parsable by Cloud Logging, with [automatic correlation of container logs to a request log](https://cloud.google.com/run/docs/logging#correlate-logs).
* **Unit and System tests**: Basic unit and system tests setup for the microservice
* **Task definition and execution**: Uses [invoke](http://www.pyinvoke.org/) to execute defined tasks in `tasks.py`.

## Local Development

### Cloud Code

This template works with [Cloud Code](https://cloud.google.com/code), an IDE extension
to let you rapidly iterate, debug, and run code on Kubernetes and Cloud Run.

Learn how to use Cloud Code for:

* Local development - [VSCode](https://cloud.google.com/code/docs/vscode/developing-a-cloud-run-service), [IntelliJ](https://cloud.google.com/code/docs/intellij/developing-a-cloud-run-service)

* Local debugging - [VSCode](https://cloud.google.com/code/docs/vscode/debugging-a-cloud-run-service), [IntelliJ](https://cloud.google.com/code/docs/intellij/debugging-a-cloud-run-service)

* Deploying a Cloud Run service - [VSCode](https://cloud.google.com/code/docs/vscode/deploying-a-cloud-run-service), [IntelliJ](https://cloud.google.com/code/docs/intellij/deploying-a-cloud-run-service)
* Creating a new application from a custom template (`.template/templates.json` allows for use as an app template) - [VSCode](https://cloud.google.com/code/docs/vscode/create-app-from-custom-template), [IntelliJ](https://cloud.google.com/code/docs/intellij/create-app-from-custom-template)

### CLI tooling

To run the `invoke` commands below, install [`invoke`](https://www.pyinvoke.org/index.html) system wide: 

```bash
pip install invoke
```

Invoke will handle establishing local virtual environments, etc. Task definitions can be found in `tasks.py`.

#### Local development

1. Set Project Id:
    ```bash
    export GOOGLE_CLOUD_PROJECT=<GCP_PROJECT_ID>
    ```
2. Start the server with hot reload:
    ```bash
    invoke dev
    ```

#### Deploying a Cloud Run service

1. Set Project Id:
    ```bash
    export GOOGLE_CLOUD_PROJECT=<GCP_PROJECT_ID>
    ```

1. Enable the Artifact Registry API:
    ```bash
    gcloud services enable artifactregistry.googleapis.com
    ```

1. Create an Artifact Registry repo:
    ```bash
    export REPOSITORY="samples"
    export REGION=us-central1
    gcloud artifacts repositories create $REPOSITORY --location $REGION --repository-format "docker"
    ```
  
1. Use the gcloud credential helper to authorize Docker to push to your Artifact Registry:
    ```bash
    gcloud auth configure-docker
    ```

2. Build the container using a buildpack:
    ```bash
    invoke build
    ```
3. Deploy to Cloud Run:
    ```bash
    invoke deploy
    ```

### Run sample tests

1. [Pass credentials via `GOOGLE_APPLICATION_CREDENTIALS` env var](https://cloud.google.com/docs/authentication/production#passing_variable):
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="[PATH]"
    ```

2. Set Project Id:
    ```bash
    export GOOGLE_CLOUD_PROJECT=<GCP_PROJECT_ID>
    ```
3. Run unit tests
    ```bash
    invoke test
    ```

4. Run system tests
    ```bash
    gcloud builds submit \
        --config test/advance.cloudbuild.yaml \
        --substitutions 'COMMIT_SHA=manual,REPO_NAME=manual'
    ```
    The Cloud Build configuration file will build and deploy the containerized service
    to Cloud Run, run tests managed by pytest, then clean up testing resources. This configuration restricts public
    access to the test service. Therefore, service accounts need to have the permission to issue ID tokens for request authorization:
    * Enable Cloud Run, Cloud Build, Artifact Registry, and IAM APIs:
        ```bash
        gcloud services enable run.googleapis.com cloudbuild.googleapis.com iamcredentials.googleapis.com artifactregistry.googleapis.com
        ```
        
    * Set environment variables.
        ```bash
        export PROJECT_ID="$(gcloud config get-value project)"
        export PROJECT_NUMBER="$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')"
        ```

    * Create an Artifact Registry repo (or use another already created repo):
        ```bash
        export REPOSITORY="samples"
        export REGION=us-central1
        gcloud artifacts repositories create $REPOSITORY --location $REGION --repository-format "docker"
        ```
  
    * Create service account `token-creator` with `Service Account Token Creator` and `Cloud Run Invoker` roles.
        ```bash
        gcloud iam service-accounts create token-creator

        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:token-creator@$PROJECT_ID.iam.gserviceaccount.com" \
            --role="roles/iam.serviceAccountTokenCreator"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:token-creator@$PROJECT_ID.iam.gserviceaccount.com" \
            --role="roles/run.invoker"
        ```

    * Add `Service Account Token Creator` role to the Cloud Build service account.
        ```bash
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
            --role="roles/iam.serviceAccountTokenCreator"
        ```
    
    * Cloud Build also requires permission to deploy Cloud Run services and administer artifacts: 

        ```bash
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
            --role="roles/run.admin"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
            --role="roles/iam.serviceAccountUser"
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$PROJECT_NUMBER@cloudbuild.gserviceaccount.com" \
            --role="roles/artifactregistry.repoAdmin"
        ```
# API Documentation
## Overview
This API provides services related to topics. It supports the following queries:
- Given any input sentence or paragraph, return the extracted themes from the text document.
- Given a theme, return a list of relevant sentences or paragraphs under the same theme.
- Given a theme, return statistics to show the change in theme popularity across time. 
- Given a theme, return statistics to show the change in sentiment across time. 

## Endpoints
### GET /
Returns a status message.
**Response**
A JSON object containing a status message.
```
json
{
    "Status": "OK"
}
```
### GET /topics
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
### GET /topic_popularity/
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
### GET /topic_sentiment/
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
### POST /predict_topic/
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
### GET /get_documents/
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
