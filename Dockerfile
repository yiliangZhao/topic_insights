FROM python:3.9-bookworm as Base
RUN pip install --upgrade pip

COPY ./app.py /server/app.py
COPY ./utils.py /server/utils.py
COPY ./data /server/data
COPY ./models /server/models
COPY ./requirements.txt /server/requirements.txt
COPY ./mydb.sqlite /server/mydb.sqlite

WORKDIR /server

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:8080 app:app