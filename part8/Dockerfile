FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install pymysql kafka-python requests

COPY data_subscriber.py data_subscriber.py

ENTRYPOINT [ "python","data_subscriber.py" ]