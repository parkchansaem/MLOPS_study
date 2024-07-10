FROM amd64/python:3.9-slim

WORKDIR /usr/app

RUN pip install -U pip &&\
    pip install pymysql

COPY create_table.py create_table.py

ENTRYPOINT [ "python","create_table.py"]