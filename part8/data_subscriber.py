from json import loads

import pymysql
import requests
from kafka import KafkaConsumer

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris_prediction(
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        iris_class int
        );
    """
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

def insert_data(db_connect,data):
    insert_row_query= f"""
    INSERT INTO iris_prediction
    (timestamp, iris_class)
    VALUES (
        '{data["timestamp"]}',
        {data["iris_class"]}
    );
    """
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()

def subscribe_data(db_connect, consumer):
    for msg in consumer:
        print(
            f"Topic : {msg.topic}\n"
            f"partition : {msg.partition}\n"
            f"Offset : {msg.offset}\n"
            f"Key : {msg.key}\n"
            f"Value : {msg.value}\n",
        )
        msg.value["payload"].pop("id")
        msg.value["payload"].pop("target")
        ts = msg.value["payload"].pop("timestamp")

        response = requests.post(
            url= "http://api-with-model:8000/predict",
            json=msg.value["payload"],
            headers = {"Content-Type": "application/json"}
        ).json()
        response["timestamp"] = ts
        print(ts)
        insert_data(db_connect,response)
    

if __name__ == "__main__":
    db_connect = pymysql.connect(
        user = "targetuser",
        password = "targetpassword",
        host = "target-mysql-server",
        port=3306,
        database = "targetdatabase",
    )
    create_table(db_connect)

    consumer = KafkaConsumer(
        "mysql-source-iris_data",
        bootstrap_servers="broker:29092",
        auto_offset_reset = "earliest",
        group_id = "iris-data-consumer-group",
        value_deserializer = lambda x: loads(x),
    )
    subscribe_data(db_connect,consumer)



