import pandas as pd
import pymysql
from sklearn.datasets import load_iris
import time

def get_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = pd.concat([X,y],axis="columns")
    rename_rule = {
    "sepal length (cm)" : "sepal_length",
    "sepal width (cm)"  : "sepal_width",
    "petal length (cm)" : "petal_length",
    "petal width (cm)"  : "petal_width",
    }
    df = df.rename(columns = rename_rule)
    return df


def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            {data.sepal_length},
            {data.sepal_width},
            {data.petal_length},
            {data.petal_width},
            {data.target}
        )
    """
    print(insert_row_query)
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()

def generate_data(db_connect,df):
    while True : 
        insert_data(db_connect, df.sample(1).squeeze())
        time.sleep(1)

if __name__ =="__main__":
    db_config = { 
    'user' :"chanseam" ,
    'password' : 'qwe123!@#',
    'host' : "localhost",
    'port' : 4000,
    'database': 'test'
    }
    db_connect = pymysql.connect(**db_config)
    df = get_data()
    generate_data(db_connect, df)

