import pymysql

def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        sepal_length float,
        sepal_width float,
        petal_length float,
        petal_width float,
        target int
    )
    """
    print(create_table_query)
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()
    
if __name__ == "__main__":
    db_config = { 
    'user' :"chanseam" ,
    'password' : 'qwe123!@#',
    'host' : "localhost",
    'port' : 4000,
    'database': 'test'
    }
    db_connect = pymysql.connect(**db_config)
    create_table(db_connect)