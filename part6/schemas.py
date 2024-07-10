from pydantic import BaseModel

class PredictIn(BaseModel):
    sepal_length : float 
    sepal_width : float
    petal_length : float
    petal_width : float

class PredictOut(BaseModel):
    iris_class : int

class TestIn(BaseModel):
    test_val : float

class TestOut(BaseModel):
    test_result:float

