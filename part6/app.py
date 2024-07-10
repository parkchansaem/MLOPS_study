import mlflow
import pandas as pd
from fastapi import FastAPI,Depends
from schemas import PredictIn, PredictOut,TestIn,TestOut

def get_model():
    model = mlflow.sklearn.load_model(model_uri="./sk-model")
    return model
MODEL = get_model()

app =FastAPI()


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])
    pred = MODEL.predict(df).item()
    return PredictOut(iris_class = pred)

# @app.get("/predict",response_model=TestOut)
# def test(value:TestIn = Depends()) -> TestOut:
#     test_value = value.test_val + 1
#     return TestOut(test_result = test_value)

# @app.get("/predict/{num}")
# def test_2(num : float):
#     return {"test": num}
