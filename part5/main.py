from fastapi import FastAPI

#create a FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    a = 10
    b = 20 
    return {"Hello":"World"}, a+b

