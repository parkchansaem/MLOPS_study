from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app =FastAPI()

USER_DB = {}

NAME_NOT_FOUND = HTTPException(status_code=400, detail="Name not found")

class CreateIn(BaseModel):
    name: str
    nickname :str

class Createout(BaseModel):
    status:str
    id:str

@app.post("/users",response_model=Createout)
def create_user(user:CreateIn)->Createout:
    USER_DB[user.name]=user.nickname
    user_dict = user.dict()
    user_dict["status"] = "success"
    user_dict["id"] = len(USER_DB)
    return user_dict

@app.get("/users")
def read_user(name:str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    return {"nickname":USER_DB[name]}

@app.put("/users")
def update_user(name:str,nickname:str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    USER_DB[name] = nickname
    return {"status":"success"}

@app.delete("/users")
def delete_user(name:str):
    if name not in USER_DB:
        raise NAME_NOT_FOUND
    del USER_DB[name]
    return {"status":"success"}
    

