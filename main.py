from fastapi import FastAPI
from pydantic import BaseModel

from backend import predict

app = FastAPI()


class Input(BaseModel):
    passage: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/generate")
async def generate(input_: Input):
    text = input_.passage
    prediction = predict(text)
    return {"prob": prediction}
