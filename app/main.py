from logging import debug
from fastapi import FastAPI
from fastapi.params import Body, Query
import joblib
import uvicorn
import pickle
from text_preprocess import text_preprocess
from remove_stopwords import remove_stopwords
import os
import numpy as np

MODEL_PATH = "E:/hoctap/FastAPI/models"

nb_model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
app = FastAPI()

def preprocess(text):
    text = text_preprocess(text)
    text = remove_stopwords(text)
    return text

@app.post("/classify_text")
async def classify_text(text: str):
    text = preprocess(text)
    label = nb_model.predict([text])
    result = {
        'label': int(label[0])
        }
    return result

