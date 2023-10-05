"""
Created on Mon Aug 15 13:12:11 2022


"""

import numpy as np
from fastapi import FastAPI, Form
import pandas as pd
from starlette.responses import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import re

app = FastAPI()

# Define the HTML form with improved styling
def input_form():
    return '''
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                text-align: center;
            }
            .container {
                background-color: #fff;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 80%;
                max-width: 400px;
                margin: 0 auto;
                margin-top: 50px;
            }
            .input-field {
                width: 100%;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 16px;
            }
            .submit-button {
                background-color: #007bff;
                color: #fff;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                cursor: pointer;
                font-size: 18px;
            }
            .submit-button:hover {
                background-color: #0056b3;
            }
            .result-container {
                margin-top: 20px;
            }
            .result {
                font-size: 20px;
                font-weight: bold;
                color: #007bff;
            }
            .progress-bar-container {
                margin-top: 20px;
                width: 100%;
                height: 20px;
                background-color: #ccc;
                border-radius: 4px;
                overflow: hidden;
            }
            .progress-bar {
                width: {probability}%;
                height: 100%;
                background-color: {color};
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DisasterAlertAI</h1>
            <p>Check if a tweet is related to a disaster.</p>
            <form method="post">
                <input class="input-field" maxlength="28" name="text" type="text" placeholder="Paste the tweet content here" />
                <br />
                <button class="submit-button" type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    '''

@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return input_form()

        
data = pd.read_csv('Sentiment.csv')
tokenizer = Tokenizer(num_words=2000, split=' ')
tokenizer.fit_on_texts(data['text'].values)

def preProcess_data(text):
    text = text.lower()
    new_text = re.sub('[^a-zA-z0-9\s]','',text)
    new_text = re.sub('rt', '', new_text)
    return new_text

def my_pipeline(text):
    text_new = preProcess_data(text)
    X = tokenizer.texts_to_sequences(pd.Series(text_new).values)
    X = pad_sequences(X, maxlen=28)
    return X

def get_prediction_text(sentiment):
    if sentiment == 0:
        return 'This tweet is not related to a disaster.'
    elif sentiment == 1:
        return 'This tweet is related to a disaster.'
    else:
        return 'Prediction not available.'

def get_color(probability):
    if probability > 0.7:
        return 'green'
    elif probability > 0.4:
        return 'yellow'
    else:
        return 'red'
    
def color_bar(probability):
    # Calculate the hue based on probability (0 to 1)
    hue = (1 - probability) * 120
    # Create a style string for the progress bar
    style = f"background: linear-gradient(90deg, hsl({hue}, 100%, 50%) {probability*100}%, transparent {probability*100}%);"
    return style
    
@app.get('/predict', response_class=HTMLResponse)
def take_inp():
    return input_form()

@app.post('/predict')
def predict(text: str = Form(...)):
    clean_text = my_pipeline(text)  # Clean and preprocess the text through the pipeline
    loaded_model = tf.keras.models.load_model('distilbert_classifier_model.keras')  # Load the saved model
    predictions = loaded_model.predict(clean_text)  # Predict the text
    sentiment = int(np.argmax(predictions)) #calculate the index of max sentiment
    probability = max(predictions.tolist()[0]) #calulate the probability
    prediction_text = get_prediction_text(sentiment)
    color = get_color(probability)
    
    return {
    "ACTUAL SENTENCE": text,
    "PREDICTED SENTIMENT": prediction_text,
    "PROBABILITY": probability,
    "COLOR": color
}


@app.get('/')
def basic_view():
    return input_form()


