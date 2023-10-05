import pickle
import numpy as np
from flask import Flask, request, render_template, jsonify
import xgboost as xgb

app = Flask(__name__)

# Load the model and TF-IDF vectorizer
model = pickle.load(open("model.pkl", 'rb'))
tfidf_vectorizer = pickle.load(open("vect.pkl", 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input tweet text from the request
        tweet = request.form.get('tweet')

        # Debugging: Print the received tweet
        print("Received Tweet:", tweet)

        # Preprocess the input text using the TF-IDF vectorizer
        processed_tweet = tfidf_vectorizer.transform([tweet])

        # Make prediction
        dtest = xgb.DMatrix(processed_tweet)
        prediction = model.predict(dtest)
        print(prediction)

        # Convert the prediction to a human-readable label
        result = "Disaster Tweet." if prediction[0] > 0.4 else "Non-Disaster Tweet."

        # Debugging: Print the prediction
        print("Prediction:", result)

        return result  # Return prediction as JSON
    except Exception as e:
        return jsonify({"error": str(e)})  # Return error as JSON

if __name__ == "__main__":
    app.run(debug=True, port=5002)
