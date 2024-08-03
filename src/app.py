from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data from the request
    input_features = np.array([data['features']])  # Extract features from the data

    # Standardize the input features
    input_features = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features)

    # Return the result as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
