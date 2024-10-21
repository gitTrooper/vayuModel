from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib  # For loading the scaler

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('aqiModel.keras')  # Adjust path if necessary

# Load the StandardScaler
scaler = joblib.load('scaler.pkl')  # Make sure this file is in your project directory

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Extract input features from the request data
    input_data = [
        data['CO'],
        data['NO2'],
        data['O3'],
        data['SO2'],
        data['PM2_5'],
        data['PM10'],
        data['NH3']
    ]

    # Convert input data to a numpy array
    input_array = np.array(input_data).reshape(1, -1)  # Reshape for a single prediction

    # Standardize the input data using the loaded scaler
    standardized_input = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(standardized_input)
    predicted_aqi = prediction[0][0]  # Get the predicted AQI value

    # Return the result as a JSON response
    return jsonify({'AQI': predicted_aqi})

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
