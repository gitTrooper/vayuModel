import tensorflow as tf
import pickle
import numpy as np

# Load the model
model = tf.keras.models.load_model("aqiModel.keras")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def preprocess(data):
    """
    Preprocesses input data by scaling.
    Input should be a list or array with features [CO, NO2, O3, SO2, PM2.5, PM10, NH3].
    """
    data = np.array(data).reshape(1, -1)
    return scaler.transform(data)

def predict_aqi(features):
    """
    Takes scaled input features and returns the predicted AQI.
    """
    scaled_data = preprocess(features)
    prediction = model.predict(scaled_data)
    return prediction[0][0]  # Return the predicted AQI
