import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

def create_classifier():
    """Create a random forest classifier for reservoir identification"""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

def extract_features(seismic_data, window_size=5):
    """Extract features from seismic data using a sliding window"""
    n_samples, n_traces = seismic_data.shape
    features = []
    
    for i in range(window_size, n_samples - window_size):
        window = seismic_data[i-window_size:i+window_size+1, :]
        features.append([
            np.mean(window),
            np.std(window),
            np.max(window),
            np.min(window),
            np.percentile(window, 25),
            np.percentile(window, 75),
            np.sum(np.abs(np.diff(window, axis=0))),  # Total variation
            np.mean(np.abs(np.fft.fft(window, axis=0)))  # Average frequency content
        ])
    
    return np.array(features)

def prepare_data_for_prediction(seismic_data):
    """Prepare seismic data for model prediction"""
    features = extract_features(seismic_data)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def predict_reservoirs(model, seismic_data):
    """Make predictions using the trained model"""
    features = prepare_data_for_prediction(seismic_data)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return predictions, probabilities

def create_prediction_map(predictions, original_shape, window_size=5):
    """Create a prediction map matching the original seismic data shape"""
    prediction_map = np.zeros(original_shape)
    prediction_map[window_size:-window_size, :] = predictions.reshape(-1, original_shape[1])
    return prediction_map

def generate_sample_training_data(seismic_data, n_samples=1000):
    """Generate synthetic training data for demonstration"""
    features = extract_features(seismic_data)
    
    # Randomly select samples
    indices = np.random.choice(len(features), n_samples, replace=False)
    X_train = features[indices]
    
    # Generate synthetic labels (0: non-reservoir, 1: reservoir)
    # Using simple amplitude-based thresholding for demonstration
    y_train = (np.mean(X_train, axis=1) > np.mean(X_train)) * 1
    
    return X_train, y_train

def train_model(X_train, y_train):
    """Train the reservoir identification model"""
    model = create_classifier()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model.fit(X_scaled, y_train)
    return model, scaler

def save_model(model, filepath='model.joblib'):
    """Save the trained model to disk"""
    joblib.dump(model, filepath)
    return filepath

def load_model(filepath='model.joblib'):
    """Load a trained model from disk"""
    try:
        return joblib.load(filepath)
    except:
        return None
