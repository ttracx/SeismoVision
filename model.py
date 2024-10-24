import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_classifier():
    """Create a new classifier model"""
    return RandomForestClassifier(n_estimators=100, random_state=42)

def generate_sample_training_data(n_samples=1000):
    """Generate synthetic training data for testing"""
    # Generate synthetic seismic traces
    X = np.random.randn(n_samples, 100)  # 100 features per sample
    # Generate synthetic labels (0: no reservoir, 1: reservoir)
    y = np.random.randint(0, 2, n_samples)
    return X, y

def train_model(model, X, y):
    """Train the model with given data"""
    try:
        model.fit(X, y)
        return True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def save_model(model, filename='model.joblib'):
    """Save the trained model"""
    try:
        joblib.dump(model, filename)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def prepare_data_for_prediction(seismic_data, window_size=100):
    """Prepare seismic data for prediction"""
    n_traces, n_samples = seismic_data.shape
    windows = []
    positions = []
    
    # Slide window over each trace
    for i in range(n_traces):
        for j in range(0, n_samples - window_size + 1, window_size):
            window = seismic_data[i, j:j+window_size]
            windows.append(window)
            positions.append((i, j))
    
    return np.array(windows), positions

def predict_reservoirs(model, windows):
    """Make predictions using the trained model"""
    try:
        predictions = model.predict_proba(windows)
        return predictions[:, 1]  # Return probability of reservoir class
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def create_prediction_map(predictions, positions, data_shape):
    """Create a 2D map of predictions"""
    try:
        prediction_map = np.zeros(data_shape)
        for pred, (i, j) in zip(predictions, positions):
            prediction_map[i, j] = pred
        return prediction_map
    except Exception as e:
        print(f"Error creating prediction map: {str(e)}")
        return None
