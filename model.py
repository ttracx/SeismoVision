from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

def create_classifier(n_estimators=100):
    """Create a Random Forest classifier for seismic reservoir identification"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    return model

def generate_sample_training_data(n_samples=1000, window_size=64):
    """Generate synthetic training data for initial model training"""
    # Generate synthetic seismic windows
    X = np.random.randn(n_samples, window_size * window_size)
    
    # Generate synthetic labels (0: non-reservoir, 1: reservoir)
    # Add some patterns to make it more realistic
    y = np.zeros(n_samples)
    
    # Add some synthetic patterns for reservoir samples
    for i in range(n_samples):
        if np.mean(X[i]) > 0.5 and np.std(X[i]) > 1.0:
            y[i] = 1
    
    return X, y

def train_model(model, X, y):
    """Train the Random Forest classifier"""
    try:
        model.fit(X, y)
        return True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def save_model(model, filepath='model.joblib'):
    """Save the trained model"""
    try:
        joblib.dump(model, filepath)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def prepare_data_for_prediction(seismic_data, window_size=64):
    """Prepare seismic data for prediction by creating sliding windows"""
    n_traces, n_samples = seismic_data.shape
    windows = []
    positions = []
    
    for i in range(0, n_traces - window_size + 1, window_size // 2):
        for j in range(0, n_samples - window_size + 1, window_size // 2):
            window = seismic_data[i:i + window_size, j:j + window_size]
            if window.shape == (window_size, window_size):
                windows.append(window.flatten())
                positions.append((i, j))
    
    return np.array(windows), positions

def predict_reservoirs(model, windows):
    """Make predictions on the prepared windows"""
    try:
        # Check if model is fitted
        from sklearn.exceptions import NotFittedError
        try:
            predictions = model.predict_proba(windows)
            return predictions
        except NotFittedError:
            print("Model not fitted. Training required.")
            return None
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def create_prediction_map(predictions, positions, original_shape, window_size=64):
    """Create a prediction map from the model outputs"""
    if predictions is None:
        return None
        
    prediction_map = np.zeros(original_shape)
    count_map = np.zeros(original_shape)
    
    for pred, (i, j) in zip(predictions, positions):
        reservoir_prob = pred[1]  # Assuming binary classification
        prediction_map[i:i + window_size, j:j + window_size] += reservoir_prob
        count_map[i:i + window_size, j:j + window_size] += 1
    
    count_map[count_map == 0] = 1  # Avoid division by zero
    prediction_map = prediction_map / count_map
    
    return prediction_map
