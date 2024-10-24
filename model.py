from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

def prepare_data_for_prediction(seismic_data, window_size=64):
    """Prepare seismic data for prediction by creating sliding windows"""
    n_traces, n_samples = seismic_data.shape
    windows = []
    positions = []
    
    for i in range(0, n_traces - window_size + 1, window_size // 2):
        for j in range(0, n_samples - window_size + 1, window_size // 2):
            window = seismic_data[i:i + window_size, j:j + window_size]
            if window.shape == (window_size, window_size):
                # Flatten the window for sklearn
                windows.append(window.flatten())
                positions.append((i, j))
    
    return np.array(windows), positions

def predict_reservoirs(model, windows):
    """Make predictions on the prepared windows"""
    # Get probability predictions
    predictions = model.predict_proba(windows)
    return predictions

def create_prediction_map(predictions, positions, original_shape, window_size=64):
    """Create a prediction map from the model outputs"""
    prediction_map = np.zeros(original_shape)
    count_map = np.zeros(original_shape)
    
    for pred, (i, j) in zip(predictions, positions):
        # Use the reservoir probability (second class)
        reservoir_prob = pred[1]  # Assuming binary classification
        prediction_map[i:i + window_size, j:j + window_size] += reservoir_prob
        count_map[i:i + window_size, j:j + window_size] += 1
    
    # Average overlapping predictions
    count_map[count_map == 0] = 1  # Avoid division by zero
    prediction_map = prediction_map / count_map
    
    return prediction_map
