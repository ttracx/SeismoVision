import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
import os

def create_classifier():
    """Create a CNN model for seismic data interpretation"""
    try:
        # Create CNN model
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(100, 1)),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error creating CNN model: {str(e)}")
        return None

def generate_sample_training_data(n_samples=1000):
    """Generate synthetic training data for testing"""
    try:
        # Generate synthetic seismic traces
        X = np.random.randn(n_samples, 100)  # 100 features per sample
        # Reshape for CNN input (samples, timesteps, features)
        X = X.reshape(n_samples, 100, 1)
        # Generate synthetic labels (0: no reservoir, 1: reservoir)
        y = np.random.randint(0, 2, n_samples)
        return X, y
    except Exception as e:
        print(f"Error generating training data: {str(e)}")
        return None, None

def train_model(model, X, y):
    """Train the model with given data"""
    try:
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
        return True
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

def save_model(model, filename='model.h5'):
    """Save the trained model"""
    try:
        model.save(filename)
        return True
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return False

def load_saved_model(filename='model.h5'):
    """Load a saved model"""
    try:
        if os.path.exists(filename):
            return load_model(filename)
        else:
            print(f"Model file {filename} not found")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def prepare_data_for_prediction(seismic_data, window_size=100):
    """Prepare seismic data for CNN prediction"""
    try:
        n_traces, n_samples = seismic_data.shape
        windows = []
        positions = []
        
        # Slide window over each trace
        for i in range(n_traces):
            for j in range(0, n_samples - window_size + 1, window_size):
                window = seismic_data[i, j:j+window_size]
                windows.append(window)
                positions.append((i, j))
        
        # Reshape for CNN input (samples, timesteps, features)
        windows = np.array(windows).reshape(-1, window_size, 1)
        return windows, positions
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        return None, None

def predict_reservoirs(model, windows):
    """Make predictions using the trained CNN model"""
    try:
        predictions = model.predict(windows)
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
