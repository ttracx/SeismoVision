import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess seismic data for model input
    """
    # Ensure data is float32
    data = data.astype(np.float32)
    
    # Reshape if needed
    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    
    # Normalize data
    data_shape = data.shape
    data_flat = data.reshape(-1, data_shape[-1])
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_flat)
    data = data_normalized.reshape(data_shape)
    
    return data
