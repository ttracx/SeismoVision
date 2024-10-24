import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import signal
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

class SeismicPreprocessor:
    def __init__(self):
        self.scaler = None
        self.pca = None
    
    def bandpass_filter(self, data, lowcut=5, highcut=125, fs=250, order=5):
        """Apply bandpass filter to reduce noise"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        if len(data.shape) == 3:
            filtered_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered_data[i] = signal.filtfilt(b, a, data[i], axis=0)
            return filtered_data
        return signal.filtfilt(b, a, data, axis=0)
    
    def normalize_data(self, data, method='standard', reshape=True):
        """Normalize data using different methods"""
        if reshape and len(data.shape) > 2:
            original_shape = data.shape
            data = data.reshape(data.shape[0], -1)
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Unsupported normalization method")
            
        normalized_data = self.scaler.fit_transform(data)
        
        if reshape and len(original_shape) > 2:
            normalized_data = normalized_data.reshape(original_shape)
        
        return normalized_data
    
    def reduce_dimensions(self, data, n_components=0.95):
        """Reduce dimensions using PCA"""
        original_shape = data.shape
        flattened_data = data.reshape(data.shape[0], -1)
        
        self.pca = PCA(n_components=n_components)
        reduced_data = self.pca.fit_transform(flattened_data)
        
        return reduced_data
    
    def denoise(self, data, sigma=1):
        """Denoise data using Gaussian filter"""
        if len(data.shape) == 3:
            denoised_data = np.zeros_like(data)
            for i in range(data.shape[0]):
                denoised_data[i] = gaussian_filter(data[i], sigma=sigma)
            return denoised_data
        return gaussian_filter(data, sigma=sigma)
    
    def augment_data(self, data, labels=None):
        """Augment data with noise and time shifts"""
        augmented_data = []
        augmented_labels = []
        
        # Original data
        augmented_data.append(data)
        if labels is not None:
            augmented_labels.append(labels)
        
        # Add random noise
        noisy_data = data + np.random.normal(0, 0.05, data.shape)
        augmented_data.append(noisy_data)
        if labels is not None:
            augmented_labels.append(labels)
        
        # Time shift
        shifted_data = np.roll(data, shift=5, axis=1)
        augmented_data.append(shifted_data)
        if labels is not None:
            augmented_labels.append(labels)
        
        return (np.concatenate(augmented_data, axis=0), 
                np.concatenate(augmented_labels, axis=0) if labels is not None else None)

def preprocess_data(data, options=None):
    """
    Preprocess seismic data with specified options
    """
    if options is None:
        options = {
            'normalize': 'standard',
            'denoise': False,
            'bandpass': False,
            'dimension_reduction': False,
            'augment': False
        }
    
    processor = SeismicPreprocessor()
    processed_data = data.astype(np.float32)
    
    # Apply preprocessing steps based on options
    if options.get('bandpass', False):
        processed_data = processor.bandpass_filter(processed_data)
    
    if options.get('denoise', False):
        processed_data = processor.denoise(processed_data)
    
    if options.get('normalize'):
        processed_data = processor.normalize_data(processed_data, method=options['normalize'])
    
    if options.get('dimension_reduction', False):
        processed_data = processor.reduce_dimensions(processed_data)
    
    if options.get('augment', False):
        processed_data, _ = processor.augment_data(processed_data)
    
    return processed_data
