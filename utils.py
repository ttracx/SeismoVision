import matplotlib.pyplot as plt
import numpy as np

def plot_seismic_data(data, sample_idx=0):
    """Plot a sample of seismic data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(data.shape) == 4:
        img = data[sample_idx, :, :, 0]
    else:
        img = data[sample_idx]
    
    im = ax.imshow(img, cmap='seismic', aspect='auto')
    plt.colorbar(im)
    ax.set_title("Seismic Data Visualization")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time")
    
    plt.tight_layout()
    return fig

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history['accuracy'], label='Training')
    ax1.plot(history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history['loss'], label='Training')
    ax2.plot(history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    return fig
