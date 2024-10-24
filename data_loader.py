import numpy as np
import pandas as pd
import segyio
import io

def load_segy(file):
    """Load SEG-Y format seismic data"""
    # Create a temporary file to save the uploaded content
    with open('temp.sgy', 'wb') as f:
        f.write(file.read())
    
    # Read the SEG-Y file
    with segyio.open('temp.sgy', 'r', ignore_geometry=True) as segy:
        # Get number of traces and samples
        n_traces = segy.tracecount
        n_samples = len(segy.samples)
        
        # Read all traces into a numpy array
        data = np.zeros((n_traces, n_samples))
        for i in range(n_traces):
            data[i] = segy.trace[i]
    
    return data

def load_csv(file):
    """Load CSV format seismic data"""
    # Read CSV file
    df = pd.read_csv(file)
    return df.values

def load_npy(file):
    """Load NumPy format seismic data"""
    return np.load(file)

def detect_format(file):
    """Detect file format based on extension"""
    filename = file.name.lower()
    if filename.endswith('.sgy') or filename.endswith('.segy'):
        return 'segy'
    elif filename.endswith('.csv'):
        return 'csv'
    elif filename.endswith('.npy'):
        return 'npy'
    else:
        raise ValueError("Unsupported file format. Please use .sgy, .segy, .csv, or .npy files.")

def load_seismic_data(file):
    """Load seismic data from various formats"""
    format_type = detect_format(file)
    
    try:
        if format_type == 'segy':
            data = load_segy(file)
        elif format_type == 'csv':
            data = load_csv(file)
        else:  # npy format
            data = load_npy(file)
            
        return data
    except Exception as e:
        raise Exception(f"Error loading {format_type} file: {str(e)}")
