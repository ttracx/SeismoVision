import numpy as np
import segyio
import plotly.graph_objects as go
from scipy import signal
import streamlit as st

def load_segy_data(file):
    """Load SEG-Y file and return trace data and header information"""
    try:
        with segyio.open(file, 'r', strict=False) as segy:
            # Get basic attributes
            n_traces = segy.tracecount
            n_samples = len(segy.samples)
            
            # Read all traces
            traces = np.zeros((n_traces, n_samples))
            for i in range(n_traces):
                traces[i] = segy.trace[i]
            
            # Get time/depth axis
            t = segy.samples
            
            return {
                'data': traces,
                'time': t,
                'n_traces': n_traces,
                'n_samples': n_samples
            }
    except Exception as e:
        st.error(f"Error loading SEG-Y file: {str(e)}")
        return None

def preprocess_seismic_data(data, params=None):
    """Apply preprocessing steps to seismic data"""
    if params is None:
        params = {
            'apply_agc': True,
            'agc_window': 500,
            'apply_bandpass': True,
            'lowcut': 5,
            'highcut': 100,
            'sampling_rate': 1000
        }
    
    processed_data = data.copy()
    
    if params['apply_bandpass']:
        nyquist = params['sampling_rate'] / 2
        low = params['lowcut'] / nyquist
        high = params['highcut'] / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        processed_data = signal.filtfilt(b, a, processed_data, axis=1)
    
    if params['apply_agc']:
        window_length = params['agc_window']
        for i in range(processed_data.shape[0]):
            trace = processed_data[i]
            window = np.ones(window_length) / window_length
            rms = np.sqrt(np.convolve(trace**2, window, 'same'))
            processed_data[i] = np.divide(trace, rms, where=rms!=0)
    
    return processed_data

def plot_seismic_section(data, time, title="Seismic Section"):
    """Create an interactive seismic section plot"""
    fig = go.Figure()
    
    # Add seismic data as a heatmap
    fig.add_trace(go.Heatmap(
        z=data,
        y=time,
        colorscale='Seismic',
        zmid=0,
        showscale=True
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title='Time/Depth',
        xaxis_title='Trace Number',
        yaxis_autorange='reversed'
    )
    
    return fig

def plot_amplitude_spectrum(data, sampling_rate):
    """Create amplitude spectrum plot"""
    # Calculate frequency spectrum
    freq = np.fft.fftfreq(data.shape[1], d=1/sampling_rate)
    spectrum = np.abs(np.fft.fft(data, axis=1))
    avg_spectrum = np.mean(spectrum, axis=0)
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freq[:freq.size//2],
        y=avg_spectrum[:freq.size//2],
        mode='lines',
        name='Amplitude Spectrum'
    ))
    
    fig.update_layout(
        title="Amplitude Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        showlegend=True
    )
    
    return fig

def create_cross_section(data, direction='inline', position=None):
    """Create cross-section view of 3D seismic data"""
    if direction == 'inline':
        section = data[position, :, :] if position else data[data.shape[0]//2, :, :]
    elif direction == 'xline':
        section = data[:, position, :] if position else data[:, data.shape[1]//2, :]
    else:
        section = data[:, :, position] if position else data[:, :, data.shape[2]//2]
    
    fig = go.Figure(data=go.Heatmap(
        z=section,
        colorscale='Seismic',
        zmid=0
    ))
    
    fig.update_layout(
        title=f"{direction.capitalize()} Cross-section",
        yaxis_title='Time/Depth',
        xaxis_title='Distance',
        yaxis_autorange='reversed'
    )
    
    return fig
