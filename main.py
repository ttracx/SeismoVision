import streamlit as st
import numpy as np
import pandas as pd
import segyio
import plotly.graph_objects as go
import io
import os
from model import (
    create_classifier, prepare_data_for_prediction, predict_reservoirs, 
    create_prediction_map, generate_sample_training_data, train_model, save_model,
    load_model
)
from preprocessing import (
    load_segy_data, preprocess_seismic_data, plot_seismic_section,
    plot_amplitude_spectrum, create_cross_section
)
import joblib

st.set_page_config(
    page_title="SeismoVision",
    page_icon="🌍",
    layout="wide"
)

st.title("SeismoVision")
st.subheader("Seismic Data Interpretation Application")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Processing", "Model Training", "Prediction"])

# Add a file uploader for SEG-Y data
uploaded_file = st.sidebar.file_uploader("Upload SEG-Y file", type=['sgy', 'segy'])

if page == "Data Processing":
    # Preprocessing parameters in sidebar
    st.sidebar.subheader("Preprocessing Parameters")
    apply_agc = st.sidebar.checkbox("Apply AGC", value=True)
    agc_window = st.sidebar.slider("AGC Window", 100, 1000, 500)
    apply_bandpass = st.sidebar.checkbox("Apply Bandpass Filter", value=True)
    lowcut = st.sidebar.slider("Low Cut Frequency (Hz)", 1, 20, 5)
    highcut = st.sidebar.slider("High Cut Frequency (Hz)", 50, 200, 100)
    sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", value=1000)

    # Process and display seismic data
    if uploaded_file is not None:
        # Create two columns for visualization
        col1, col2 = st.columns(2)
        
        with st.spinner("Loading and processing seismic data..."):
            # Save the uploaded file temporarily
            with open("temp.sgy", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load the SEG-Y data
            seismic_data = load_segy_data("temp.sgy")
            
            if seismic_data is not None:
                # Store the processed data in session state
                preprocessing_params = {
                    'apply_agc': apply_agc,
                    'agc_window': agc_window,
                    'apply_bandpass': apply_bandpass,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'sampling_rate': sampling_rate
                }
                
                processed_data = preprocess_seismic_data(seismic_data['data'], preprocessing_params)
                st.session_state['processed_data'] = processed_data
                st.session_state['seismic_data'] = seismic_data
                
                # Plot seismic section
                with col1:
                    st.subheader("Seismic Section")
                    fig = plot_seismic_section(processed_data, seismic_data['time'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Plot amplitude spectrum
                with col2:
                    st.subheader("Amplitude Spectrum")
                    fig = plot_amplitude_spectrum(processed_data, sampling_rate)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cleanup temporary file
                os.remove("temp.sgy")
    else:
        st.info("Please upload a SEG-Y file to begin analysis.")

elif page == "Model Training":
    st.header("Model Training Interface")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please upload and process seismic data first in the Data Processing tab.")
    else:
        st.subheader("Training Options")
        n_samples = st.slider("Number of training samples", 100, 5000, 1000)
        
        if st.button("Generate Training Data"):
            with st.spinner("Generating training data..."):
                X_train, y_train = generate_sample_training_data(
                    st.session_state['processed_data'],
                    n_samples=n_samples
                )
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train
                st.success("Training data generated successfully!")
                
                # Display class distribution
                unique, counts = np.unique(y_train, return_counts=True)
                st.write("Class distribution:")
                st.write(dict(zip(unique, counts)))
        
        if 'X_train' in st.session_state and st.button("Train Model"):
            with st.spinner("Training model..."):
                model, scaler = train_model(
                    st.session_state['X_train'],
                    st.session_state['y_train']
                )
                save_model(model)
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.success("Model trained and saved successfully!")

elif page == "Prediction":
    st.header("Reservoir Prediction")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please upload and process seismic data first in the Data Processing tab.")
    else:
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train a model first.")
        else:
            st.session_state['model'] = model
            if st.button("Run Prediction"):
                with st.spinner("Making predictions..."):
                    predictions, probabilities = predict_reservoirs(
                        st.session_state['model'],
                        st.session_state['processed_data']
                    )
                    
                    prediction_map = create_prediction_map(
                        predictions,
                        st.session_state['processed_data'].shape
                    )
                    
                    # Plot original data and predictions side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Seismic Section")
                        fig = plot_seismic_section(
                            st.session_state['processed_data'],
                            st.session_state['seismic_data']['time']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Reservoir Prediction")
                        fig = go.Figure(data=go.Heatmap(
                            z=prediction_map,
                            colorscale='Viridis',
                            showscale=True
                        ))
                        fig.update_layout(
                            title="Reservoir Prediction Map",
                            yaxis_title="Time/Depth",
                            xaxis_title="Trace Number",
                            yaxis_autorange='reversed'
                        )
                        st.plotly_chart(fig, use_container_width=True)
