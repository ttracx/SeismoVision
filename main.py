import streamlit as st
import numpy as np
import pandas as pd
import segyio
import plotly.graph_objects as go
import io
import os
from model import (
    create_classifier, prepare_data_for_prediction, predict_reservoirs, 
    create_prediction_map, generate_sample_training_data, train_model, 
    save_model, load_saved_model
)

st.set_page_config(
    page_title="Seismic Data Interpreter",
    page_icon="🌍",
    layout="wide"
)

st.title("Seismic Data Interpretation Application")

@st.cache_resource
def load_classifier():
    """Load or create and train the CNN classifier"""
    try:
        if os.path.exists('model.h5'):
            st.info("Loading pre-trained model...")
            model = load_saved_model('model.h5')
            if model is None:
                raise Exception("Failed to load pre-trained model")
        else:
            st.info("Training new model with synthetic data...")
            model = create_classifier()
            if model is None:
                raise Exception("Failed to create model")
                
            # Generate and train with synthetic data
            X, y = generate_sample_training_data()
            if X is None or y is None:
                raise Exception("Failed to generate training data")
                
            if train_model(model, X, y):
                st.success("Model trained successfully!")
                # Save the trained model
                if save_model(model):
                    st.success("Model saved successfully!")
                else:
                    st.warning("Could not save the model, but it can still be used for predictions.")
            else:
                raise Exception("Error occurred during model training")
        return model
    except Exception as e:
        st.error(f"Error in model initialization: {str(e)}")
        return None

# Rest of the main.py file remains unchanged
