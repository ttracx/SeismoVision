import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model import SeismicCNN
from preprocessing import preprocess_data
from utils import plot_seismic_data, plot_training_history
import io
import pandas as pd

st.set_page_config(
    page_title="Seismic Data Interpreter",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Seismic Data Interpretation System")
    st.sidebar.title("Controls")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = SeismicCNN()
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None

    # Sidebar options
    action = st.sidebar.selectbox(
        "Choose Action",
        ["Upload & Visualize", "Train Model", "Make Predictions"]
    )

    if action == "Upload & Visualize":
        upload_and_visualize()
    elif action == "Train Model":
        train_model()
    else:
        make_predictions()

def upload_and_visualize():
    st.header("Data Upload & Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_file = st.file_uploader("Upload Seismic Data (.npy)", type=['npy'])
        if data_file is not None:
            try:
                seismic_data = np.load(data_file)
                st.session_state['seismic_data'] = seismic_data
                
                st.success("Data loaded successfully!")
                st.write("Data shape:", seismic_data.shape)
                
                fig = plot_seismic_data(seismic_data)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    with col2:
        labels_file = st.file_uploader("Upload Labels (optional, .npy)", type=['npy'])
        if labels_file is not None:
            try:
                labels = np.load(labels_file)
                st.session_state['labels'] = labels
                st.success("Labels loaded successfully!")
                st.write("Labels shape:", labels.shape)
            except Exception as e:
                st.error(f"Error loading labels: {str(e)}")

def train_model():
    st.header("Model Training")
    
    if 'seismic_data' not in st.session_state or 'labels' not in st.session_state:
        st.warning("Please upload both seismic data and labels first!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Number of epochs", 5, 50, 10)
        batch_size = st.selectbox("Batch size", [16, 32, 64, 128])
        
    with col2:
        validation_split = st.slider("Validation split", 0.1, 0.3, 0.2)
        
    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            X = st.session_state['seismic_data']
            y = st.session_state['labels']
            
            # Preprocess data
            X_processed = preprocess_data(X)
            
            # Train model
            history = st.session_state.model.train(
                X_processed, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
            
            st.session_state.training_history = history
            
            # Plot training history
            fig = plot_training_history(history)
            st.pyplot(fig)
            
            st.success("Training completed!")

def make_predictions():
    st.header("Make Predictions")
    
    if not hasattr(st.session_state.model, 'model'):
        st.warning("Please train the model first!")
        return
        
    pred_data = st.file_uploader("Upload data for prediction (.npy)", type=['npy'])
    
    if pred_data is not None:
        try:
            X_pred = np.load(pred_data)
            X_pred_processed = preprocess_data(X_pred)
            
            predictions = st.session_state.model.predict(X_pred_processed)
            confidence_scores = np.max(predictions, axis=1)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Prediction': np.argmax(predictions, axis=1),
                'Confidence': confidence_scores
            })
            
            st.write("Prediction Results:")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

if __name__ == "__main__":
    main()
