import streamlit as st
import numpy as np
import pandas as pd
import segyio
import plotly.graph_objects as go
import io
import os
from model import create_classifier, prepare_data_for_prediction, predict_reservoirs, create_prediction_map
import joblib

st.set_page_config(
    page_title="Seismic Data Interpreter",
    page_icon="🌍",
    layout="wide"
)

st.title("Seismic Data Interpretation Application")

@st.cache_resource
def load_classifier():
    """Load or create the Random Forest classifier"""
    if os.path.exists('model.joblib'):
        model = joblib.load('model.joblib')
    else:
        model = create_classifier()
        # Note: In production, you would train the model here or load pre-trained weights
    return model

def load_segy_data(uploaded_file):
    """Load SEG-Y data and return the data array"""
    try:
        # Save the uploaded file temporarily
        with open('temp.sgy', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Open the SEG-Y file
        with segyio.open('temp.sgy', 'r', strict=False) as segy:
            # Get dimensions
            n_traces = segy.tracecount
            n_samples = len(segy.samples)
            
            # Read all traces into a numpy array
            data = np.zeros((n_traces, n_samples))
            for i in range(n_traces):
                data[i] = segy.trace[i]
            
            return data, segy.samples
            
    except Exception as e:
        st.error(f"Error loading SEG-Y file: {str(e)}")
        return None, None
    finally:
        # Clean up temporary file
        if os.path.exists('temp.sgy'):
            os.remove('temp.sgy')

def plot_seismic_section(data, samples, colormap='RdBu', contrast=1.0):
    """Create an interactive plot of the seismic section"""
    # Scale the data
    scaled_data = data * contrast
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=scaled_data,
        y=samples,
        colorscale=colormap,
        zmid=0,  # Center the colorscale at zero
    ))
    
    # Update layout
    fig.update_layout(
        title="Seismic Section",
        yaxis_title="Time/Depth",
        xaxis_title="Trace Number",
        height=700,
    )
    
    return fig

def plot_prediction_overlay(data, prediction_map, samples, colormap='RdBu', contrast=1.0):
    """Create a plot with prediction overlay"""
    # Create figure with secondary axis
    fig = go.Figure()
    
    # Add seismic data
    fig.add_trace(go.Heatmap(
        z=data * contrast,
        y=samples,
        colorscale=colormap,
        zmid=0,
        name="Seismic Data"
    ))
    
    # Add prediction overlay
    fig.add_trace(go.Heatmap(
        z=prediction_map,
        y=samples,
        colorscale="Viridis",
        opacity=0.3,
        name="Reservoir Prediction"
    ))
    
    fig.update_layout(
        title="Seismic Section with Reservoir Predictions",
        yaxis_title="Time/Depth",
        xaxis_title="Trace Number",
        height=700,
    )
    
    return fig

def main():
    # Load classifier
    model = load_classifier()
    
    # Sidebar controls
    st.sidebar.header("Visualization Controls")
    
    # File upload section
    st.markdown("### Upload Seismic Data")
    uploaded_file = st.file_uploader("Choose a SEG-Y file", type=['sgy', 'segy'])
    
    if uploaded_file is not None:
        with st.spinner('Loading seismic data...'):
            # Load the data
            data, samples = load_segy_data(uploaded_file)
            
            if data is not None:
                st.success("File loaded successfully!")
                
                # Store in session state
                st.session_state['seismic_data'] = data
                st.session_state['samples'] = samples
                st.session_state['filename'] = uploaded_file.name
                
                # Display data information
                st.markdown("### Seismic Data Information")
                info = {
                    "Number of Traces": data.shape[0],
                    "Samples per Trace": data.shape[1],
                    "Data Range": f"{data.min():.2f} to {data.max():.2f}",
                    "Data Type": data.dtype
                }
                for key, value in info.items():
                    st.text(f"{key}: {value}")
                
                # Visualization controls
                colormap = st.sidebar.selectbox(
                    "Color Map",
                    ['RdBu', 'RdGy', 'Spectral', 'RdYlBu', 'Viridis'],
                    index=0
                )
                
                contrast = st.sidebar.slider(
                    "Contrast",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1
                )
                
                # Plot the seismic section
                st.markdown("### Seismic Section Visualization")
                fig = plot_seismic_section(data, samples, colormap, contrast)
                st.plotly_chart(fig, use_container_width=True)
                
                # Reservoir Prediction Section
                st.markdown("### Reservoir Prediction")
                if st.button("Run Reservoir Prediction"):
                    with st.spinner("Running reservoir prediction..."):
                        # Prepare data for prediction
                        windows, positions = prepare_data_for_prediction(data)
                        
                        # Make predictions
                        predictions = predict_reservoirs(model, windows)
                        
                        # Create prediction map
                        prediction_map = create_prediction_map(predictions, positions, data.shape)
                        
                        # Store predictions in session state
                        st.session_state['prediction_map'] = prediction_map
                        
                        # Plot results with overlay
                        st.markdown("### Prediction Results")
                        fig_overlay = plot_prediction_overlay(data, prediction_map, samples, colormap, contrast)
                        st.plotly_chart(fig_overlay, use_container_width=True)
                        
                        # Display prediction statistics
                        st.markdown("### Prediction Statistics")
                        stats = {
                            "Average Reservoir Probability": f"{np.mean(prediction_map):.2%}",
                            "Max Reservoir Probability": f"{np.max(prediction_map):.2%}",
                            "Area Above 50% Probability": f"{np.mean(prediction_map > 0.5):.2%}"
                        }
                        for key, value in stats.items():
                            st.text(f"{key}: {value}")
                
                # Add trace extraction functionality
                st.markdown("### Trace Analysis")
                trace_number = st.number_input(
                    "Select Trace Number",
                    min_value=0,
                    max_value=data.shape[0]-1,
                    value=0
                )
                
                if st.button("Extract Trace"):
                    # Plot single trace
                    trace_fig = go.Figure()
                    trace_fig.add_trace(go.Scatter(
                        y=samples,
                        x=data[int(trace_number)],
                        mode='lines',
                        name=f'Trace {trace_number}'
                    ))
                    trace_fig.update_layout(
                        title=f"Trace {trace_number}",
                        xaxis_title="Amplitude",
                        yaxis_title="Time/Depth",
                        height=400
                    )
                    st.plotly_chart(trace_fig, use_container_width=True)
            else:
                st.error("Failed to load the file. Please check if it's a valid SEG-Y file.")
    else:
        st.info("Please upload a SEG-Y format seismic data file to begin.")
        st.markdown("""
        ### Supported Data Format:
        - SEG-Y files (.sgy, .segy)
        
        ### Expected Data:
        - 2D or 3D seismic data
        - Standard SEG-Y format
        """)

if __name__ == "__main__":
    main()
