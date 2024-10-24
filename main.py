import streamlit as st
import numpy as np
import pandas as pd
import segyio
import io
import os

st.set_page_config(
    page_title="Seismic Data Interpreter",
    page_icon="🌍",
    layout="wide"
)

st.title("Seismic Data Interpretation Application")
st.subheader("Data Loading Module")

def load_segy_file(uploaded_file):
    """Load and validate SEG-Y file"""
    try:
        # Save the uploaded file temporarily
        with open('temp.sgy', 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Open the SEG-Y file
        with segyio.open('temp.sgy', 'r', strict=False) as segy:
            # Get basic information
            n_traces = segy.tracecount
            sample_rate = segy.samples[1] - segy.samples[0]
            n_samples = len(segy.samples)
            
            # Read the first trace to get data type and range
            first_trace = segy.trace[0]
            
            info = {
                "Number of Traces": n_traces,
                "Sample Rate (ms)": sample_rate,
                "Samples per Trace": n_samples,
                "Data Range": f"{first_trace.min():.2f} to {first_trace.max():.2f}",
                "Data Type": first_trace.dtype
            }
            
            return info, True
            
    except Exception as e:
        st.error(f"Error loading SEG-Y file: {str(e)}")
        return None, False
    finally:
        # Clean up temporary file
        if os.path.exists('temp.sgy'):
            os.remove('temp.sgy')

def main():
    # File upload section
    st.markdown("### Upload Seismic Data")
    uploaded_file = st.file_uploader("Choose a SEG-Y file", type=['sgy', 'segy'])
    
    if uploaded_file is not None:
        with st.spinner('Loading seismic data...'):
            info, success = load_segy_file(uploaded_file)
            
            if success:
                st.success("File loaded successfully!")
                
                # Display file information
                st.markdown("### Seismic Data Information")
                for key, value in info.items():
                    st.text(f"{key}: {value}")
                
                # Store the file information in session state
                st.session_state['seismic_info'] = info
                st.session_state['filename'] = uploaded_file.name
            else:
                st.error("Failed to load the file. Please check if it's a valid SEG-Y file.")
    
    # Display instructions if no file is uploaded
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
