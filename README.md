# SeismoVision

## Overview
SeismoVision is a sophisticated seismic data interpretation application designed for reservoir identification using advanced machine learning techniques. This web-based tool provides an intuitive interface for geoscientists and engineers to analyze and interpret seismic data efficiently.

## Features and Capabilities
- **Data Upload**: Support for SEG-Y format seismic data files
- **Interactive Visualization**: 
  - Dynamic seismic section display
  - Customizable color maps and contrast controls
  - Interactive trace analysis
- **Reservoir Identification**:
  - Machine learning-based reservoir prediction
  - Real-time prediction overlay
  - Statistical analysis of predictions
- **Data Analysis Tools**:
  - Trace extraction and analysis
  - Amplitude analysis
  - Cross-sectional visualization

## Installation
1. Clone the repository
2. Install the required dependencies:
```bash
pip install streamlit numpy pandas segyio plotly scikit-learn
```

## Usage
1. Start the application:
```bash
streamlit run main.py
```
2. Access the application through your web browser at `http://localhost:5000`
3. Upload your SEG-Y format seismic data
4. Use the sidebar controls to adjust visualization parameters
5. Run reservoir predictions and analyze results

## Dependencies
- streamlit
- numpy
- pandas
- segyio
- plotly
- scikit-learn
- joblib

## Development
The application is built using Python and Streamlit framework, focusing on:
- Clean, maintainable code structure
- Efficient data processing
- Interactive visualization capabilities
- Machine learning integration

## Attribution
Developed by Tommy Xaypanya on 10/23/2024
