import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Disaster Risk Monitoring Systems",
    page_icon="🛰️",
    layout="wide"
)

st.title("Disaster Risk Monitoring Systems and Data Pre-processing")

st.markdown("""
## Disaster Risk Monitoring

Natural disasters such as floods, wildfires, droughts, and severe storms cause billions in damages 
and disrupt communities worldwide. Early detection and monitoring systems can help minimize their impact.

### Key Focus Areas:

1. **Flood Detection**
   - Overflow of water bodies
   - Accumulation of rainwater
   - River channel capacity exceedance

2. **Satellite Imagery**
   - Sentinel-1 SAR data
   - Day/night operation capability
   - Cloud cover penetration
   - 6-day repeat cycle

3. **Computer Vision Applications**
   - Classification
   - Object detection
   - Semantic segmentation
   - Instance segmentation
""")

# Create sample data for visualization
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
disaster_types = ['Flood', 'Wildfire', 'Drought', 'Storm']
data = []

for disaster in disaster_types:
    incidents = np.random.normal(loc=100, scale=20, size=len(dates))
    incidents = np.abs(incidents)  # Make all values positive
    for date, count in zip(dates, incidents):
        data.append({
            'Date': date,
            'Disaster_Type': disaster,
            'Incident_Count': count
        })

df = pd.DataFrame(data)

# Plot disaster incidents
st.header("Disaster Incidents Over Time")

fig = go.Figure()
for disaster in disaster_types:
    disaster_data = df[df['Disaster_Type'] == disaster]
    fig.add_trace(go.Scatter(
        x=disaster_data['Date'],
        y=disaster_data['Incident_Count'],
        name=disaster,
        mode='lines+markers'
    ))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Number of Incidents',
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Data Processing Pipeline
st.header("Data Processing Pipeline")

st.markdown("""
### Satellite Data Processing

The processing pipeline consists of several key steps:

1. **Data Collection**
   - Sentinel-1 SAR imagery
   - Ground truth data
   - Historical records

2. **Pre-processing**
   - Radiometric calibration
   - Speckle filtering
   - Terrain correction

3. **Data Augmentation**
   - Rotation
   - Flipping
   - Scaling
   - Noise addition
""")

# Create columns for key metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Satellite Coverage", "250,000 km²/day")
col2.metric("Resolution", "10 meters")
col3.metric("Revisit Time", "6 days")
col4.metric("Data Volume", "1.5 TB/day")

# DALI Pipeline
st.header("DALI Pipeline")

st.markdown("""
### NVIDIA DALI

DALI (Data Loading Library) is a collection of highly optimized building blocks for data pre-processing:

- GPU-accelerated data loading and augmentation
- Seamless integration with deep learning frameworks
- Support for various data formats and transformations
- Optimized memory handling and transfer
""")

# Add interactive elements
st.sidebar.header("Data Processing Parameters")

# Image processing parameters
image_size = st.sidebar.slider("Image Size", 128, 1024, 512, 128)
batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 8, 16, 32])
augmentation = st.sidebar.multiselect(
    "Augmentation Techniques",
    ["Rotation", "Flip", "Scale", "Noise", "Blur"],
    ["Rotation", "Flip"]
)

# Processing options
processing_device = st.sidebar.radio("Processing Device", ["GPU", "CPU"])
num_workers = st.sidebar.slider("Number of Workers", 1, 8, 4)

st.sidebar.markdown("""
**Note**: These parameters affect the data processing pipeline.
Adjust based on your available computational resources and requirements.
""")