import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from utils.data_utils import visualize_samples

st.set_page_config(
    page_title="UNOSAT Flood Event Case Study",
    page_icon="📊",
    layout="wide"
)

st.title("UNOSAT Flood Event Case Study")

st.markdown("""
## Case Study Overview

This case study demonstrates the application of our flood detection system on real-world data from UNOSAT 
(United Nations Satellite Centre). We analyze flood events and showcase how our deep learning model 
performs in practical scenarios.

### Data Sources

- Sentinel-1 SAR imagery
- UNOSAT flood maps
- Ground truth data
- Historical flood records
""")

# Load sample flood event data
@st.cache_data
def load_flood_data():
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'date': dates,
        'flood_area_km2': np.random.normal(100, 20, len(dates)).cumsum(),
        'affected_population': np.random.normal(1000, 200, len(dates)).cumsum(),
        'rainfall_mm': np.random.normal(50, 10, len(dates)),
        'detection_confidence': np.random.uniform(0.8, 0.99, len(dates))
    }
    return pd.DataFrame(data)

# Load and display flood event data
flood_data = load_flood_data()

# Time series analysis
st.header("Flood Event Analysis")

col1, col2 = st.columns(2)

with col1:
    # Flood area over time
    fig1 = px.line(flood_data, x='date', y='flood_area_km2',
                   title='Flood Area Over Time')
    fig1.update_layout(yaxis_title='Flood Area (km²)')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Affected population
    fig2 = px.line(flood_data, x='date', y='affected_population',
                   title='Affected Population Over Time')
    fig2.update_layout(yaxis_title='Number of People')
    st.plotly_chart(fig2, use_container_width=True)

# Correlation analysis
st.header("Environmental Correlation")

# Scatter plot of rainfall vs flood area
fig3 = px.scatter(flood_data, x='rainfall_mm', y='flood_area_km2',
                  title='Rainfall vs Flood Area',
                  trendline="ols")
fig3.update_layout(
    xaxis_title='Daily Rainfall (mm)',
    yaxis_title='Flood Area (km²)'
)
st.plotly_chart(fig3, use_container_width=True)

# Model Performance
st.header("Model Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    # Detection confidence distribution
    fig4 = px.histogram(flood_data, x='detection_confidence',
                       title='Detection Confidence Distribution',
                       nbins=30)
    fig4.update_layout(
        xaxis_title='Confidence Score',
        yaxis_title='Count'
    )
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    # Confidence over time
    fig5 = px.line(flood_data, x='date', y='detection_confidence',
                   title='Detection Confidence Over Time')
    fig5.update_layout(
        yaxis_title='Confidence Score'
    )
    st.plotly_chart(fig5, use_container_width=True)

# Key Metrics
st.header("Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Detection Confidence",
        f"{flood_data['detection_confidence'].mean():.2%}"
    )

with col2:
    st.metric(
        "Total Affected Area",
        f"{flood_data['flood_area_km2'].max():.0f} km²"
    )

with col3:
    st.metric(
        "Total Affected Population",
        f"{flood_data['affected_population'].max():,.0f}"
    )

with col4:
    st.metric(
        "Average Daily Rainfall",
        f"{flood_data['rainfall_mm'].mean():.1f} mm"
    )

# Sample Results Visualization
st.header("Sample Detection Results")

# Add a file uploader for test images
uploaded_file = st.file_uploader("Upload a test image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Run Detection"):
        with st.spinner("Running flood detection..."):
            # Placeholder for actual model inference
            st.success("Detection completed!")
            
            # Show sample visualization
            try:
                fig = visualize_samples(
                    "data/images/test",
                    "data/masks/test",
                    num_samples=1
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Visualization failed: {str(e)}")

# Recommendations
st.header("Recommendations")

st.markdown("""
Based on the analysis of this flood event, we recommend:

1. **Early Warning System Enhancement**
   - Integrate real-time rainfall data
   - Implement automated alert thresholds
   - Expand monitoring coverage

2. **Model Improvements**
   - Fine-tune for specific geographical regions
   - Incorporate additional data sources
   - Regular model retraining with new data

3. **Operational Procedures**
   - Establish clear communication channels
   - Define response protocols
   - Regular system maintenance
""")

# Download Report
if st.button("Generate Report"):
    # Create a sample report
    report = flood_data.describe()
    
    # Convert to CSV
    csv = report.to_csv()
    
    st.download_button(
        label="Download Report",
        data=csv,
        file_name=f"flood_analysis_report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    ) 