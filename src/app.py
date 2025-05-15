import streamlit as st

st.set_page_config(
    page_title="Disaster Risk Monitoring Using Satellite Imagery",
    page_icon="🛰️",
    layout="wide"
)

st.title("Disaster Risk Monitoring Using Satellite Imagery")

st.markdown("""
Welcome to the Disaster Risk Monitoring Using Satellite Imagery application. This application demonstrates how to:

1. Process and analyze satellite imagery data for disaster monitoring
2. Train efficient deep learning models for flood detection
3. Deploy models for real-time inference
4. Apply the system to real flood event case studies

### Key Features

- 🛰️ Satellite imagery processing and analysis
- 🤖 Deep learning model training with TAO Toolkit
- 🚀 Model deployment with TensorRT
- 📊 Real-time flood detection and monitoring
- 📈 Performance optimization and scaling

### Navigation

Use the sidebar to navigate between different sections of the application:

1. **Disaster Risk Monitoring Systems** - Learn about the fundamentals and data pre-processing
2. **Efficient Model Training** - Explore model training with TAO Toolkit
3. **Model Deployment** - Understand deployment strategies and optimization
4. **Case Study** - See real-world application with UNOSAT flood event data

### Getting Started

Select a page from the sidebar to begin exploring the application. Each page contains detailed information, interactive elements, and code examples to help you understand the complete workflow of building a disaster risk monitoring system.
""")

st.sidebar.success("Select a page above.")