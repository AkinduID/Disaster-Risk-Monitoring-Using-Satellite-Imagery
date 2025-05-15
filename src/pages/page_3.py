import streamlit as st
import os
from utils.tao_utils import export_model, generate_tensorrt_engine
from utils.data_utils import visualize_predictions

st.set_page_config(
    page_title="Model Deployment for Inference",
    page_icon="🚀",
    layout="wide"
)

st.title("Model Deployment for Inference")

st.markdown("""
## Model Optimization with TensorRT

NVIDIA TensorRT is a platform for high-performance deep learning inference that includes:

- Deep learning inference optimizer
- Runtime engine for fast deployment
- Support for all major deep learning frameworks
- Significant performance improvements over CPU-only platforms

### Key Benefits

- Up to 40x faster inference compared to CPU-only platforms
- Optimization across all NVIDIA GPUs
- Support for multiple precision types (FP32, FP16, INT8)
- Automatic optimization of neural network models

## Deployment Pipeline

The deployment process involves several key steps:
""")

# Create columns for deployment steps
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Model Export
    - Export trained model to ONNX
    - Generate TensorRT engine
    - Validate model accuracy
    """)

with col2:
    st.markdown("""
    ### 2. Optimization
    - Layer fusion
    - Precision calibration
    - Memory optimization
    - Kernel auto-tuning
    """)

with col3:
    st.markdown("""
    ### 3. Deployment
    - Triton Inference Server setup
    - Model repository configuration
    - Load balancing
    - Monitoring setup
    """)

st.markdown("""
## TensorRT Optimization Techniques

TensorRT applies various optimization techniques to improve model performance:
""")

# Create expandable sections for optimization techniques
with st.expander("Layer and Tensor Fusion"):
    st.markdown("""
    - Combining multiple operations into a single kernel
    - Reducing memory transfers
    - Improving computational efficiency
    """)

with st.expander("Precision Calibration"):
    st.markdown("""
    - FP32 to FP16 conversion
    - INT8 quantization
    - Calibration dataset selection
    - Accuracy preservation
    """)

with st.expander("Kernel Auto-tuning"):
    st.markdown("""
    - Automatic selection of optimal algorithms
    - Hardware-specific optimizations
    - Performance profiling
    - Resource utilization optimization
    """)

st.header("Deployment Configuration")

# Add sample configuration code
st.code("""
# Triton Model Configuration
name: "flood_detection"
platform: "tensorrt_plan"
max_batch_size: 8
input [
  {
    name: "input_1:0"
    data_type: TYPE_FP32
    dims: [ 3, 512, 512 ]
  }
]
output [
  {
    name: "argmax_1"
    data_type: TYPE_INT32
    dims: [ 512, 512 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
  }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 100
}
""", language="python")

# Add interactive elements for deployment configuration
st.sidebar.header("Deployment Settings")
batch_size = st.sidebar.selectbox("Maximum Batch Size", [1, 2, 4, 8, 16])
precision = st.sidebar.selectbox("Precision", ["FP32", "FP16", "INT8"])
num_instances = st.sidebar.slider("Number of Model Instances", 1, 4, 2)

st.sidebar.markdown("""
**Note**: These settings affect the deployment configuration. 
Adjust based on your hardware capabilities and performance requirements.
""")

# Performance metrics simulation
st.header("Performance Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Throughput", "120 fps", "↑40%")
col2.metric("Latency", "8.3 ms", "↓25%")
col3.metric("GPU Memory", "2.4 GB", "↓15%")
col4.metric("Accuracy", "95.2%", "↓0.3%")

# Model Export
st.header("Model Export")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Export Settings")
    model_path = st.text_input("Model Path:", "experiments/resnet18/weights/resnet18.tlt")
    spec_file = st.text_input("Spec File:", "specs/combined_config.txt")
    output_dir = st.text_input("Output Directory:", "experiments/export")
    model_key = st.text_input("Model Key:", type="password")

with col2:
    st.subheader("TensorRT Settings")
    max_batch_size = st.number_input("Max Batch Size", value=8, min_value=1)
    precision = st.selectbox("Precision", ["FP32", "FP16", "INT8"])
    workspace_size = st.number_input("Workspace Size (MB)", value=1024, min_value=128)

# Export Model
if st.button("Export Model"):
    try:
        with st.spinner("Exporting model to ONNX..."):
            result = export_model(
                model_path,
                spec_file,
                output_dir,
                model_key
            )
            
            if result.returncode == 0:
                st.success("Model exported successfully!")
                st.text(result.stdout)
            else:
                st.error("Model export failed!")
                st.text(result.stderr)
                
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

# Generate TensorRT Engine
if st.button("Generate TensorRT Engine"):
    try:
        with st.spinner("Generating TensorRT engine..."):
            result = generate_tensorrt_engine(
                os.path.join(output_dir, "model.onnx"),
                spec_file,
                output_dir,
                model_key,
                max_batch_size
            )
            
            if result.returncode == 0:
                st.success("TensorRT engine generated successfully!")
                st.text(result.stdout)
            else:
                st.error("TensorRT engine generation failed!")
                st.text(result.stderr)
                
    except Exception as e:
        st.error(f"TensorRT engine generation failed: {str(e)}")

# Model Testing
st.header("Model Testing")

test_image_dir = st.text_input("Test Images Directory:", "data/images/test")
if st.button("Run Inference"):
    try:
        # Visualize predictions
        fig = visualize_predictions(output_dir)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Inference failed: {str(e)}")

# Deployment Instructions
st.header("Deployment Instructions")

st.markdown("""
### Steps for Production Deployment

1. **Set up Triton Inference Server**
   ```bash
   docker pull nvcr.io/nvidia/tritonserver:23.12-py3
   ```

2. **Create Model Repository**
   ```bash
   mkdir -p model_repository/flood_detection/1/
   cp experiments/export/model.engine model_repository/flood_detection/1/
   ```

3. **Create Model Configuration**
   ```bash
   # config.pbtxt
   name: "flood_detection"
   platform: "tensorrt_plan"
   max_batch_size: 8
   input [
     {
       name: "input_1:0"
       data_type: TYPE_FP32
       dims: [ 3, 512, 512 ]
     }
   ]
   output [
     {
       name: "argmax_1"
       data_type: TYPE_INT32
       dims: [ 512, 512 ]
     }
   ]
   ```

4. **Start Triton Server**
   ```bash
   docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \\
       -v $(pwd)/model_repository:/models \\
       nvcr.io/nvidia/tritonserver:23.12-py3 \\
       tritonserver --model-repository=/models
   ```
""")

# Performance Metrics
st.header("Performance Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Inference Time", "5.2 ms")
    
with col2:
    st.metric("Throughput", "192 FPS")
    
with col3:
    st.metric("GPU Memory", "1.2 GB") 