import streamlit as st
import os
import json
from utils.tao_utils import setup_tao_environment, create_tao_config, run_tao_command
from utils.data_utils import split_dataset, visualize_samples

st.set_page_config(
    page_title="Efficient Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("Efficient Model Training with TAO Toolkit")

# NGC API Key Input
api_key = st.text_input("Enter your NGC API Key:", type="password")
if api_key:
    try:
        setup_tao_environment(api_key)
        st.success("Successfully configured NGC environment!")
    except Exception as e:
        st.error(f"Failed to configure NGC environment: {str(e)}")

# Dataset Configuration
st.header("Dataset Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Paths")
    data_root = st.text_input("Data Root Directory:", "/workspace/tao-experiments/data")
    
    # Dataset split
    if st.button("Split Dataset"):
        try:
            train_files, val_files = split_dataset(
                os.path.join(data_root, "images/all_images"),
                os.path.join(data_root, "images/train"),
                os.path.join(data_root, "images/val")
            )
            st.success(f"Split dataset into {len(train_files)} training and {len(val_files)} validation images")
        except Exception as e:
            st.error(f"Failed to split dataset: {str(e)}")

with col2:
    st.subheader("Augmentation Settings")
    augment = st.checkbox("Enable Augmentation", value=True)
    hflip_prob = st.slider("Horizontal Flip Probability", 0.0, 1.0, 0.5)
    vflip_prob = st.slider("Vertical Flip Probability", 0.0, 1.0, 0.5)
    crop_prob = st.slider("Crop and Resize Probability", 0.0, 1.0, 0.5)

# Preview Dataset
if st.button("Preview Dataset"):
    try:
        fig = visualize_samples(
            os.path.join(data_root, "images/train"),
            os.path.join(data_root, "masks/train")
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to visualize samples: {str(e)}")

# Model Configuration
st.header("Model Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Architecture")
    model_width = st.number_input("Input Width", value=512, step=16)
    model_height = st.number_input("Input Height", value=512, step=16)
    model_channels = st.number_input("Input Channels", value=3, min_value=1, max_value=3)
    num_layers = st.selectbox("Number of Layers", [10, 18, 34, 50, 101], index=1)

with col2:
    st.subheader("Training Parameters")
    batch_size = st.number_input("Batch Size", value=1, min_value=1)
    epochs = st.number_input("Number of Epochs", value=5, min_value=1)
    learning_rate = st.number_input("Learning Rate", value=0.0001, format="%.4f")
    checkpoint_interval = st.number_input("Checkpoint Interval", value=5, min_value=1)

# Create Configuration
if st.button("Create Configuration"):
    try:
        # Dataset config
        dataset_config = create_tao_config("dataset", {
            "dataset": "custom",
            "augment": augment,
            "hflip_probability": hflip_prob,
            "vflip_probability": vflip_prob,
            "crop_and_resize_prob": crop_prob,
            "input_image_type": "color",
            "train_images_path": f"{data_root}/images/train",
            "train_masks_path": f"{data_root}/masks/train",
            "val_images_path": f"{data_root}/images/val",
            "val_masks_path": f"{data_root}/masks/val"
        })

        # Model config
        model_config = create_tao_config("model", {
            "width": model_width,
            "height": model_height,
            "channels": model_channels,
            "num_layers": num_layers,
            "all_projections": True,
            "arch": "resnet",
            "precision": "FLOAT32"
        })

        # Training config
        training_config = create_tao_config("training", {
            "batch_size": batch_size,
            "epochs": epochs,
            "log_steps": 10,
            "checkpoint_interval": checkpoint_interval,
            "loss": "cross_dice_sum",
            "learning_rate": learning_rate,
            "regularizer_type": "L2",
            "regularizer_weight": 2e-5,
            "adam_epsilon": 9.99999993923e-09,
            "adam_beta1": 0.899999976158,
            "adam_beta2": 0.999000012875
        })

        # Save configs
        os.makedirs("specs", exist_ok=True)
        
        with open("specs/dataset_config.txt", "w") as f:
            f.write(dataset_config)
        with open("specs/model_config.txt", "w") as f:
            f.write(model_config)
        with open("specs/training_config.txt", "w") as f:
            f.write(training_config)
            
        # Combine configs
        combined_config = f"{dataset_config}\n{model_config}\n{training_config}"
        with open("specs/combined_config.txt", "w") as f:
            f.write(combined_config)
            
        st.success("Created configuration files successfully!")
        
    except Exception as e:
        st.error(f"Failed to create configuration: {str(e)}")

# Training
st.header("Model Training")

if st.button("Start Training"):
    try:
        with st.spinner("Training model..."):
            result = run_tao_command(
                "train",
                "specs/combined_config.txt",
                "experiments/resnet18",
                api_key
            )
            
            if result.returncode == 0:
                st.success("Training completed successfully!")
                st.text(result.stdout)
            else:
                st.error("Training failed!")
                st.text(result.stderr)
                
    except Exception as e:
        st.error(f"Training failed: {str(e)}")

# Model Evaluation
st.header("Model Evaluation")

if st.button("Evaluate Model"):
    try:
        with st.spinner("Evaluating model..."):
            result = run_tao_command(
                "evaluate",
                "specs/combined_config.txt",
                "experiments/resnet18",
                api_key
            )
            
            if result.returncode == 0:
                st.success("Evaluation completed successfully!")
                
                # Try to parse and display metrics
                try:
                    metrics = json.loads(result.stdout)
                    st.json(metrics)
                except:
                    st.text(result.stdout)
            else:
                st.error("Evaluation failed!")
                st.text(result.stderr)
                
    except Exception as e:
        st.error(f"Evaluation failed: {str(e)}") 