import os
import json
import subprocess
from typing import Dict, Any

def setup_tao_environment(api_key: str) -> None:
    """Setup TAO environment with NGC API key."""
    # Create config dictionary
    config_dict = {
        'apikey': api_key,
        'format_type': 'json',
        'org': 'nvidia'
    }
    
    # Write config file
    config_path = os.path.expanduser('~/.ngc/config')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(';WARNING - This is a machine generated file. Do not edit manually.\n')
        f.write(';WARNING - To update local config settings, see "ngc config set -h"\n')
        f.write('\n[CURRENT]\n')
        for k, v in config_dict.items():
            f.write(f'{k}={v}\n')
            
    # Login to NGC docker registry
    subprocess.run(['docker', 'login', '-u', '$oauthtoken', '-p', api_key, 'nvcr.io'])
    
    # Pull TAO containers
    subprocess.run(['docker', 'pull', 'nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5'])
    subprocess.run(['docker', 'pull', 'nvcr.io/nvidia/tao/tao-toolkit:5.5.0-deploy'])

def create_tao_config(config_type: str, params: Dict[str, Any]) -> str:
    """Create TAO configuration file content."""
    if config_type == "dataset":
        return f"""dataset_config {{
  dataset: "{params['dataset']}"
  augment: {str(params['augment']).lower()}
  augmentation_config {{
    spatial_augmentation {{
      hflip_probability: {params['hflip_probability']}
      vflip_probability: {params['vflip_probability']}
      crop_and_resize_prob: {params['crop_and_resize_prob']}
    }}
  }}
  input_image_type: "{params['input_image_type']}"
  train_images_path: "{params['train_images_path']}"
  train_masks_path: "{params['train_masks_path']}"
  val_images_path: "{params['val_images_path']}"
  val_masks_path: "{params['val_masks_path']}"
}}"""
    elif config_type == "model":
        return f"""model_config {{
  model_input_width: {params['width']}
  model_input_height: {params['height']}
  model_input_channels: {params['channels']}
  num_layers: {params['num_layers']}
  all_projections: {str(params['all_projections']).lower()}
  arch: "{params['arch']}"
  training_precision {{
    backend_floatx: {params['precision']}
  }}
}}"""
    elif config_type == "training":
        return f"""training_config {{
  batch_size: {params['batch_size']}
  epochs: {params['epochs']}
  log_summary_steps: {params['log_steps']}
  checkpoint_interval: {params['checkpoint_interval']}
  loss: "{params['loss']}"
  learning_rate: {params['learning_rate']}
  regularizer {{
    type: {params['regularizer_type']}
    weight: {params['regularizer_weight']}
  }}
  optimizer {{
    adam {{
      epsilon: {params['adam_epsilon']}
      beta1: {params['adam_beta1']}
      beta2: {params['adam_beta2']}
    }}
  }}
}}"""
    else:
        raise ValueError(f"Unknown config type: {config_type}")

def run_tao_command(command: str, spec_file: str, output_dir: str, model_key: str) -> subprocess.CompletedProcess:
    """Run a TAO command with proper environment setup."""
    base_cmd = f"tao model unet {command}"
    full_cmd = f"{base_cmd} -e {spec_file} -r {output_dir} -k {model_key}"
    
    if command == "train":
        full_cmd += " -n resnet18"
    
    return subprocess.run(full_cmd.split(), capture_output=True, text=True)

def export_model(model_path: str, spec_file: str, output_dir: str, model_key: str) -> subprocess.CompletedProcess:
    """Export trained model to ONNX format."""
    return subprocess.run([
        "tao", "model", "unet", "export",
        "-m", model_path,
        "-e", spec_file,
        "-k", model_key,
        "--gen_ds_config"
    ], capture_output=True, text=True)

def generate_tensorrt_engine(
    model_path: str, 
    spec_file: str, 
    output_dir: str, 
    model_key: str,
    max_batch_size: int = 8
) -> subprocess.CompletedProcess:
    """Generate TensorRT engine from exported model."""
    return subprocess.run([
        "tao", "deploy", "unet", "gen_trt_engine",
        "-m", model_path,
        "-e", spec_file,
        "-r", output_dir,
        "-k", model_key,
        "--engine_file", f"{output_dir}/model.engine",
        "--max_batch_size", str(max_batch_size)
    ], capture_output=True, text=True) 