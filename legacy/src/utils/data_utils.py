import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple

def split_dataset(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    split_ratio: float = 0.75
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation sets."""
    # Get all image files
    files = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Randomly split files
    train_files = random.sample(files, int(len(files) * split_ratio))
    val_files = [f for f in files if f not in train_files]
    
    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy files to respective directories
    for f in train_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(train_dir, f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            
    for f in val_files:
        src = os.path.join(source_dir, f)
        dst = os.path.join(val_dir, f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            
    return train_files, val_files

def visualize_samples(
    images_dir: str,
    masks_dir: str,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Visualize sample images and their masks."""
    # Get random samples
    image_files = os.listdir(images_dir)
    samples = random.sample(image_files, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(figsize[0], figsize[1] * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
        
    # Set titles for columns
    axes[0, 0].set_title('Original Image')
    axes[0, 1].set_title('Mask')
    axes[0, 2].set_title('Overlay')
    
    for idx, fname in enumerate(samples):
        # Load image and mask
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # Create overlay
        overlay = img.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay for flood areas
        
        # Display
        axes[idx, 0].imshow(img)
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 2].imshow(overlay)
        
        # Remove ticks
        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    return fig

def visualize_predictions(
    model_output_dir: str,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (20, 5)
) -> plt.Figure:
    """Visualize model predictions."""
    # Get paths
    overlay_path = os.path.join(model_output_dir, 'vis_overlay_tlt')
    inference_path = os.path.join(model_output_dir, 'mask_labels_tlt')
    
    # Get random samples
    image_files = os.listdir(overlay_path)
    samples = random.sample(image_files, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(figsize[0], figsize[1] * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
        
    # Set titles
    axes[0, 0].set_title('Overlay')
    axes[0, 1].set_title('Predicted Mask')
    
    for idx, fname in enumerate(samples):
        # Load images
        overlay = plt.imread(os.path.join(overlay_path, fname))
        mask = plt.imread(os.path.join(inference_path, fname))
        
        # Display
        axes[idx, 0].imshow(overlay)
        axes[idx, 1].imshow(mask, cmap='gray')
        
        # Remove ticks
        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    return fig 