import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

DATA_DIR = 'dataset'


def load_image(path):
    return np.array(Image.open(path))


def plot_samples(n_samples=4):
    train_dir = os.path.join(DATA_DIR, 'train')
    files = [f.replace('_sat.jpg', '') for f in os.listdir(train_dir) if f.endswith('_sat.jpg')]
    files = random.sample(files, min(n_samples, len(files)))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    
    for i, file_id in enumerate(files):
        sat_path = os.path.join(train_dir, f'{file_id}_sat.jpg')
        mask_path = os.path.join(train_dir, f'{file_id}_mask.png')
        
        sat_img = load_image(sat_path)
        mask_img = load_image(mask_path)
        
        axes[i, 0].imshow(sat_img)
        axes[i, 0].set_title(f'Satellite {file_id}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 1].set_title('Mask')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(sat_img)
        axes[i, 2].imshow(mask_img, alpha=0.5)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150)
    plt.show()


def analyze_dataset():
    splits = ['train', 'valid', 'test']
    
    print("Dataset Analysis")
    print("="*50)
    
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        sat_files = [f for f in os.listdir(split_dir) if f.endswith('_sat.jpg')]
        mask_files = [f for f in os.listdir(split_dir) if f.endswith('_mask.png')]
        
        print(f"\n{split.upper()}:")
        print(f"  Images: {len(sat_files)}")
        print(f"  Masks: {len(mask_files)}")
    
    print("\n" + "="*50)
    print("Class Distribution")
    print("="*50)
    
    class_df = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
    print(class_df.to_string(index=False))


def analyze_masks():
    train_dir = os.path.join(DATA_DIR, 'train')
    files = [f.replace('_mask.png', '') for f in os.listdir(train_dir) if f.endswith('_mask.png')]
    
    road_pixels = []
    total_pixels = []
    
    for file_id in files[:100]:
        mask_path = os.path.join(train_dir, f'{file_id}_mask.png')
        mask = load_image(mask_path)
        
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        
        road_pixels.append(np.sum(mask > 0))
        total_pixels.append(mask.size)
    
    road_ratio = np.array(road_pixels) / np.array(total_pixels)
    
    print(f"\nMask Statistics (first 100 samples):")
    print(f"  Mean road ratio: {road_ratio.mean():.2%}")
    print(f"  Std road ratio: {road_ratio.std():.2%}")
    print(f"  Min road ratio: {road_ratio.min():.2%}")
    print(f"  Max road ratio: {road_ratio.max():.2%}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(road_ratio, bins=30, edgecolor='black')
    plt.xlabel('Road Pixel Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Road Coverage')
    
    plt.subplot(1, 2, 2)
    sample_mask = load_image(os.path.join(train_dir, f'{files[0]}_mask.png'))
    if len(sample_mask.shape) == 3:
        sample_mask = sample_mask[:,:,0]
    plt.hist(sample_mask.flatten(), bins=50, edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.title('Mask Pixel Value Distribution')
    
    plt.tight_layout()
    plt.savefig('mask_analysis.png', dpi=150)
    plt.show()


def check_image_sizes():
    splits = ['train', 'valid', 'test']
    
    sizes = []
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        files = [f for f in os.listdir(split_dir) if f.endswith('_sat.jpg')][:10]
        for f in files:
            img = load_image(os.path.join(split_dir, f))
            sizes.append(img.shape)
    
    unique_sizes = list(set(sizes))
    print(f"\nImage Sizes: {unique_sizes}")


if __name__ == '__main__':
    analyze_dataset()
    check_image_sizes()
    plot_samples(4)
    analyze_masks()
