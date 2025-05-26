import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def count_images_in_split(split_path):
    """Count number of images in a split directory"""
    image_extensions = ['.jpg', '.jpeg', '.png']
    count = 0
    for ext in image_extensions:
        count += len(glob.glob(os.path.join(split_path, f'*{ext}')))
    return count

def count_annotations(label_path):
    """Count annotations per class in a split"""
    class_counts = {i: 0 for i in range(10)}  # Initialize counts for 10 classes
    if not os.path.exists(label_path):
        return class_counts
    for label_file in glob.glob(os.path.join(label_path, '*.txt')):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    return class_counts

def plot_split_distribution(split_counts, class_names):
    """Plot the distribution of images across splits"""
    plt.figure(figsize=(10, 6))
    splits = list(split_counts.keys())
    counts = list(split_counts.values())
    
    plt.bar(splits, counts)
    plt.title('Number of Images per Split')
    plt.xlabel('Dataset Split')
    plt.ylabel('Number of Images')
    
    # Add count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('split_distribution.png')
    plt.close()

def plot_class_distribution_per_split(class_counts, class_names):
    """Plot the distribution of annotations per class for each split separately, with value labels on bars"""
    for split, counts in class_counts.items():
        plt.figure(figsize=(12, 6))
        class_ids = list(range(len(class_names)))
        values = [counts[i] for i in class_ids]
        ax = sns.barplot(x=class_names, y=values, palette="viridis")
        plt.title(f'Class Distribution in {split.capitalize()} Set')
        plt.xlabel('Class')
        plt.ylabel('Number of Annotations')
        plt.xticks(rotation=45, ha='right')
        # Add value labels on top of each bar
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.01, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'class_distribution_{split}.png')
        plt.close()

def main():
    # Load dataset configuration
    yaml_path = 'model/dataset/data.yaml'
    data_config = load_yaml(yaml_path)
    class_names = data_config['names']
    
    # Get absolute paths for splits
    base_path = os.path.dirname(yaml_path)
    split_paths = {
        'train': os.path.join(base_path, data_config['train']),
        'val': os.path.join(base_path, data_config['val']),
        'test': os.path.join(base_path, data_config['test'])
    }
    label_paths = {
        'train': os.path.join(base_path, '../dataset/labels/train'),
        'val': os.path.join(base_path, '../dataset/labels/val'),
        'test': os.path.join(base_path, '../dataset/labels/test')
    }
    
    # Count images in each split
    split_counts = {split: count_images_in_split(path) 
                   for split, path in split_paths.items()}
    
    # Count annotations per class in each split
    class_counts = {split: count_annotations(label_path)
                   for split, label_path in label_paths.items()}
    
    # Create visualizations
    plot_split_distribution(split_counts, class_names)
    plot_class_distribution_per_split(class_counts, class_names)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print("-" * 50)
    print("Images per split:")
    for split, count in split_counts.items():
        print(f"{split}: {count} images")
    
    print("\nAnnotations per class:")
    for split in split_paths.keys():
        print(f"\n{split.upper()} split:")
        for class_id, count in class_counts[split].items():
            print(f"{class_names[class_id]}: {count} annotations")

if __name__ == "__main__":
    main() 