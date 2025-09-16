#!/usr/bin/env python3
"""
Dataset preparation script for 12-class base training.
This script converts your base_training dataset to the format expected by S2A-Net.
"""

import os
import pickle
import json
import cv2
import numpy as np
from pathlib import Path
import argparse

def create_dataset_structure(base_path):
    """Create the required dataset directory structure."""
    base_path = Path(base_path)
    
    # Create required directories
    (base_path / 'annotations').mkdir(exist_ok=True)
    (base_path / 'images').mkdir(exist_ok=True)
    
    print(f"Created dataset structure in: {base_path}")
    return base_path

def convert_annotations_to_mmdet_format(annotation_file, image_dir, output_file):
    """
    Convert your annotation format to MMDetection format.
    Assumes your annotations are in a format with:
    - image_path: path to image
    - bboxes: list of rotated bounding boxes
    - labels: list of class labels (0-11 for 12 classes)
    """
    
    # Load your annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)  # or pickle.load(f) if using pickle
    
    mmdet_annotations = []
    
    for idx, ann in enumerate(annotations):
        # Get image info
        image_path = ann['image_path']
        image_name = os.path.basename(image_path)
        
        # Copy image to images directory
        src_path = image_path
        dst_path = os.path.join('images', image_name)
        if os.path.exists(src_path):
            # Copy image if not already in the right location
            if not os.path.exists(dst_path):
                import shutil
                shutil.copy2(src_path, dst_path)
        
        # Get image dimensions
        img = cv2.imread(dst_path)
        height, width = img.shape[:2]
        
        # Convert bboxes to MMDetection format
        bboxes = []
        labels = []
        
        for bbox, label in zip(ann['bboxes'], ann['labels']):
            # Convert your bbox format to [x, y, w, h, angle]
            # Adjust this based on your actual bbox format
            if len(bbox) == 8:  # If you have 4 corner points
                # Convert 4 corner points to [x, y, w, h, angle]
                bbox = convert_corners_to_xywha(bbox)
            elif len(bbox) == 5:  # If already in [x, y, w, h, angle] format
                bbox = bbox
            else:
                print(f"Warning: Unknown bbox format with {len(bbox)} points")
                continue
            
            bboxes.append(bbox)
            labels.append(label)  # Should be 0-11 for 12 classes
        
        # Create MMDetection annotation entry
        mmdet_ann = {
            'filename': image_name,
            'width': width,
            'height': height,
            'ann': {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64),
                'bboxes_ignore': np.zeros((0, 5), dtype=np.float32),
                'labels_ignore': np.zeros((0,), dtype=np.int64)
            }
        }
        
        mmdet_annotations.append(mmdet_ann)
    
    # Save in pickle format
    with open(output_file, 'wb') as f:
        pickle.dump(mmdet_annotations, f)
    
    print(f"Converted {len(mmdet_annotations)} annotations to {output_file}")

def convert_corners_to_xywha(corners):
    """
    Convert 4 corner points to [x, y, w, h, angle] format.
    corners: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    corners = np.array(corners).reshape(4, 2)
    
    # Calculate center
    center = np.mean(corners, axis=0)
    
    # Calculate width and height
    # Find the two longest sides
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        sides.append((length, i, (i + 1) % 4))
    
    sides.sort(reverse=True)
    w = sides[0][0]  # Longest side
    h = sides[1][0]  # Second longest side
    
    # Calculate angle
    # Use the longest side to determine angle
    p1_idx, p2_idx = sides[0][1], sides[0][2]
    p1, p2 = corners[p1_idx], corners[p2_idx]
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    angle = np.degrees(angle)
    
    return [center[0], center[1], w, h, angle]

def create_sample_annotations():
    """Create sample annotation files for testing."""
    
    # Sample data structure - replace with your actual data loading
    sample_annotations = [
        {
            'image_path': 'sample1.jpg',
            'bboxes': [[100, 100, 200, 100, 200, 200, 100, 200]],  # 4 corner points
            'labels': [0]  # Class 0
        },
        {
            'image_path': 'sample2.jpg', 
            'bboxes': [[150, 150, 250, 150, 250, 250, 150, 250]],
            'labels': [1]  # Class 1
        }
    ]
    
    return sample_annotations

def main():
    parser = argparse.ArgumentParser(description='Prepare base training dataset')
    parser.add_argument('--base_path', default='data/base_training', 
                       help='Path to base training dataset')
    parser.add_argument('--annotation_file', 
                       help='Path to your annotation file')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample annotation files for testing')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    create_dataset_structure(base_path)
    
    if args.create_sample:
        print("Creating sample annotation files...")
        sample_anns = create_sample_annotations()
        
        # Save sample annotations
        with open(base_path / 'annotations' / 'sample_annotations.json', 'w') as f:
            json.dump(sample_anns, f, indent=2)
        
        print("Sample annotations created. Please replace with your actual data.")
        print("Expected format:")
        print("- Each annotation should have: image_path, bboxes, labels")
        print("- bboxes: list of [x1, y1, x2, y2, x3, y3, x4, y4] or [x, y, w, h, angle]")
        print("- labels: list of class indices (0-11 for 12 classes)")
    
    if args.annotation_file:
        print(f"Converting annotations from {args.annotation_file}...")
        convert_annotations_to_mmdet_format(
            args.annotation_file,
            base_path / 'images',
            base_path / 'annotations' / 'train.pkl'
        )

if __name__ == '__main__':
    main()

