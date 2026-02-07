import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from data.dataset import WESADDataset, load_wesad_data
from utils.config import WESAD_dataset_path

print("Loading WESAD data...")
subject_data = load_wesad_data(WESAD_dataset_path)

# Create dataset
dataset = WESADDataset(subject_data, mode='classifier')

# Check label distribution
all_labels = []
print(f"Analyzing {len(dataset)} samples...")
for i in range(len(dataset)):
    _, label = dataset[i]
    all_labels.append(label.item())

labels_array = np.array(all_labels)
unique, counts = np.unique(labels_array, return_counts=True)

print("="*60)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*60)
print(f"Total samples: {len(labels_array)}")
print(f"\nClass distribution:")
for label, count in zip(unique, counts):
    percentage = (count / len(labels_array)) * 100
    print(f"  Class {label}: {count:4d} samples ({percentage:5.2f}%)")

print(f"\nMost common class: {unique[np.argmax(counts)]} ({max(counts)/len(labels_array)*100:.2f}%)")
print("="*60)

# Check if 78.43% matches any class distribution
for label, count in zip(unique, counts):
    percentage = (count / len(labels_array)) * 100
    if abs(percentage - 78.43) < 0.5:
        print(f"\n⚠️  WARNING: Class {label} has {percentage:.2f}% of samples!")
        print(f"   This matches your stuck accuracy of 78.43%!")
        print(f"   The model is predicting class {label} for ALL samples!")
        print(f"\n   Solution needed: Handle class imbalance with weighted loss")
        break
else:
    print("\n✓  Accuracy doesn't exactly match majority class")
    print("   But model may still be biased towards majority class")
