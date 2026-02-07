
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import normalize_signal, segment_signal

def load_wesad_data(dataset_path):
    """
    Loads WESAD dataset from the given path.
    Args:
        dataset_path: Path to the WESAD directory containing subject folders (S2, S3, ...).
    Returns:
        subject_data: Dictionary mapping subject IDs to their data.
    """
    subject_data = {}
    # WESAD subjects are usually S2-S17
    # valid_subjects = [f'S{i}' for i in range(2, 18) if i != 12] # S12 is often excluded or missing
    
    print(f"Loading WESAD data from {dataset_path}...")
    if not os.path.exists(dataset_path):
         print(f"Error: Path {dataset_path} does not exist.")
         return {}

    # Iterate over all items in directory
    for subject_id in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject_id)
        if os.path.isdir(subject_path) and subject_id.startswith('S'):
            pkl_file = os.path.join(subject_path, f'{subject_id}.pkl')
            if os.path.exists(pkl_file):
                print(f"Loading {subject_id}...")
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                        subject_data[subject_id] = data
                except Exception as e:
                    print(f"Error loading {subject_id}: {e}")
    
    print(f"Loaded {len(subject_data)} subjects.")
    return subject_data

class WESADDataset(Dataset):
    def __init__(self, subject_data, window_size=240, step_size=60, mode='train'):
        """
        Args:
            subject_data: Dictionary of subject data loaded from .pkl files.
            window_size: Window size in samples (e.g., 60s * 4Hz).
            step_size: Step size in samples for overlap.
            mode: 'train' or 'test'.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.mode = mode
        self.data_segments = []
        self.labels = []
        
        # Load and process data
        print(f"Processing data for {len(subject_data)} subjects...")
        self._process_data(subject_data)
        print(f"Total segments generated: {len(self.data_segments)}")
        
    def _process_data(self, subject_data):
        """
        Processes raw subject data into segments.
        Extracts EDA, Temp, and BVP signals.
        Resamples BVP to match 4Hz (EDA/Temp frequency) to simplify multimodal fusion.
        Normalization is applied per subject before segmentation.
        """
        # Iterate over each subject's data
        for subject_id, data in subject_data.items():
            print(f"Processing subject {subject_id}...")
            # Signal Data
            try:
                raw_eda = data['signal']['wrist']['EDA'].flatten()
                raw_temp = data['signal']['wrist']['TEMP'].flatten()
                raw_bvp = data['signal']['wrist']['BVP'].flatten()
                labels = data['label'].flatten()
                print(f"  Signals found. EDA length: {len(raw_eda)}")
            except KeyError as e:
                print(f"  Error extracting signals for {subject_id}: {e}")
                continue

            # Synchronize Labels (700Hz) to 4Hz (EDA)
            # EDA is sampled at 4Hz. Labels are provided at 700Hz.
            # Downsample labels by taking the mode every 175 samples (700/4)
            # For simplicity in this implementation, we assume labels are aligned 
            # or we re-sample signals. WESAD synchronization is complex.
            # A common approach is using the chest index, but for wrist only we approximate.
            
            # Let's resample BVP (64Hz) to 4Hz
            # 64Hz -> 4Hz = factor of 16
            bvp_resampled = raw_bvp[::16]
            
            # Ensure lengths match (truncate slightly if needed)
            min_len = min(len(raw_eda), len(raw_temp), len(bvp_resampled))
            eda = raw_eda[:min_len]
            temp = raw_temp[:min_len]
            bvp = bvp_resampled[:min_len]
            
            
            # Labels alignment with wrist data
            # WESAD labels are sampled at 700Hz and aligned with chest sensor
            # Wrist EDA/TEMP are at 4Hz, BVP at 64Hz
            # We need to map labels to the 4Hz wrist timeline
            # 
            # The simplest approach: resample labels using scipy
            from scipy.signal import resample
            
            # Target length is min_len (4Hz wrist data length)
            # Source length is labels array
            if len(labels) != min_len:
                # Resample labels to match wrist data length
                labels_downsampled = resample(labels, min_len).astype(int)
            else:
                labels_downsampled = labels[:min_len].astype(int)

            # Normalize per subject
            eda = normalize_signal(eda)
            temp = normalize_signal(temp)
            bvp = normalize_signal(bvp)
            
            # Stack modalities: (Time, Channels) -> (Time, 3)
            # Channels: EDA, TEMP, BVP
            multimodal_data = np.stack([eda, temp, bvp], axis=1)

            # Segment data
            # We only keep segments that have a consistent label (or majority)
            # Target Labels: 1 (Baseline), 2 (Stress), 3 (Amusement)
            # 0 is transient, 4 is meditation. We filter those out unless 'pretrain' mode.
            
            
            count = 0
            label_distribution = {}
            # Create sliding windows
            for i in range(0, len(multimodal_data) - self.window_size + 1, self.step_size):
                segment = multimodal_data[i : i + self.window_size]
                label_segment = labels_downsampled[i : i + self.window_size]
                
                # Determine segment label (majority vote)
                counts = np.bincount(label_segment.astype(int))
                majority_label = np.argmax(counts)
                
                # Track label distribution for debugging
                label_distribution[majority_label] = label_distribution.get(majority_label, 0) + 1
                
                # Filter based on mode
                if self.mode == 'classifier':
                    # For classifier: only use baseline, stress, amusement
                    if majority_label not in [1, 2, 3]:
                        continue
                else:
                    # For pretrain (SSL): use baseline, stress, amusement, and meditation
                    # Exclude transient (0) and ignore labels (5, 6, 7)
                    if majority_label not in [1, 2, 3, 4]:
                         continue

                self.data_segments.append(segment)
                self.labels.append(majority_label)
                count += 1
            
            print(f"  Label distribution: {label_distribution}")
            print(f"  Added {count} segments for {subject_id}")

    def __len__(self):
        return len(self.data_segments)

    def __getitem__(self, idx):
        # Shape: (Time, Channels) -> Transpose to (Channels, Time) for PyTorch Conv1D
        data = torch.FloatTensor(self.data_segments[idx]).transpose(0, 1)
        label = self.labels[idx]
        
        # Remap labels to zero-indexed for classifier mode
        # WESAD labels: 1=baseline, 2=stress, 3=amusement
        # PyTorch expects: 0, 1, 2
        if self.mode == 'classifier':
            label = label - 1  # Map 1→0, 2→1, 3→2
        
        label = torch.LongTensor([label])
        return data, label
