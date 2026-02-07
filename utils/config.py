
import os

# Data Paths
# TODO: Update this path to where your WESAD dataset is located
WESAD_dataset_path = r'C:\Users\amenp\Downloads\WESAD' 

# Signal Parameters
FS_dict = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
WINDOW_SIZE = 120 # seconds - Increased from 60 for longer context
WINDOW_SHIFT = 0.25 # seconds (overlap)

# Training Hyperparameters
BATCH_SIZE = 32  # Reduced from 128 for better convergence with limited data
LEARNING_RATE = 3e-4
EPOCHS = 500  # Increased from 300 for even better SSL convergence
TEMPERATURE = 0.1 # For NT-Xent loss
PROJECTION_DIM = 128
ENCODER_OUTPUT_DIM = 256

# Labels
LABEL_MAP = {0: 'transient', 1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation', 5: 'ignore', 6: 'ignore', 7: 'ignore'}
TARGET_LABELS = [1, 2, 3] # Baseline, Stress, Amusement

