
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

def k_fold_cross_validate(subject_data, encoder_class, num_classes=3, k_folds=5, 
                          epochs=50, batch_size=32, device='cpu'):
    """
    Perform k-fold cross-validation for stress detection model.
    
    Args:
        subject_data: Dictionary of subject data
        encoder_class: Encoder class to instantiate
        num_classes: Number of classification classes
        k_folds: Number of folds for cross-validation
        epochs: Training epochs per fold
        batch_size: Batch size
        device: Device to train on
        
    Returns:
        results: Dictionary with accuracy and F1 scores for each fold
    """
    from stress_detection.data.dataset import WESADDataset
    from stress_detection.training.train_classifier import train_linear_classifier, evaluate
    import torch.nn as nn
    import torch.optim as optim
    
    subjects = list(subject_data.keys())
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    results = {
        'fold_accuracies': [],
        'fold_f1_scores': [],
        'mean_accuracy': 0.0,
        'mean_f1': 0.0,
        'std_accuracy': 0.0,
        'std_f1': 0.0
    }
    
    print(f"Starting {k_folds}-Fold Cross-Validation...")
    print(f"Total subjects: {len(subjects)}")
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(subjects)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'='*50}")
        
        # Split subjects
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        
        # Create datasets
        train_data = {k: subject_data[k] for k in train_subjects}
        test_data = {k: subject_data[k] for k in test_subjects}
        
        train_dataset = WESADDataset(train_data, mode='classifier')
        test_dataset = WESADDataset(test_data, mode='classifier')
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Warning: Empty dataset in fold {fold + 1}. Skipping...")
            continue
        
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=4, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4, persistent_workers=True)
        
        # Initialize model
        encoder = encoder_class(input_channels=3).to(device)
        
        # Load pre-trained weights if available
        import os
        if os.path.exists(os.path.join('stress_detection', 'models', 'encoder_pretrained.pth')):
            encoder.load_state_dict(torch.load(os.path.join('stress_detection', 'models', 'encoder_pretrained.pth')))
        
        # Train and evaluate
        train_linear_classifier(train_loader, test_loader, encoder, num_classes, epochs, device)
        
        # Get final accuracy
        classifier = nn.Linear(256, num_classes).to(device)
        classifier.load_state_dict(torch.load('stress_detection/models/classifier_best.pth'))
        
        acc, f1 = evaluate(test_loader, encoder, classifier, device)
        
        results['fold_accuracies'].append(acc)
        results['fold_f1_scores'].append(f1)
        
        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # Calculate statistics
    if results['fold_accuracies']:
        results['mean_accuracy'] = np.mean(results['fold_accuracies'])
        results['std_accuracy'] = np.std(results['fold_accuracies'])
        results['mean_f1'] = np.mean(results['fold_f1_scores'])
        results['std_f1'] = np.std(results['fold_f1_scores'])
    
    print(f"\n{'='*50}")
    print("Cross-Validation Results:")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Mean F1 Score: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in results['fold_accuracies']]}")
    
    return results
