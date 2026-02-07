"""
Leave-One-Subject-Out Cross-Validation for WESAD.

This is the gold standard for cross-subject evaluation.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def leave_one_subject_out_cv(subject_data, encoder_class, num_classes=3, epochs=100, device='cpu'):
    """
    Perform leave-one-subject-out cross-validation.
    
    Train on N-1 subjects, test on 1 subject. Repeat for all subjects.
    
    Args:
        subject_data: Dict of subject IDs to data
        encoder_class: Encoder class to instantiate
        num_classes: Number of classes
        epochs: Training epochs per fold
        device: Device to use
        
    Returns:
        results: Dict with per-subject and average accuracies
    """
    from stress_detection.training.train_smote import train_classifier_with_smote, evaluate_with_details
    from stress_detection.data.dataset import WESADDataset
    
    subject_ids = list(subject_data.keys())
    print(f"="*80)
    print(f"LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION")
    print(f"="*80)
    print(f"Total subjects: {len(subject_ids)}")
    print(f"Each subject will be used as test set once")
    print(f"="*80)
    
    all_results = {}
    all_accuracies = []
    all_f1_scores = []
    
    for i, test_subject in enumerate(subject_ids):
        print(f"\n{'='*80}")
        print(f"FOLD {i+1}/{len(subject_ids)}: Testing on {test_subject}")
        print(f"{'='*80}")
        
        # Split into train and test subjects
        train_subjects = {sid: data for sid, data in subject_data.items() if sid != test_subject}
        test_subjects = {test_subject: subject_data[test_subject]}
        
        print(f"Train subjects: {list(train_subjects.keys())}")
        print(f"Test subject: {test_subject}")
        
        # Create datasets
        train_dataset = WESADDataset(train_subjects, mode='classifier')
        test_dataset = WESADDataset(test_subjects, mode='classifier')
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        # Initialize encoder
        encoder = encoder_class(input_channels=3).to(device)
        
        # Load pre-trained weights if available
        pretrained_path = 'stress_detection/models/encoder_pretrained.pth'
        import os
        if os.path.exists(pretrained_path):
            encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
            print(f"✓ Loaded pre-trained encoder")
        
        # Train with SMOTE
        classifier, train_acc = train_classifier_with_smote(
            train_loader, test_loader, encoder, num_classes, epochs, device
        )
        
        # Final evaluation
        acc, f1, per_class_acc = evaluate_with_details(test_loader, encoder, classifier, device)
        
        print(f"\n{'='*80}")
        print(f"FOLD {i+1} RESULTS - Test Subject: {test_subject}")
        print(f"{'='*80}")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"F1 Score: {f1:.4f}")
        print(f"Per-class accuracy:")
        for c, acc_c in enumerate(per_class_acc):
            print(f"  Class {c}: {acc_c*100:.1f}%")
        
        all_results[test_subject] = {
            'accuracy': acc,
            'f1': f1,
            'per_class_acc': per_class_acc
        }
        all_accuracies.append(acc)
        all_f1_scores.append(f1)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Average Accuracy: {np.mean(all_accuracies)*100:.2f}% ± {np.std(all_accuracies)*100:.2f}%")
    print(f"Average F1 Score: {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}")
    print(f"\nPer-subject results:")
    for subject_id, results in all_results.items():
        print(f"  {subject_id}: {results['accuracy']*100:.2f}% (F1={results['f1']:.4f})")
    
    return all_results, np.mean(all_accuracies), np.mean(all_f1_scores)
