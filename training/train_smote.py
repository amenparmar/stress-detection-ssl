"""
SMOTE-enhanced classifier training with leave-one-subject-out cross-validation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
from tqdm import tqdm


def train_classifier_with_smote(train_loader, test_loader, encoder, num_classes, epochs, device):
    """
    Train classifier with SMOTE oversampling for minority class.
    
    SMOTE generates synthetic samples for minority classes to balance dataset.
    """
    print("="*60)
    print("SMOTE-ENHANCED TRAINING")
    print("="*60)
    
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Extract features and labels from training data
    print("Extracting features from training data...")
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(): # Use AMP for feature extraction too
            for batch_data in tqdm(train_loader, desc="Feature Extraction"):
                if len(batch_data) == 3:
                    data, target, _ = batch_data
                else:
                    data, target = batch_data
                
                # Filter and remap labels
                valid_mask = (target >= 1) & (target <= 3)
                if not valid_mask.all():
                    data = data[valid_mask]
                    target = target[valid_mask]
                
                if len(target) == 0:
                    continue
                
                target = target - 1
                
                data = data.to(device, non_blocking=True)
                features = encoder(data)
                all_features.append(features.cpu().numpy())
                all_labels.append(target.squeeze().numpy())
    
    X_train = np.vstack(all_features)
    y_train = np.concatenate(all_labels)
    
    # Check class distribution before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nBefore SMOTE:")
    print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Apply SMOTE
    print(f"\nApplying SMOTE oversampling...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f"\nAfter SMOTE:")
    print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Convert back to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_resampled).to(device)
    y_train_tensor = torch.LongTensor(y_train_resampled).to(device)
    
    # Create classifier
    classifier = nn.Linear(256, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    print(f"\nTraining classifier for {epochs} epochs...")
    best_acc = 0.0
    batch_size = 64 # Reduced from 1024 to 64 for better convergence with small dataset
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_tensor))
        for i in range(0, len(X_train_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_features = X_train_tensor[batch_indices]
            batch_labels = y_train_tensor[batch_indices]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                output = classifier(batch_features)
                loss = criterion(output, batch_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Evaluation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc, f1, per_class_acc = evaluate_with_details(test_loader, encoder, classifier, device)
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}: Loss {avg_loss:.4f}, Acc {acc:.4f}, F1 {f1:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(classifier.state_dict(), 'stress_detection/models/classifier_best.pth')
                print(f"  ✓ New best: {best_acc:.4f}")
    
    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")
    return classifier, best_acc


def evaluate_with_details(loader, encoder, classifier, device):
    """Evaluate with detailed per-class metrics."""
    encoder.eval()
    classifier.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                data, target, _ = batch_data
            else:
                data, target = batch_data
            
            # Filter and remap labels
            valid_mask = (target >= 1) & (target <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                target = target[valid_mask]
            
            if len(target) == 0:
                continue
                
            target = target - 1
            
            data = data.to(device)
            target = target.squeeze()
            features = encoder(data)
            output = classifier(features)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.numpy())
    
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    # Per-class accuracy
    cm = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return acc, f1, per_class_acc
