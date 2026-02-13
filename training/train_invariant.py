
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from training.invariant_losses import SubjectInvariantLoss
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_classifier_with_invariant_loss(train_loader, test_loader, encoder, num_classes=3,
                                        epochs=100, device='cpu', lr=3e-4,
                                        mmd_weight=0.01, coral_weight=0.01, contrastive_weight=0.1):
    """
    Train classifier with subject-invariant losses.
    
    Combines:
    - Standard classification loss
    - MMD loss (distribution matching)
    - CORAL loss (covariance alignment)
    - Contrastive subject loss (pull same-stress different-subject pairs together)
    
    Args:
        train_loader: Training data loader (returns (data, label, subject_id))
        test_loader: Test data loader
        encoder: Feature encoder
        num_classes: Number of stress classes (default: 3)
        epochs: Number of training epochs
        device: Training device
        lr: Learning rate
        mmd_weight: Weight for MMD loss
        coral_weight: Weight for CORAL loss
        contrastive_weight: Weight for contrastive subject loss
    
    Returns:
        Tuple of (encoder, classifier, best_test_accuracy)
    """
    encoder.train()
    
    # Initialize classifier
    classifier = nn.Linear(256, num_classes).to(device)
    
    # Initialize subject-invariant loss
    invariant_loss_fn = SubjectInvariantLoss(
        mmd_weight=mmd_weight,
        coral_weight=coral_weight,
        contrastive_weight=contrastive_weight
    ).to(device)
    
    # Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)
    
    # Classification loss
    class_criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    best_test_acc = 0.0
    
    print("\n" + "="*80)
    print("SUBJECT-INVARIANT LOSS TRAINING")
    print("="*80)
    print(f"Encoder: {encoder.__class__.__name__}")
    print(f"Num Classes: {num_classes}")
    print(f"MMD Weight: {mmd_weight}, CORAL Weight: {coral_weight}, Contrastive Weight: {contrastive_weight}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print("="*80 + "\n")
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_class_loss = 0.0
        total_mmd = 0.0
        total_coral = 0.0
        total_contrastive = 0.0
        correct = 0
        total_samples = 0
        
        encoder.train()
        classifier.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, subject_ids = batch_data
            else:
                data, labels = batch_data
                subject_ids = torch.arange(data.size(0))
            
            # Filter and remap labels: remove label 4 (meditation), remap 1,2,3 to 0,1,2
            valid_mask = (labels >= 1) & (labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                labels = labels[valid_mask]
                subject_ids = subject_ids[valid_mask]
            
            if len(labels) == 0:
                continue
            
            labels = labels - 1
            
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            subject_ids = subject_ids.to(device, non_blocking=True)
            
            encoder_optimizer.zero_grad(set_to_none=True)
            classifier_optimizer.zero_grad(set_to_none=True)
            
            # AMP: Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                # Forward pass
                features = encoder(data)
                logits = classifier(features)
                
                # Classification loss
                class_loss = class_criterion(logits, labels)
                
                # Subject-invariant losses
                inv_loss, mmd_loss, coral_loss, contrastive_loss = invariant_loss_fn(
                    features, labels, subject_ids
                )
                
                # Total loss
                loss = class_loss + inv_loss
            
            # Backward pass with Scaling
            scaler.scale(loss).backward()
            scaler.step(encoder_optimizer)
            scaler.step(classifier_optimizer)
            scaler.update()
            
            # Metrics
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_mmd += mmd_loss.item()
            total_coral += coral_loss.item()
            total_contrastive += contrastive_loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'class': f"{class_loss.item():.4f}",
                'inv': f"{inv_loss.item():.4f}"
            })
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_class = total_class_loss / len(train_loader)
        avg_mmd = total_mmd / len(train_loader)
        avg_coral = total_coral / len(train_loader)
        avg_contrastive = total_contrastive / len(train_loader)
        train_acc = correct / total_samples
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Class Loss: {avg_class:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  MMD Loss: {avg_mmd:.4f}")
        print(f"  CORAL Loss: {avg_coral:.4f}")
        print(f"  Contrastive Loss: {avg_contrastive:.4f}")
        
        # Evaluation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_acc, test_f1 = evaluate_classifier(
                test_loader, encoder, classifier, device
            )
            
            print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Test F1 Score: {test_f1:.4f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('stress_detection/models', exist_ok=True)
                torch.save(encoder.state_dict(), 'stress_detection/models/encoder_invariant.pth')
                torch.save(classifier.state_dict(), 'stress_detection/models/classifier_invariant.pth')
                print(f"  ✓ New best model saved! Accuracy: {best_test_acc*100:.2f}%")
        
        print("="*80)
    
    print(f"\n{'='*80}")
    print(f"Invariant Loss Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return encoder, classifier, best_test_acc


def evaluate_classifier(test_loader, encoder, classifier, device):
    """
    Evaluate classifier on test set.
    
    Returns:
        Tuple of (accuracy, f1_score)
    """
    encoder.eval()
    classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data
            
            # Filter and remap labels
            valid_mask = (labels >= 1) & (labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                labels = labels[valid_mask]
            
            if len(labels) == 0:
                continue
            
            labels = labels - 1  # Remap: 1→0, 2→1, 3→2
            
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = encoder(data)
            logits = classifier(features)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return acc, f1
