
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.gradient_reversal import GradientReversalLayer, compute_lambda_schedule
from models.domain_classifier import DomainClassifier, compute_domain_accuracy, compute_subject_invariance_score
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_dann(train_loader, test_loader, encoder, num_classes=3, num_subjects=15, 
               epochs=100, device='cpu', lr=3e-4, alpha=0.1):
    """
    Train encoder with Domain Adversarial Neural Network (DANN).
    
    The encoder learns to:
    1. Classify stress states correctly (via stress classifier)
    2. Confuse subject ID prediction (via adversarial domain classifier)
    
    This creates subject-invariant representations that generalize better
    across different individuals.
    
    Args:
        train_loader: Training data loader (returns (data, stress_label, subject_id))
        test_loader: Test data loader
        encoder: Feature encoder (e.g., Encoder or MultiModalFusionEncoder)
        num_classes: Number of stress classes (default: 3)
        num_subjects: Number of subjects/domains (default: 15 for WESAD)
        epochs: Number of training epochs
        device: Training device
        lr: Learning rate
        alpha: Weight for adversarial loss (default: 0.1)
    
    Returns:
        Tuple of (encoder, stress_classifier, best_test_accuracy)
    """
    encoder.train()
    
    # Initialize components
    grl = GradientReversalLayer().to(device)
    domain_classifier = DomainClassifier(input_dim=256, num_subjects=num_subjects).to(device)
    stress_classifier = nn.Linear(256, num_classes).to(device)
    
    # Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=lr, weight_decay=1e-5)
    stress_optimizer = optim.Adam(stress_classifier.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss functions
    stress_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    best_test_acc = 0.0
    
    print("\n" + "="*80)
    print("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN) TRAINING")
    print("="*80)
    print(f"Encoder: {encoder.__class__.__name__}")
    print(f"Num Classes: {num_classes}, Num Subjects: {num_subjects}")
    print(f"Adversarial Weight (alpha): {alpha}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print("="*80 + "\n")
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        # Update GRL lambda (increases from 0 to 1)
        lambda_ = compute_lambda_schedule(epoch, epochs)
        grl.set_lambda(lambda_)
        
        # Training metrics
        total_loss = 0.0
        total_stress_loss = 0.0
        total_domain_loss = 0.0
        domain_acc_sum = 0.0
        stress_correct = 0
        total_samples = 0
        
        encoder.train()
        stress_classifier.train()
        domain_classifier.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack batch
            if len(batch_data) == 3:
                data, stress_labels, subject_ids = batch_data
            else:
                data, stress_labels = batch_data
                subject_ids = torch.zeros(data.size(0), dtype=torch.long)
            
            # Filter and remap labels: remove label 4 (meditation), remap 1,2,3 to 0,1,2
            valid_mask = (stress_labels >= 1) & (stress_labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                stress_labels = stress_labels[valid_mask]
                subject_ids = subject_ids[valid_mask]
            
            if len(stress_labels) == 0:
                continue
            
            stress_labels = stress_labels - 1
            
            data = data.to(device, non_blocking=True)
            stress_labels = stress_labels.to(device, non_blocking=True)
            subject_ids = subject_ids.to(device, non_blocking=True)
            
            batch_size = data.size(0)
            
            encoder_optimizer.zero_grad(set_to_none=True)
            stress_optimizer.zero_grad(set_to_none=True)
            domain_optimizer.zero_grad(set_to_none=True)
            
            # AMP: Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                # Forward pass
                features = encoder(data)
                stress_logits = stress_classifier(features)
                stress_loss = stress_criterion(stress_logits, stress_labels)
                
                reversed_features = grl(features)
                domain_logits = domain_classifier(reversed_features)
                domain_loss = domain_criterion(domain_logits, subject_ids)
                
                total_batch_loss = stress_loss + alpha * domain_loss
            
            # Backward pass with Scaling
            scaler.scale(total_batch_loss).backward()
            scaler.step(encoder_optimizer)
            scaler.step(stress_optimizer)
            scaler.step(domain_optimizer)
            scaler.update()
            
            # Metrics
            total_loss += total_batch_loss.item()
            total_stress_loss += stress_loss.item()
            total_domain_loss += domain_loss.item()
            
            _, stress_pred = torch.max(stress_logits, 1)
            stress_correct += (stress_pred == stress_labels).sum().item()
            
            domain_acc = compute_domain_accuracy(domain_logits, subject_ids)
            domain_acc_sum += domain_acc
            total_samples += batch_size
            
            progress_bar.set_postfix({
                'loss': f"{total_batch_loss.item():.4f}",
                'stress_loss': f"{stress_loss.item():.4f}",
                'domain_acc': f"{domain_acc:.4f}"
            })
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_stress_loss = total_stress_loss / len(train_loader)
        avg_domain_loss = total_domain_loss / len(train_loader)
        train_stress_acc = stress_correct / total_samples
        avg_domain_acc = domain_acc_sum / len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Stress Loss: {avg_stress_loss:.4f} | Train Acc: {train_stress_acc*100:.2f}%")
        print(f"  Domain Loss: {avg_domain_loss:.4f} | Domain Acc: {avg_domain_acc*100:.2f}%")
        print(f"  GRL Lambda: {lambda_:.4f}")
        
        # Evaluation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_acc, test_f1, domain_invariance = evaluate_dann(
                test_loader, encoder, stress_classifier, domain_classifier, 
                num_classes, device
            )
            
            print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Test F1 Score: {test_f1:.4f}")
            print(f"  Subject-Invariance Score: {domain_invariance:.4f} (higher = better)")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('stress_detection/models', exist_ok=True)
                torch.save(encoder.state_dict(), 'stress_detection/models/encoder_dann.pth')
                torch.save(stress_classifier.state_dict(), 'stress_detection/models/classifier_dann.pth')
                print(f"  ✓ New best model saved! Accuracy: {best_test_acc*100:.2f}%")
        
        print("="*80)
    
    print(f"\n{'='*80}")
    print(f"DANN Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return encoder, stress_classifier, best_test_acc


def evaluate_dann(test_loader, encoder, stress_classifier, domain_classifier, 
                  num_classes, device):
    """
    Evaluate DANN model on test set.
    
    Returns:
        Tuple of (stress_accuracy, stress_f1, subject_invariance_score)
    """
    encoder.eval()
    stress_classifier.eval()
    domain_classifier.eval()
    
    all_stress_preds = []
    all_stress_labels = []
    all_domain_logits = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack batch
            if len(batch_data) == 3:
                data, stress_labels, subject_ids = batch_data
            else:
                data, stress_labels = batch_data
                subject_ids = torch.zeros(data.size(0), dtype=torch.long)
            
            data = data.to(device)
            stress_labels = stress_labels.to(device)
            subject_ids = subject_ids.to(device)
            
            # Forward pass
            features = encoder(data)
            stress_logits = stress_classifier(features)
            domain_logits = domain_classifier(features)
            
            _, stress_pred = torch.max(stress_logits, 1)
            
            all_stress_preds.extend(stress_pred.cpu().numpy())
            all_stress_labels.extend(stress_labels.cpu().numpy())
            all_domain_logits.append(domain_logits.cpu())
            all_subject_ids.extend(subject_ids.cpu().numpy())
    
    # Compute metrics
    stress_acc = accuracy_score(all_stress_labels, all_stress_preds)
    stress_f1 = f1_score(all_stress_labels, all_stress_preds, average='macro', zero_division=0)
    
    # Compute subject-invariance score
    all_domain_logits = torch.cat(all_domain_logits, dim=0)
    all_subject_ids = torch.tensor(all_subject_ids)
    invariance_score = compute_subject_invariance_score(all_domain_logits, all_subject_ids)
    
    return stress_acc, stress_f1, invariance_score
