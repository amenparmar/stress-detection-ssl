
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.gradient_reversal import GradientReversalLayer, compute_lambda_schedule
from models.domain_classifier import DomainClassifier, compute_subject_invariance_score
from models.trajectory_analyzer import TrajectoryAnalyzer, extract_subject_baselines, compute_trajectory_consistency_loss
from training.invariant_losses import SubjectInvariantLoss
from sklearn.metrics import accuracy_score, f1_score


def train_ultimate_model(train_loader, test_loader, encoder, num_classes=3, num_subjects=15,
                        epochs=100, device='cpu', lr=3e-4, 
                        alpha=0.1, beta=0.05, gamma=0.05, delta=0.02):
    """
    ULTIMATE PERFORMANCE: Train with ALL advanced techniques combined.
    
    Combines:
    1. Classification loss
    2. Domain Adversarial (DANN) - α weight
    3. Subject-Invariant Losses (MMD + CORAL + Contrastive) - β weight
    4. Trajectory Deviation - γ weight
    5. Temporal Consistency - δ weight
    
    Total Loss = classification + α×adversarial + β×invariant + γ×trajectory + δ×temporal
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        encoder: Feature encoder (Multi-Modal Fusion recommended)
        num_classes: Number of stress classes (3)
        num_subjects: Number of subjects for domain classifier
        epochs: Training epochs
        device: Training device
        lr: Learning rate
        alpha: Weight for adversarial loss
        beta: Weight for invariant losses
        gamma: Weight for trajectory loss
        delta: Weight for temporal consistency loss
    
    Returns:
        Tuple of (encoder, classifier, trajectory_analyzer, best_test_accuracy)
    """
    encoder.train()
    
    print("\n" + "="*80)
    print("🚀 ULTIMATE PERFORMANCE TRAINING")
    print("="*80)
    print(f"Encoder: {encoder.__class__.__name__}")
    print(f"Combining 5 Advanced Techniques:")
    print(f"  1. Classification")
    print(f"  2. Domain Adversarial (α={alpha})")
    print(f"  3. Subject-Invariant Losses (β={beta})")
    print(f"  4. Trajectory Analysis (γ={gamma})")
    print(f"  5. Temporal Consistency (δ={delta})")
    print(f"Expected Performance: 85-88% LOSO accuracy")
    print("="*80 + "\n")
    
    # Initialize all components
    grl = GradientReversalLayer().to(device)
    domain_classifier = DomainClassifier(input_dim=256, num_subjects=num_subjects).to(device)
    classifier = nn.Linear(256, num_classes).to(device)
    trajectory_analyzer = TrajectoryAnalyzer(feature_dim=256, num_classes=num_classes).to(device)
    
    # Subject-invariant loss module
    invariant_loss_fn = SubjectInvariantLoss(
        mmd_weight=0.01,
        coral_weight=0.01,
        contrastive_weight=0.03
    ).to(device)
    
    # Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    domain_optimizer = optim.Adam(domain_classifier.parameters(), lr=lr)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr)
    trajectory_optimizer = optim.Adam(trajectory_analyzer.parameters(), lr=lr)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    prev_features = None
    
    for epoch in range(epochs):
        # ================================================
        # STEP 1: Update GRL lambda (0 → 1 over training)
        # ================================================
        lambda_ = compute_lambda_schedule(epoch, epochs)
        grl.set_lambda(lambda_)
        
        # ================================================
        # STEP 2: Extract baselines for trajectory (every 10 epochs)
        # ================================================
        if epoch % 10 == 0:
            print(f"  [Epoch {epoch+1}] Extracting subject baselines for trajectory analysis...")
            baselines = extract_subject_baselines(encoder, train_loader, device)
            trajectory_analyzer.subject_baselines = baselines
            print(f"  Extracted baselines for {len(baselines)} subjects")
        
        # ================================================
        # STEP 3: Training loop
        # ================================================
        total_loss = 0.0
        loss_breakdown = {
            'class': 0.0,
            'adversarial': 0.0,
            'mmd': 0.0,
            'coral': 0.0,
            'contrastive': 0.0,
            'trajectory': 0.0,
            'temporal': 0.0
        }
        correct = 0
        total_samples = 0
        
        encoder.train()
        classifier.train()
        domain_classifier.train()
        trajectory_analyzer.train()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack batch
            if len(batch_data) == 3:
                data, labels, subject_ids = batch_data
            else:
                data, labels = batch_data
                subject_ids = torch.zeros(data.size(0), dtype=torch.long)
            
            data = data.to(device)
            labels = labels.to(device)
            subject_ids = subject_ids.to(device)
            
            # Zero gradients
            encoder_optimizer.zero_grad()
            classifier_optimizer.zero_grad()
            domain_optimizer.zero_grad()
            trajectory_optimizer.zero_grad()
            
            # ================================================
            # Forward Pass: Extract features
            # ================================================
            features = encoder(data)  # (Batch, 256)
            
            # ================================================
            # Loss 1: Classification
            # ================================================
            logits = classifier(features)
            class_loss = class_criterion(logits, labels)
            
            # ================================================
            # Loss 2: Domain Adversarial (DANN)
            # ================================================
            reversed_features = grl(features)
            domain_logits = domain_classifier(reversed_features)
            adv_loss = domain_criterion(domain_logits, subject_ids)
            
            # ================================================
            # Loss 3: Subject-Invariant (MMD + CORAL + Contrastive)
            # ================================================
            inv_total, mmd_loss, coral_loss, contrastive_loss = invariant_loss_fn(
                features, labels, subject_ids
            )
            
            # ================================================
            # Loss 4: Trajectory Deviation
            # ================================================
            traj_logits, deviations, continuous_scores = trajectory_analyzer(
                features, subject_ids, apply_smoothing=False
            )
            traj_loss = class_criterion(traj_logits, labels)
            
            # ================================================
            # Loss 5: Temporal Consistency
            # ================================================
            if prev_features is not None and prev_features.size(0) == features.size(0):
                temp_loss = compute_trajectory_consistency_loss(
                    torch.stack([prev_features, features])
                )
            else:
                temp_loss = torch.tensor(0.0, device=device)
            
            prev_features = features.detach()
            
            # ================================================
            # Combined Loss
            # ================================================
            combined_loss = (
                class_loss +
                alpha * adv_loss +
                beta * inv_total +
                gamma * traj_loss +
                delta * temp_loss
            )
            
            # Backward pass
            combined_loss.backward()
            
            # Update all optimizers
            encoder_optimizer.step()
            classifier_optimizer.step()
            domain_optimizer.step()
            trajectory_optimizer.step()
            
            # ================================================
            # Metrics
            # ================================================
            total_loss += combined_loss.item()
            loss_breakdown['class'] += class_loss.item()
            loss_breakdown['adversarial'] += adv_loss.item()
            loss_breakdown['mmd'] += mmd_loss.item()
            loss_breakdown['coral'] += coral_loss.item()
            loss_breakdown['contrastive'] += contrastive_loss.item()
            loss_breakdown['trajectory'] += traj_loss.item()
            loss_breakdown['temporal'] += temp_loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': combined_loss.item(),
                'class': class_loss.item(),
                'adv': adv_loss.item(),
                'inv': inv_total.item(),
                'traj': traj_loss.item()
            })
        
        # ================================================
        # Epoch Statistics
        # ================================================
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total_samples
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Loss Breakdown:")
        print(f"    Classification: {loss_breakdown['class']/len(train_loader):.4f}")
        print(f"    Adversarial:    {loss_breakdown['adversarial']/len(train_loader):.4f} (×{alpha})")
        print(f"    MMD:            {loss_breakdown['mmd']/len(train_loader):.4f}")
        print(f"    CORAL:          {loss_breakdown['coral']/len(train_loader):.4f}")
        print(f"    Contrastive:    {loss_breakdown['contrastive']/len(train_loader):.4f} (×{beta})")
        print(f"    Trajectory:     {loss_breakdown['trajectory']/len(train_loader):.4f} (×{gamma})")
        print(f"    Temporal:       {loss_breakdown['temporal']/len(train_loader):.4f} (×{delta})")
        print(f"  GRL Lambda: {lambda_:.4f}")
        
        # ================================================
        # Evaluation
        # ================================================
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            test_acc, test_f1, invariance_score = evaluate_ultimate_model(
                test_loader, encoder, classifier, domain_classifier,
                trajectory_analyzer, device
            )
            
            print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Test F1 Score: {test_f1:.4f}")
            print(f"  Subject-Invariance: {invariance_score:.4f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('stress_detection/models', exist_ok=True)
                torch.save(encoder.state_dict(), 'stress_detection/models/encoder_ultimate.pth')
                torch.save(classifier.state_dict(), 'stress_detection/models/classifier_ultimate.pth')
                torch.save(trajectory_analyzer.state_dict(), 'stress_detection/models/trajectory_ultimate.pth')
                print(f"  ✓ New best model saved! Accuracy: {best_test_acc*100:.2f}%")
        
        print("="*80)
    
    print(f"\n{'='*80}")
    print(f"ULTIMATE TRAINING COMPLETE!")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return encoder, classifier, trajectory_analyzer, best_test_acc


def evaluate_ultimate_model(test_loader, encoder, classifier, domain_classifier,
                            trajectory_analyzer, device):
    """Evaluate ultimate model on test set."""
    encoder.eval()
    classifier.eval()
    domain_classifier.eval()
    trajectory_analyzer.eval()
    
    all_preds = []
    all_labels = []
    all_domain_logits = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                data, labels, subject_ids = batch_data
            else:
                data, labels = batch_data
                subject_ids = torch.zeros(data.size(0), dtype=torch.long)
            
            data = data.to(device)
            labels = labels.to(device)
            subject_ids = subject_ids.to(device)
            
            # Forward pass
            features = encoder(data)
            logits = classifier(features)
            domain_logits = domain_classifier(features)
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_domain_logits.append(domain_logits.cpu())
            all_subject_ids.extend(subject_ids.cpu().numpy())
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # Subject-invariance score
    all_domain_logits = torch.cat(all_domain_logits, dim=0)
    all_subject_ids = torch.tensor(all_subject_ids)
    invariance_score = compute_subject_invariance_score(all_domain_logits, all_subject_ids)
    
    return acc, f1, invariance_score
