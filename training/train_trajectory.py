
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.trajectory_analyzer import TrajectoryAnalyzer, extract_subject_baselines, compute_trajectory_consistency_loss
from sklearn.metrics import accuracy_score, f1_score


def train_trajectory_model(train_loader, test_loader, encoder, num_classes=3, 
                          epochs=100, device='cpu', lr=3e-4, beta=0.1):
    """
    Train encoder with trajectory-based classification.
    
    Optimizes for:
    1. Deviation-based classification (relative to baseline)
    2. Temporal consistency (smooth trajectories)
    
    Args:
        train_loader: Training data loader (returns (data, label, subject_id))
        test_loader: Test data loader
        encoder: Feature encoder
        num_classes: Number of stress classes (default: 3)
        epochs: Number of training epochs
        device: Training device
        lr: Learning rate
        beta: Weight for temporal consistency loss
    
    Returns:
        Tuple of (encoder, trajectory_analyzer, best_test_accuracy)
    """
    encoder.train()
    
    # Initialize trajectory analyzer
    trajectory_analyzer = TrajectoryAnalyzer(
        feature_dim=256, 
        num_classes=num_classes,
        smoothing_window=5
    ).to(device)
    
    # Optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    trajectory_optimizer = optim.Adam(trajectory_analyzer.parameters(), lr=lr, weight_decay=1e-5)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Metrics tracking
    best_test_acc = 0.0
    
    print("\n" + "="*80)
    print("LATENT TRAJECTORY ANALYSIS TRAINING")
    print("="*80)
    print(f"Encoder: {encoder.__class__.__name__}")
    print(f"Num Classes: {num_classes}")
    print(f"Temporal Consistency Weight (beta): {beta}")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")
    print("="*80 + "\n")
    
    for epoch in range(epochs):
        # ================================================
        # Step 1: Extract baselines from current encoder
        # ================================================
        if epoch % 10 == 0:
            print("Extracting subject baselines...")
            baselines = extract_subject_baselines(encoder, train_loader, device)
            trajectory_analyzer.subject_baselines = baselines
            print(f"Extracted baselines for {len(baselines)} subjects")
        
        # ================================================
        # Step 2: Training loop
        # ================================================
        total_loss = 0.0
        total_class_loss = 0.0
        total_consistency_loss = 0.0
        correct = 0
        total_samples = 0
        
        encoder.train()
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
            
            # Forward pass: extract features
            features = encoder(data)  # (Batch, 256)
            
            # Trajectory-based prediction
            logits, deviations, continuous_scores = trajectory_analyzer(
                features, subject_ids, apply_smoothing=False
            )
            
            # Classification loss
            class_loss = criterion(logits, labels)
            
            # Temporal consistency loss (if batch has sequential structure)
            # For now, use simple feature smoothness
            consistency_loss = torch.tensor(0.0, device=device)
            if batch_idx > 0 and hasattr(train_trajectory_model, 'prev_features'):
                consistency_loss = compute_trajectory_consistency_loss(
                    torch.stack([train_trajectory_model.prev_features, features])
                )
            train_trajectory_model.prev_features = features.detach()
            
            # Total loss
            loss = class_loss + beta * consistency_loss
            
            # Backward pass
            encoder_optimizer.zero_grad()
            trajectory_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            trajectory_optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_consistency_loss += consistency_loss.item()
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'class_loss': class_loss.item(),
                'consistency': consistency_loss.item()
            })
        
        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        avg_consistency = total_consistency_loss / len(train_loader)
        train_acc = correct / total_samples
        
        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Classification Loss: {avg_class_loss:.4f}")
        print(f"  Consistency Loss: {avg_consistency:.4f}")
        print(f"  Train Accuracy: {train_acc*100:.2f}%")
        
        # Evaluation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Update baselines before evaluation
            baselines = extract_subject_baselines(encoder, train_loader, device)
            trajectory_analyzer.subject_baselines = baselines
            
            test_acc, test_f1, avg_deviation = evaluate_trajectory_model(
                test_loader, encoder, trajectory_analyzer, num_classes, device
            )
            
            print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Test F1 Score: {test_f1:.4f}")
            print(f"  Average Deviation: {avg_deviation:.4f}")
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('stress_detection/models', exist_ok=True)
                torch.save(encoder.state_dict(), 'stress_detection/models/encoder_trajectory.pth')
                torch.save(trajectory_analyzer.state_dict(), 'stress_detection/models/trajectory_analyzer.pth')
                # Also save baselines
                torch.save(baselines, 'stress_detection/models/trajectory_baselines.pth')
                print(f"  ✓ New best model saved! Accuracy: {best_test_acc*100:.2f}%")
        
        print("="*80)
    
    print(f"\n{'='*80}")
    print(f"Trajectory Training Complete!")
    print(f"Best Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return encoder, trajectory_analyzer, best_test_acc


def evaluate_trajectory_model(test_loader, encoder, trajectory_analyzer, num_classes, device):
    """
    Evaluate trajectory-based model on test set.
    
    Returns:
        Tuple of (accuracy, f1_score, avg_deviation)
    """
    encoder.eval()
    trajectory_analyzer.eval()
    
    all_preds = []
    all_labels = []
    all_deviations = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack batch
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
            logits, deviations, continuous_scores = trajectory_analyzer(
                features, subject_ids, apply_smoothing=True
            )
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_deviations.extend(deviations.cpu().numpy())
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_deviation = np.mean(all_deviations)
    
    return acc, f1, avg_deviation
