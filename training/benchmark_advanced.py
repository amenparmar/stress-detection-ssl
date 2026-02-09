
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import training functions
from training.train_ssl import train_simclr
from training.train_dann import train_dann, evaluate_dann
from training.train_ultimate import train_ultimate_model
from models.encoder import Encoder
from models.multimodal_encoder import MultiModalFusionEncoder
from models.ssl_head import SSLHead
from training.loss import NTXentLoss
from utils.config import TEMPERATURE, LEARNING_RATE
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np


def train_with_smote(train_loader, test_loader, encoder, num_classes=3, epochs=100, device='cpu'):
    """Train classifier with SMOTE oversampling for class balance."""
    encoder.eval()
    
    # Collect all training data
    all_data, all_labels = [], []
    for batch_data in train_loader:
        if len(batch_data) == 3:
            data, labels, _ = batch_data
        else:
            data, labels = batch_data
        
        # Filter valid labels
        valid_mask = (labels >= 1) & (labels <= 3)
        data, labels = data[valid_mask], labels[valid_mask]
        if len(labels) == 0:
            continue
        labels = labels - 1
        
        with torch.no_grad():
            features = encoder(data.to(device))
        
        all_data.append(features.cpu().numpy())
        all_labels.append(labels.numpy())
    
    X = np.vstack(all_data)
    y = np.concatenate(all_labels)
    
    # Apply SMOTE
    print("    Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train classifier on balanced data
    classifier = nn.Linear(256, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_resampled).to(device)
    y_tensor = torch.LongTensor(y_resampled).to(device)
    
    # Training loop
    classifier.train()
    batch_size = 32
    for epoch in range(epochs):
        indices = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_tensor[batch_indices]
            batch_y = y_tensor[batch_indices]
            
            logits = classifier(batch_X)
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data
            
            valid_mask = (labels >= 1) & (labels <= 3)
            data, labels = data[valid_mask], labels[valid_mask]
            if len(labels) == 0:
                continue
            labels = labels - 1
            
            features = encoder(data.to(device))
            logits = classifier(features)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return classifier, acc, f1


def benchmark_advanced_models(train_loader, test_loader, device='cpu', quick_mode=False):
    """
    Benchmark all advanced stress detection techniques.
    
    Tests:
    1. SMOTE Oversampling
    2. DANN (Domain Adversarial)
    3. Subject-Invariant Loss
    4. Ultimate Performance (All Techniques + Ensemble)
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        device: Training device
        quick_mode: If True, use reduced epochs
    
    Returns:
        List of tuples: (model_name, accuracy, f1_score, training_time)
    """
    results = []
    
    # Epoch configurations
    ssl_epochs = 50 if quick_mode else 500
    train_epochs = 20 if quick_mode else 100
    
    print("\n" + "="*80)
    print("🏆 ADVANCED BENCHMARK: TESTING STATE-OF-THE-ART TECHNIQUES")
    print("="*80)
    print(f"Mode: {'QUICK (reduced epochs)' if quick_mode else 'FULL'}")
    print(f"SSL Epochs: {ssl_epochs}, Training Epochs: {train_epochs}")
    print("="*80 + "\n")
    
    total_start = time.time()
    
    # ============================================================================
    # 1. SMOTE OVERSAMPLING
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 1/4: SMOTE OVERSAMPLING (Class Balance)")
    print("="*80)
    start = time.time()
    
    encoder = Encoder(input_channels=3, output_dim=256).to(device)
    projection_head = SSLHead(input_dim=256, hidden_dim=128, output_dim=64).to(device)
    
    # SSL Pre-training
    criterion = NTXentLoss(batch_size=train_loader.batch_size, temperature=TEMPERATURE, device=device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()), 
        lr=LEARNING_RATE
    )
    print(f"  Pre-training with SimCLR ({ssl_epochs} epochs)...")
    train_simclr(train_loader, encoder, projection_head, optimizer, None, criterion, ssl_epochs, device)
    
    # SMOTE + Classifier Training
    print(f"  Training with SMOTE ({train_epochs} epochs)...")
    classifier, acc, f1 = train_with_smote(
        train_loader, test_loader, encoder, num_classes=3, epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("SMOTE Oversampling", acc, f1, elapsed))
    print(f"✓ SMOTE Complete: {acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 2. DANN (DOMAIN ADVERSARIAL NEURAL NETWORK)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 2/4: DANN (Domain Adversarial Neural Network)")
    print("="*80)
    start = time.time()
    
    encoder_dann = Encoder(input_channels=3, output_dim=256).to(device)
    print(f"  Training DANN model ({train_epochs} epochs)...")
    
    # train_dann returns (encoder, classifier, best_acc)
    _, dann_classifier, _ = train_dann(
        train_loader, test_loader, encoder_dann, 
        num_classes=3, num_subjects=15, epochs=train_epochs, device=device
    )
    
    # Evaluate DANN
    from models.domain_classifier import DomainClassifier
    domain_classifier = DomainClassifier(input_dim=256, num_subjects=15).to(device)
    dann_acc, dann_f1, _ = evaluate_dann(
        test_loader, encoder_dann, dann_classifier, domain_classifier,
        num_classes=3, device=device
    )
    
    elapsed = time.time() - start
    results.append(("DANN (Domain Adversarial)", dann_acc, dann_f1, elapsed))
    print(f"✓ DANN Complete: {dann_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 3. SUBJECT-INVARIANT LOSS
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 3/4: SUBJECT-INVARIANT LOSS (MMD + CORAL + Contrastive)")
    print("="*80)
    start = time.time()
    
    # train_classifier_with_invariant_loss
    from training.train_invariant import train_classifier_with_invariant_loss, evaluate_classifier
    encoder_inv = Encoder(input_channels=3, output_dim=256).to(device)
    print(f"  Training with invariant losses ({train_epochs} epochs)...")
    # Returns (encoder, classifier, best_acc)
    _, inv_classifier, _ = train_classifier_with_invariant_loss(
        train_loader, test_loader, encoder_inv, 
        num_classes=3, epochs=train_epochs, device=device
    )
    # Evaluate
    inv_acc, inv_f1 = evaluate_classifier(
        test_loader, encoder_inv, inv_classifier, device
    )
    elapsed = time.time() - start
    results.append(("Subject-Invariant Loss", inv_acc, inv_f1, elapsed))
    print(f"✓ Invariant Complete: {inv_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 4. ULTIMATE PERFORMANCE
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 4/4: ULTIMATE PERFORMANCE (All Techniques + Ensemble)")
    print("="*80)
    start = time.time()
    
    encoder_ult = Encoder(input_channels=3, output_dim=256).to(device)
    print(f"  Training Ultimate model ({train_epochs} epochs)...")
    # train_ultimate_model returns (encoder, classifier, domain_classifier, trajectory_analyzer, best_acc)
    _, ult_classifier, ult_domain_clf, ult_traj, _ = train_ultimate_model(
        train_loader, test_loader, encoder_ult, 
        num_classes=3, num_subjects=15, epochs=train_epochs, device=device
    )
    # Evaluate Ultimate
    ult_acc, ult_f1, _ = evaluate_ultimate_model(
        test_loader, encoder_ult, ult_classifier, ult_domain_clf,
        ult_traj, device
    )
    
    elapsed = time.time() - start
    results.append(("Ultimate Performance", ult_acc, ult_f1, elapsed))
    print(f"✓ Ultimate Complete: {ult_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    total_elapsed = time.time() - total_start
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 FINAL RANKINGS - ADVANCED TECHNIQUES COMPARED")
    print("="*80)
    print(f"Total Benchmark Time: {total_elapsed/3600:.2f} hours\n")
    
    print(f"{'Rank':<6} {'Model':<45} {'Accuracy':<12} {'F1 Score':<12} {'Time (min)':<12}")
    print("-" * 90)
    
    for rank, (name, acc, f1, t) in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{medal:<6} {name:<45} {acc*100:>6.2f}%     {f1:>6.4f}      {t/60:>6.1f}")
    
    print("=" * 90)
    
    # Summary statistics
    accs = [r[1] for r in results]
    print(f"\nSummary Statistics:")
    print(f"  Best:  {max(accs)*100:.2f}% - {results[0][0]}")
    print(f"  Worst: {min(accs)*100:.2f}% - {results[-1][0]}")
    print(f"  Mean:  {sum(accs)/len(accs)*100:.2f}%")
    print(f"  Range: {(max(accs)-min(accs))*100:.2f}%")
    print("\n" + "="*90)
    print("NOTE: This benchmark tests advanced stress detection techniques:")
    print("  - SMOTE: Class balancing via synthetic oversampling")
    print("  - DANN: Subject-invariant features via adversarial training")
    print("  - Subject-Invariant: MMD + CORAL + Contrastive losses")
    print("  - Ultimate: Combines all techniques with ensemble models")
    print("\nFor basic models, see Option 14 (Baseline, Multi-Modal, Ensemble).")
    print("=" * 90 + "\n")
    
    return results
