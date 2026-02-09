
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Direct imports - no relative imports
from training.train_ssl import train_simclr
from models.encoder import Encoder
from models.multimodal_encoder import MultiModalFusionEncoder
from models.ssl_head import SSLHead
from training.loss import NTXentLoss
from utils.config import TEMPERATURE, LEARNING_RATE, EPOCHS
from sklearn.metrics import accuracy_score, f1_score


def train_classifier_simple(train_loader, test_loader, encoder, num_classes=3, epochs=100, device='cpu'):
    """Simple classifier training without relative imports."""
    encoder.eval()  # Freeze encoder
    classifier = nn.Linear(256, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        classifier.train()
        for batch_data in train_loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data
            
            # Filter and remap labels from WESAD to PyTorch format
            # WESAD: 1=baseline, 2=stress, 3=amusement, 4=meditation
            # PyTorch needs: 0, 1, 2 (we only use 3 classes)
            # Remove any samples with label 4 (meditation) or invalid labels
            valid_mask = (labels >= 1) & (labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                labels = labels[valid_mask]
            
            if len(labels) == 0:  # Skip if no valid labels in batch
                continue
            
            # Remap labels: 1→0, 2→1, 3→2
            labels = labels - 1
            
            data, labels = data.to(device), labels.to(device)
            
            with torch.no_grad():
                features = encoder(data)
            
            logits = classifier(features)
            loss = criterion(logits, labels)
            
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
            
            # Filter and remap labels (same as training)
            valid_mask = (labels >= 1) & (labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                labels = labels[valid_mask]
            
            if len(labels) == 0:
                continue
            
            # Remap labels: 1→0, 2→1, 3→2
            labels = labels - 1
            
            data = data.to(device)
            features = encoder(data)
            logits = classifier(features)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return classifier, acc, f1



def benchmark_all_models(train_loader, test_loader, device='cpu', quick_mode=False):
    """
    Run key model configurations and rank them by accuracy.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        device: Training device
        quick_mode: If True, use reduced epochs for faster benchmarking
    
    Returns:
        List of tuples: (model_name, accuracy, f1_score, training_time)
    """
    results = []
    
    # Epoch configurations
    ssl_epochs = 50 if quick_mode else 500
    train_epochs = 20 if quick_mode else 100
    
    print("\n" + "="*80)
    print("ðŸ† BENCHMARK: COMPARING KEY MODEL CONFIGURATIONS")
    print("="*80)
    print(f"Mode: {'QUICK (reduced epochs)' if quick_mode else 'FULL'}")
    print(f"SSL Epochs: {ssl_epochs}, Training Epochs: {train_epochs}")
    print("="*80 + "\n")
    
    total_start = time.time()
    
    # ============================================================================
    # 1. BASELINE (SSL + Standard Classifier)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 1/3: BASELINE (SSL + Standard Classifier)")
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
    
    # Classifier Training
    print(f"  Training classifier ({train_epochs} epochs)...")
    classifier, acc, f1 = train_classifier_simple(
        train_loader, test_loader, encoder, num_classes=3, epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("Baseline (SSL + Classifier)", acc, f1, elapsed))
    print(f"âœ“ Baseline Complete: {acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 2. MULTI-MODAL FUSION
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 2/3: MULTI-MODAL FUSION")
    print("="*80)
    start = time.time()
    
    mm_encoder = MultiModalFusionEncoder(output_dim=256).to(device)
    mm_projection_head = SSLHead(input_dim=256, hidden_dim=128, output_dim=64).to(device)
    
    # SSL Pre-training
    mm_optimizer = torch.optim.Adam(
        list(mm_encoder.parameters()) + list(mm_projection_head.parameters()), 
        lr=LEARNING_RATE
    )
    print(f"  Pre-training multi-modal encoder ({ssl_epochs} epochs)...")
    train_simclr(train_loader, mm_encoder, mm_projection_head, mm_optimizer, None, criterion, ssl_epochs, device)
    
    # Classifier Training
    print(f"  Training classifier ({train_epochs} epochs)...")
    mm_classifier, mm_acc, mm_f1 = train_classifier_simple(
        train_loader, test_loader, mm_encoder, num_classes=3, epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Fusion", mm_acc, mm_f1, elapsed))
    print(f"âœ“ Multi-Modal Complete: {mm_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 3. MULTI-MODAL ENSEMBLE (3 models for benchmark)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 3/3: MULTI-MODAL ENSEMBLE (3 models)")
    print("="*80)
    start = time.time()
    
    ensemble_models = []
    for i in range(3):
        print(f"\n  Training Ensemble Model {i+1}/3...")
        
        ens_encoder = MultiModalFusionEncoder(output_dim=256).to(device)
        ens_projection = SSLHead(input_dim=256, hidden_dim=128, output_dim=64).to(device)
        ens_optimizer = torch.optim.Adam(
            list(ens_encoder.parameters()) + list(ens_projection.parameters()), 
            lr=LEARNING_RATE
        )
        
        print(f"    Pre-training ({ssl_epochs} epochs)...")
        train_simclr(train_loader, ens_encoder, ens_projection, ens_optimizer, None, criterion, ssl_epochs, device)
        
        print(f"    Training classifier ({train_epochs} epochs)...")
        ens_classifier, _, _ = train_classifier_simple(
            train_loader, test_loader, ens_encoder, num_classes=3, epochs=train_epochs, device=device
        )
        
        ensemble_models.append((ens_encoder, ens_classifier))
    
    # Evaluate ensemble
    print("\n  Evaluating ensemble...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data
            
            # Filter and remap labels (same as training)
            valid_mask = (labels >= 1) & (labels <= 3)
            if not valid_mask.all():
                data = data[valid_mask]
                labels = labels[valid_mask]
            
            if len(labels) == 0:
                continue
            
            # Remap labels: 1→0, 2→1, 3→2
            labels = labels - 1
            
            data = data.to(device)
            
            # Get predictions from all models
            model_preds = []
            for enc, clf in ensemble_models:
                enc.eval()
                clf.eval()
                features = enc(data)
                logits = clf(features)
                _, preds = torch.max(logits, 1)
                model_preds.append(preds.cpu())
            
            # Majority voting
            model_preds = torch.stack(model_preds)  # (num_models, batch_size)
            ensemble_pred = torch.mode(model_preds, dim=0)[0]
            
            all_preds.extend(ensemble_pred.numpy())
            all_labels.extend(labels.numpy())
    
    ensemble_acc = accuracy_score(all_labels, all_preds)
    ensemble_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Ensemble (3)", ensemble_acc, ensemble_f1, elapsed))
    print(f"âœ“ Ensemble Complete: {ensemble_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    total_elapsed = time.time() - total_start
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("ðŸ† FINAL RANKINGS - KEY MODELS COMPARED")
    print("="*80)
    print(f"Total Benchmark Time: {total_elapsed/3600:.2f} hours\n")
    
    print(f"{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'F1 Score':<12} {'Time (min)':<12}")
    print("-" * 85)
    
    for rank, (name, acc, f1, t) in enumerate(results, 1):
        medal = "ðŸ¥‡" if rank == 1 else "ðŸ¥ˆ" if rank == 2 else "ðŸ¥‰"
        print(f"{medal:<6} {name:<40} {acc*100:>6.2f}%     {f1:>6.4f}      {t/60:>6.1f}")
    
    print("=" * 85)
    
    # Summary statistics
    accs = [r[1] for r in results]
    print(f"\nSummary Statistics:")
    print(f"  Best:  {max(accs)*100:.2f}% - {results[0][0]}")
    print(f"  Worst: {min(accs)*100:.2f}% - {results[-1][0]}")
    print(f"  Mean:  {sum(accs)/len(accs)*100:.2f}%")
    print(f"  Range: {(max(accs)-min(accs))*100:.2f}%")
    print("\n" + "="*85)
    print("NOTE: This benchmark compares 3 key configurations:")
    print("  1. Baseline (standard encoder)")
    print("  2. Multi-Modal Fusion (separate encoders per modality)")
    print("  3. Ensemble (3 multi-modal models with majority voting)")
    print("\nFor advanced techniques (DANN, SMOTE, Trajectory, etc.), use Options 7-13.")
    print("=" * 85 + "\n")
    
    return results


