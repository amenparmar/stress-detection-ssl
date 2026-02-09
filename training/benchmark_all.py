
import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import actual training functions from main.py's pattern
from training.train_ssl import train_simclr
from training.train_classifier import train_linear_classifier, evaluate_model
from models.encoder import Encoder
from models.multimodal_encoder import MultiModalFusionEncoder
from models.projection_head import SSLHead
from utils.loss import NTXentLoss
from utils.config import TEMPERATURE, LEARNING_RATE


def benchmark_all_models(train_loader, test_loader, device='cpu', quick_mode=False):
    """
    Run ALL model configurations and rank them by accuracy.
    
    NOTE: This is a simplified benchmark that runs key configurations.
    For full benchmark including all advanced techniques, each would need
    their dedicated implementation from main.py.
    
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
    print("🏆 BENCHMARK: COMPARING KEY MODEL CONFIGURATIONS")
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
    classifier, acc, f1 = train_linear_classifier(
        train_loader, test_loader, encoder, num_classes=3, epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("Baseline (SSL + Classifier)", acc, f1, elapsed))
    print(f"✓ Baseline Complete: {acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
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
    mm_classifier, mm_acc, mm_f1 = train_linear_classifier(
        train_loader, test_loader, mm_encoder, num_classes=3, epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Fusion", mm_acc, mm_f1, elapsed))
    print(f"✓ Multi-Modal Complete: {mm_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
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
        ens_classifier, _, _ = train_linear_classifier(
            train_loader, test_loader, ens_encoder, num_classes=3, epochs=train_epochs, device=device
        )
        
        ensemble_models.append((ens_encoder, ens_classifier))
    
    # Evaluate ensemble
    print("\n  Evaluating ensemble...")
    ens_encoder, ens_classifier = ensemble_models[0]  # Use first model's structure
    ens_encoder.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) == 3:
                data, labels, _ = batch_data
            else:
                data, labels = batch_data
            
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
    
    from sklearn.metrics import accuracy_score, f1_score
    ensemble_acc = accuracy_score(all_labels, all_preds)
    ensemble_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Ensemble (3)", ensemble_acc, ensemble_f1, elapsed))
    print(f"✓ Ensemble Complete: {ensemble_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    total_elapsed = time.time() - total_start
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 FINAL RANKINGS - KEY MODELS COMPARED")
    print("="*80)
    print(f"Total Benchmark Time: {total_elapsed/3600:.2f} hours\n")
    
    print(f"{'Rank':<6} {'Model':<40} {'Accuracy':<12} {'F1 Score':<12} {'Time (min)':<12}")
    print("-" * 85)
    
    for rank, (name, acc, f1, t) in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
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
    print("NOTE: This is a simplified benchmark comparing 3 key configurations.")
    print("For full testing of all 9 advanced techniques, each requires dedicated")
    print("implementation from the corresponding option in run.bat (Options 4-13).")
    print("=" * 85 + "\n")
    
    return results
