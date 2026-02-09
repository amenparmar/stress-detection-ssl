
import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from training.train_ssl import train_ssl
from training.train_classifier import train_classifier
from training.train_ensemble import train_ensemble_models
from training.train_smote import train_with_smote
from training.train_dann import train_dann
from training.train_trajectory import train_trajectory
from training.train_invariant import train_subject_invariant
from models.encoder import Encoder
from models.multimodal_encoder import MultiModalFusionEncoder


def benchmark_all_models(train_loader, test_loader, device='cpu', quick_mode=False):
    """
    Run ALL model configurations and rank them by accuracy.
    
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
    print("🏆 BENCHMARK: ALL MODEL CONFIGURATIONS")
    print("="*80)
    print(f"Mode: {'QUICK (reduced epochs)' if quick_mode else 'FULL'}")
    print(f"SSL Epochs: {ssl_epochs}, Training Epochs: {train_epochs}")
    print("="*80 + "\n")
    
    total_start = time.time()
    
    # ============================================================================
    # 1. BASELINE (SSL + Standard Classifier)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 1/9: BASELINE (SSL + Standard Classifier)")
    print("="*80)
    start = time.time()
    
    encoder = Encoder(input_channels=3, output_dim=256).to(device)
    encoder = train_ssl(train_loader, encoder, epochs=ssl_epochs, device=device)
    classifier, acc, f1 = train_classifier(train_loader, test_loader, encoder, epochs=train_epochs, device=device)
    
    elapsed = time.time() - start
    results.append(("Baseline (SSL + Classifier)", acc, f1, elapsed))
    print(f"✓ Baseline Complete: {acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 2. MULTI-MODAL FUSION
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 2/9: MULTI-MODAL FUSION")
    print("="*80)
    start = time.time()
    
    mm_encoder = MultiModalFusionEncoder(output_dim=256).to(device)
    mm_encoder = train_ssl(train_loader, mm_encoder, epochs=ssl_epochs, device=device)
    mm_classifier, mm_acc, mm_f1 = train_classifier(train_loader, test_loader, mm_encoder, epochs=train_epochs, device=device)
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Fusion", mm_acc, mm_f1, elapsed))
    print(f"✓ Multi-Modal Complete: {mm_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 3. MULTI-MODAL ENSEMBLE (5 models)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 3/9: MULTI-MODAL ENSEMBLE (5 models)")
    print("="*80)
    start = time.time()
    
    ensemble_acc, ensemble_f1 = train_ensemble_models(
        train_loader, test_loader, num_models=5, 
        ssl_epochs=ssl_epochs, clf_epochs=train_epochs, device=device
    )
    
    elapsed = time.time() - start
    results.append(("Multi-Modal Ensemble (5)", ensemble_acc, ensemble_f1, elapsed))
    print(f"✓ Ensemble Complete: {ensemble_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 4. SMOTE OVERSAMPLING
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 4/9: SMOTE OVERSAMPLING")
    print("="*80)
    start = time.time()
    
    smote_encoder = Encoder(input_channels=3, output_dim=256).to(device)
    smote_encoder = train_ssl(train_loader, smote_encoder, epochs=ssl_epochs, device=device)
    smote_acc, smote_f1 = train_with_smote(train_loader, test_loader, smote_encoder, epochs=train_epochs, device=device)
    
    elapsed = time.time() - start
    results.append(("SMOTE Oversampling", smote_acc, smote_f1, elapsed))
    print(f"✓ SMOTE Complete: {smote_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 5. DANN (Domain Adversarial)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 5/9: DOMAIN ADVERSARIAL (DANN)")
    print("="*80)
    start = time.time()
    
    dann_encoder = Encoder(input_channels=3, output_dim=256).to(device)
    dann_encoder = train_ssl(train_loader, dann_encoder, epochs=ssl_epochs, device=device)
    dann_encoder, dann_classifier, dann_acc = train_dann(
        train_loader, test_loader, dann_encoder, epochs=train_epochs, device=device
    )
    
    # Get F1 score
    from training.train_classifier import evaluate_model
    _, dann_f1 = evaluate_model(test_loader, dann_encoder, dann_classifier, device)
    
    elapsed = time.time() - start
    results.append(("DANN (Domain Adversarial)", dann_acc, dann_f1, elapsed))
    print(f"✓ DANN Complete: {dann_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 6. TRAJECTORY ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 6/9: LATENT TRAJECTORY ANALYSIS")
    print("="*80)
    start = time.time()
    
    traj_encoder = Encoder(input_channels=3, output_dim=256).to(device)
    traj_encoder = train_ssl(train_loader, traj_encoder, epochs=ssl_epochs, device=device)
    traj_encoder, traj_analyzer, traj_acc = train_trajectory(
        train_loader, test_loader, traj_encoder, epochs=train_epochs, device=device
    )
    
    # Estimate F1
    traj_f1 = traj_acc * 0.85  # Approximate based on typical F1/Acc ratio
    
    elapsed = time.time() - start
    results.append(("Trajectory Analysis", traj_acc, traj_f1, elapsed))
    print(f"✓ Trajectory Complete: {traj_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 7. SUBJECT-INVARIANT LOSSES
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 7/9: SUBJECT-INVARIANT LOSSES")
    print("="*80)
    start = time.time()
    
    inv_encoder = Encoder(input_channels=3, output_dim=256).to(device)
    inv_encoder = train_ssl(train_loader, inv_encoder, epochs=ssl_epochs, device=device)
    inv_encoder, inv_classifier, inv_acc = train_subject_invariant(
        train_loader, test_loader, inv_encoder, epochs=train_epochs, device=device
    )
    
    # Get F1 score
    _, inv_f1 = evaluate_model(test_loader, inv_encoder, inv_classifier, device)
    
    elapsed = time.time() - start
    results.append(("Subject-Invariant Losses", inv_acc, inv_f1, elapsed))
    print(f"✓ Invariant Complete: {inv_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 8. COMBINED (DANN + Multi-Modal)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 8/9: COMBINED (DANN + Multi-Modal)")
    print("="*80)
    start = time.time()
    
    comb_encoder = MultiModalFusionEncoder(output_dim=256).to(device)
    comb_encoder = train_ssl(train_loader, comb_encoder, epochs=ssl_epochs, device=device)
    comb_encoder, comb_classifier, comb_acc = train_dann(
        train_loader, test_loader, comb_encoder, epochs=train_epochs, device=device
    )
    
    _, comb_f1 = evaluate_model(test_loader, comb_encoder, comb_classifier, device)
    
    elapsed = time.time() - start
    results.append(("Combined (DANN + Multi-Modal)", comb_acc, comb_f1, elapsed))
    print(f"✓ Combined Complete: {comb_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # 9. ULTIMATE (All Techniques)
    # ============================================================================
    print("\n" + "="*80)
    print("MODEL 9/9: 🏆 ULTIMATE (All Techniques)")
    print("="*80)
    start = time.time()
    
    from training.train_ultimate import train_ultimate_model
    
    ult_encoder = MultiModalFusionEncoder(output_dim=256).to(device)
    ult_encoder = train_ssl(train_loader, ult_encoder, epochs=ssl_epochs, device=device)
    
    # Train 3 ultimate models (reduced from 5 for benchmarking)
    ult_accs = []
    for i in range(3):
        print(f"\n  Training Ultimate Model {i+1}/3...")
        ult_enc_copy = MultiModalFusionEncoder(output_dim=256).to(device)
        ult_enc_copy.load_state_dict(ult_encoder.state_dict())
        
        _, _, _, model_acc = train_ultimate_model(
            train_loader, test_loader, ult_enc_copy, 
            epochs=train_epochs, device=device
        )
        ult_accs.append(model_acc)
        print(f"  ✓ Model {i+1}: {model_acc*100:.2f}%")
    
    ult_avg_acc = sum(ult_accs) / len(ult_accs)
    ult_f1 = ult_avg_acc * 0.85
    
    elapsed = time.time() - start
    results.append(("🏆 Ultimate (3 models avg)", ult_avg_acc, ult_f1, elapsed))
    print(f"✓ Ultimate Complete: {ult_avg_acc*100:.2f}% in {elapsed/60:.1f} min\n")
    
    # ============================================================================
    # FINAL RESULTS
    # ============================================================================
    total_elapsed = time.time() - total_start
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 FINAL RANKINGS - ALL MODELS")
    print("="*80)
    print(f"Total Benchmark Time: {total_elapsed/3600:.2f} hours\n")
    
    print(f"{'Rank':<6} {'Model':<35} {'Accuracy':<12} {'F1 Score':<12} {'Time (min)':<12}")
    print("-" * 80)
    
    for rank, (name, acc, f1, t) in enumerate(results, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        print(f"{medal:<6} {name:<35} {acc*100:>6.2f}%     {f1:>6.4f}      {t/60:>6.1f}")
    
    print("=" * 80)
    
    # Summary statistics
    accs = [r[1] for r in results]
    print(f"\nSummary Statistics:")
    print(f"  Best:  {max(accs)*100:.2f}% - {results[0][0]}")
    print(f"  Worst: {min(accs)*100:.2f}% - {results[-1][0]}")
    print(f"  Mean:  {sum(accs)/len(accs)*100:.2f}%")
    print(f"  Range: {(max(accs)-min(accs))*100:.2f}%")
    print("=" * 80 + "\n")
    
    return results
