
import itertools
import torch
from stress_detection.models.encoder import Encoder
from stress_detection.utils.cross_validation import k_fold_cross_validate

def grid_search_hyperparameters(subject_data, device='cpu'):
    """
    Grid search for optimal hyperparameters.
    Tests different combinations and reports best configuration.
    """
    
    # Define hyperparameter grid
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'epochs': [50, 100],
    }
    
    best_accuracy = 0
    best_params = None
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} hyperparameter combinations...")
    print(f"{'='*60}")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
        
        # Run cross-validation with these parameters
        cv_results = k_fold_cross_validate(
            subject_data=subject_data,
            encoder_class=Encoder,
            num_classes=3,
            k_folds=3,  # Use 3-fold for faster search
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            device=device
        )
        
        mean_acc = cv_results['mean_accuracy']
        results.append({
            'params': params,
            'accuracy': mean_acc,
            'f1': cv_results['mean_f1']
        })
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_params = params
        
        print(f"Result: Accuracy = {mean_acc:.4f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"{'='*60}")
    print(f"\nBest Configuration:")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Parameters: {best_params}")
    
    print(f"\nAll Results (sorted by accuracy):")
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for r in results_sorted:
        print(f"  Acc: {r['accuracy']:.4f}, F1: {r['f1']:.4f} - {r['params']}")
    
    return best_params, results


def save_best_config(params, filename='stress_detection/best_config.txt'):
    """Save best hyperparameters to file."""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        f.write("# Best Hyperparameters from Grid Search\n")
        f.write(f"# Accuracy: {params.get('accuracy', 'N/A')}\n\n")
        for key, value in params.items():
            f.write(f"{key.upper()} = {value}\n")
    
    print(f"Best configuration saved to {filename}")


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from stress_detection.data.dataset import load_wesad_data
    from stress_detection.utils.config import WESAD_dataset_path
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    subject_data = load_wesad_data(WESAD_dataset_path)
    
    # Run grid search
    best_params, all_results = grid_search_hyperparameters(subject_data, device)
    
    # Save results
    save_best_config(best_params)
