
import torch
import torch.nn as nn
import os
from tqdm import tqdm

def train_ensemble(train_loader, test_loader, encoder_class, num_models=5, 
                   num_classes=3, epochs=100, device='cpu', save_dir='stress_detection/models/ensemble'):
    """
    Train an ensemble of models with different random seeds.
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        encoder_class: Encoder class to instantiate
        num_models: Number of models in ensemble
        num_classes: Number of classification classes
        epochs: Training epochs per model
        device: Device to train on
        save_dir: Directory to save ensemble models
        
    Returns:
        ensemble_models: List of trained (encoder, classifier) pairs
        ensemble_accuracy: Final ensemble accuracy
    """
    from stress_detection.training.train_classifier import train_linear_classifier, evaluate
    
    os.makedirs(save_dir, exist_ok=True)
    ensemble_models = []
    
    print(f"Training Ensemble of {num_models} Models")
    print(f"{'='*60}")
    
    for i in range(num_models):
        print(f"\n[Model {i+1}/{num_models}]")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility but different for each model
        torch.manual_seed(42 + i)
        
        # Initialize model
        encoder = encoder_class(input_channels=3).to(device)
        
        # Load pre-trained weights if available
        pretrained_path = os.path.join('stress_detection', 'models', 'encoder_pretrained.pth')
        if os.path.exists(pretrained_path):
            encoder.load_state_dict(torch.load(pretrained_path))
            print(f"Loaded pre-trained weights from {pretrained_path}")
        
        # Train classifier
        train_linear_classifier(train_loader, test_loader, encoder, num_classes, epochs, device)
        
        # Load best classifier
        classifier = nn.Linear(256, num_classes).to(device)
        classifier_path = os.path.join('stress_detection', 'models', 'classifier_best.pth')
        if os.path.exists(classifier_path):
            classifier.load_state_dict(torch.load(classifier_path))
        
        # Save ensemble model
        model_path = os.path.join(save_dir, f'model_{i}.pth')
        torch.save({
            'encoder': encoder.state_dict(),
            'classifier': classifier.state_dict()
        }, model_path)
        print(f"Saved model to {model_path}")
        
        ensemble_models.append((encoder, classifier))
    
    # Evaluate ensemble
    print(f"\n{'='*60}")
    print("Evaluating Ensemble Performance")
    print(f"{'='*60}")
    
    ensemble_acc, ensemble_f1 = evaluate_ensemble(test_loader, ensemble_models, device)
    
    print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
    print(f"Ensemble F1 Score: {ensemble_f1:.4f}")
    
    return ensemble_models, ensemble_acc


def evaluate_ensemble(test_loader, ensemble_models, device):
    """
    Evaluate ensemble by averaging predictions from all models.
    """
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    all_targets = []
    all_ensemble_preds = []
    
    for data, target in tqdm(test_loader, desc="Ensemble Eval"):
        data = data.to(device)
        target = target.squeeze().cpu().numpy()
        
        # Get predictions from all models
        model_predictions = []
        for encoder, classifier in ensemble_models:
            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                features = encoder(data)
                output = classifier(features)
                pred = torch.softmax(output, dim=1).cpu().numpy()
                model_predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(model_predictions, axis=0)
        ensemble_pred = np.argmax(ensemble_pred, axis=1)
        
        all_targets.extend(target)
        all_ensemble_preds.extend(ensemble_pred)
    
    accuracy = accuracy_score(all_targets, all_ensemble_preds)
    f1 = f1_score(all_targets, all_ensemble_preds, average='weighted')
    
    return accuracy, f1
