
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from ..models.encoder import Encoder

def train_linear_classifier(train_loader, test_loader, encoder, num_classes, epochs, device, finetune_encoder=False):
    """
    Train classifier with optional encoder fine-tuning.
    
    Args:
        finetune_encoder: If True, unfreeze last layers for fine-tuning
    """
    # Detect encoder type
    encoder_type = type(encoder).__name__
    is_multimodal = 'MultiModal' in encoder_type
    
    # Fine-tuning strategy
    if finetune_encoder:
        print(f"🔧 Fine-tuning mode for {encoder_type}...")
        encoder.train()
        
        # Freeze all layers first
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze layers based on encoder type
        if is_multimodal:
            # MultiModalFusionEncoder: Unfreeze fusion layer and final fc
            if hasattr(encoder, 'fusion_layer'):
                for param in encoder.fusion_layer.parameters():
                    param.requires_grad = True
                print("  Unfrozen: fusion_layer")
            if hasattr(encoder, 'fc'):
                for param in encoder.fc.parameters():
                    param.requires_grad = True
                print("  Unfrozen: fc")
        else:
            # Standard Encoder: Unfreeze last 2 conv layers + fc
            if hasattr(encoder, 'layer3'):
                for param in encoder.layer3.parameters():
                    param.requires_grad = True
                print("  Unfrozen: layer3")
            if hasattr(encoder, 'layer4'):
                for param in encoder.layer4.parameters():
                    param.requires_grad = True
                print("  Unfrozen: layer4")
            if hasattr(encoder, 'fc'):
                for param in encoder.fc.parameters():
                    param.requires_grad = True
                print("  Unfrozen: fc")
    else:
        print("Frozen encoder mode (no fine-tuning)")
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
    
    # Calculate class weights to handle imbalance
    print("Calculating class weights from training data...")
    all_labels = []
    for _, target in train_loader:
        all_labels.extend(target.squeeze().numpy())
    
    unique_labels, counts = torch.unique(torch.tensor(all_labels), return_counts=True)
    total_samples = len(all_labels)
    class_weights = total_samples / (len(unique_labels) * counts.float())
    class_weights = class_weights.to(device)
    
    print(f"Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    print(f"Class weights: {class_weights.tolist()}")
        
    # Linear Classifier
    # Encoder output dim is 256
    classifier = nn.Linear(256, num_classes).to(device)
    
    # Optimizer: Include encoder params if fine-tuning
    if finetune_encoder:
        # Collect trainable encoder parameters
        encoder_params = [p for p in encoder.parameters() if p.requires_grad]
        
        if encoder_params:
            optimizer = optim.Adam([
                {'params': encoder_params, 'lr': 1e-5},  # Very low LR for encoder
                {'params': classifier.parameters(), 'lr': 3e-4}  # Higher LR for classifier
            ])
            print(f"Optimizer: Fine-tuning {len(encoder_params)} encoder params (lr=1e-5) + classifier (lr=3e-4)")
        else:
            # Fallback if no encoder params unfrozen
            optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
            print("Optimizer: Classifier only (lr=3e-4) - no encoder params unfrozen")
    else:
        optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
        print("Optimizer: Classifier only (lr=3e-4)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Weighted loss
    
    best_acc = 0.0
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(epochs):
        # Set modes
        if finetune_encoder:
            encoder.train()
        classifier.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Classifier Epoch {epoch+1}")
        for data, target in progress_bar:
            data, target = data.to(device, non_blocking=True), target.squeeze().to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP: Automatic Mixed Precision
            with torch.cuda.amp.autocast():
                if finetune_encoder:
                    features = encoder(data)
                else:
                    with torch.no_grad():
                        features = encoder(data)
                    
                output = classifier(features)
                loss = criterion(output, target)
            
            # Scale loss and step optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Evaluation
        acc, f1 = evaluate(test_loader, encoder, classifier, device)
        avg_loss = total_loss  / len(train_loader)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.4f}, F1 {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), 'stress_detection/models/classifier_best.pth')
            print(f"  ✓ New best accuracy: {best_acc:.4f}")

def evaluate(loader, encoder, classifier, device):
    encoder.eval()
    classifier.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.squeeze().to(device)
            features = encoder(data)
            output = classifier(features)
            preds = torch.argmax(output, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate overall metrics
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    # Calculate per-class accuracy
    from sklearn.metrics import confusion_matrix
    import numpy as np
    cm = confusion_matrix(all_targets, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Print per-class accuracy every 10 epochs
    if hasattr(evaluate, 'call_count'):
        evaluate.call_count += 1
    else:
        evaluate.call_count = 1
    
    if evaluate.call_count % 10 == 0:
        print(f"  Per-class accuracy: Class 0: {per_class_acc[0]:.3f}, Class 1: {per_class_acc[1]:.3f}, Class 2: {per_class_acc[2]:.3f}")
    
    return acc, f1
