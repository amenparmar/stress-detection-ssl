
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from ..models.encoder import Encoder

def train_linear_classifier(train_loader, test_loader, encoder, num_classes, epochs, device):
    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
        
    # Linear Classifier
    # Encoder output dim is 256
    classifier = nn.Linear(256, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0
        
        for data, target in tqdm(train_loader, desc=f"Classifier Epoch {epoch+1}"):
            data, target = data.to(device), target.squeeze().to(device)
            
            with torch.no_grad():
                features = encoder(data)
                
            output = classifier(features)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Evaluation
        acc, f1 = evaluate(test_loader, encoder, classifier, device)
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}, Acc {acc:.4f}, F1 {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            # Save classifier
            torch.save(classifier.state_dict(), 'stress_detection/models/classifier_best.pth')

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
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1
