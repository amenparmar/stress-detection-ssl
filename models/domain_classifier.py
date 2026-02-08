
import torch
import torch.nn as nn


class DomainClassifier(nn.Module):
    """
    Domain classifier for subject ID prediction.
    
    Used in Domain Adversarial Neural Network (DANN) to enforce 
    subject-invariant feature learning. The encoder is trained to 
    fool this classifier, creating features that cannot distinguish
    between different subjects.
    
    Architecture:
        - Input: Latent features (e.g., 256-dim from encoder)
        - Hidden: 2-layer MLP with dropout
        - Output: Subject ID logits (num_subjects classes)
    """
    
    def __init__(self, input_dim=256, hidden_dim=128, num_subjects=15, dropout=0.3):
        """
        Args:
            input_dim: Dimension of input features from encoder
            hidden_dim: Hidden layer dimension
            num_subjects: Number of subjects (domains) to classify
            dropout: Dropout probability for regularization
        """
        super(DomainClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, num_subjects)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features (Batch, input_dim)
        
        Returns:
            Subject ID logits (Batch, num_subjects)
        """
        return self.classifier(x)


def compute_domain_accuracy(domain_logits, domain_labels):
    """
    Compute domain classification accuracy.
    
    Lower accuracy = better subject-invariance
    (encoder successfully fools domain classifier)
    
    Args:
        domain_logits: Predicted subject logits (Batch, num_subjects)
        domain_labels: True subject IDs (Batch,)
    
    Returns:
        Domain classification accuracy [0, 1]
    """
    _, predicted = torch.max(domain_logits, 1)
    correct = (predicted == domain_labels).sum().item()
    total = domain_labels.size(0)
    return correct / total


def compute_subject_invariance_score(domain_logits, domain_labels):
    """
    Compute subject-invariance score.
    
    Score = 1 - (domain_accuracy / random_chance)
    where random_chance = 1 / num_subjects
    
    Higher score = better subject-invariance
    - Score = 0: Perfect subject discrimination (bad)
    - Score = 0.5: Random chance (neutral)
    - Score = 1: Perfect confusion (excellent subject-invariance)
    
    Args:
        domain_logits: Predicted subject logits (Batch, num_subjects)
        domain_labels: True subject IDs (Batch,)
    
    Returns:
        Subject-invariance score [0, 1]
    """
    domain_acc = compute_domain_accuracy(domain_logits, domain_labels)
    num_subjects = domain_logits.size(1)
    random_chance = 1.0 / num_subjects
    
    # Normalize accuracy relative to random chance
    normalized_acc = min(domain_acc / random_chance, 1.0)
    invariance_score = 1.0 - normalized_acc
    
    return invariance_score
