
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) from:
    "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
    
    Forward pass: identity function
    Backward pass: reverses gradient and scales by lambda
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass is identity.
        
        Args:
            x: Input tensor
            lambda_: Reversal strength (typically increases during training)
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass reverses and scales gradient.
        """
        lambda_ = ctx.lambda_
        grad_input = grad_output.neg() * lambda_
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer wrapper.
    
    Usage:
        encoder = Encoder()
        grl = GradientReversalLayer()
        domain_classifier = DomainClassifier()
        
        features = encoder(x)
        reversed_features = grl(features)  # Forward: identity, Backward: reversed
        domain_pred = domain_classifier(reversed_features)
    """
    
    def __init__(self, lambda_=1.0):
        """
        Args:
            lambda_: Initial reversal strength
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def set_lambda(self, lambda_):
        """Update lambda during training (typically increases from 0 to 1)."""
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def compute_lambda_schedule(epoch, max_epochs, lambda_max=1.0, gamma=10.0):
    """
    Compute lambda for GRL using the schedule from Ganin et al. 2016.
    
    Lambda increases from 0 to lambda_max following:
        lambda = (2 / (1 + exp(-gamma * p))) - 1
    where p = epoch / max_epochs
    
    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total number of epochs
        lambda_max: Maximum lambda value (default: 1.0)
        gamma: Schedule steepness (default: 10.0)
    
    Returns:
        Lambda value for current epoch
    """
    p = float(epoch) / float(max_epochs)
    lambda_ = (2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)))) - 1.0
    return lambda_max * lambda_.item()
