
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        z_i, z_j: representations of the two augmented versions of the batch.
        Shape: (Batch_Size, Feature_Dim)
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Positive pairs: (i, batch_size + i) and (batch_size + i, i)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negatives = sim[self.mask_correlated_samples(batch_size)].reshape(N, -1)
        
        labels = torch.zeros(N).to(self.device).long()
        logits = torch.cat((positive_samples, negatives), dim=1)
        
        # In this formulation, the positive is always at index 0
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
