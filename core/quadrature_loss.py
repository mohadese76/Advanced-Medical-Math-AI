import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import roots_legendre

class GaussianQuadratureLoss(nn.Module):
    def __init__(self, degree=5, sigma=1.0):
        super(GaussianQuadratureLoss, self).__init__()
        # extracting points and weights
        nodes, weights = roots_legendre(degree)
        self.nodes = torch.tensor(nodes, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.sigma = sigma

    def forward(self, logits, targets):
        # Error Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Convert predictions to probabilities (Softmax)       
        probs = F.softmax(logits, dim=1)
        conf = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze()

        #Creative Math Section: Quadrature-Based Weighting        
        quad_weight = torch.zeros_like(conf)
        for i in range(len(self.nodes)):
        # Using Legendre nodes and weights to modify the language level
            quad_weight += self.weights[i] * torch.exp(-((conf - self.nodes[i])**2) / (2 * self.sigma**2))
        
        # combination of totall error
        weighted_loss = ce_loss * quad_weight
        return weighted_loss.mean()

print(" Gaussian Quadrature Loss Function (Thesis-Informed) initialized.")