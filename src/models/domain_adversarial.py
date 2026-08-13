import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and multiply by lambda
        output = grad_output.neg() * ctx.lambda_val
        return output, None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL).
    Passes data unchanged during forward, but reverses and scales gradients during backward.
    """
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

class DomainAdversarialHead(nn.Module):
    """
    Attempts to predict the generator domain.
    Through GRL, forces the fusion representation to become generator-agnostic.
    """
    def __init__(self, in_features: int = 512, num_domains: int = 8, lambda_val: float = 1.0):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_val)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_domains)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rev = self.grl(x)
        return self.classifier(x_rev)
