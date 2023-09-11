"""Sort a list of integers using MultiheadAttention with residuals and LayerNorms.

Important ideas:
    - MultiheadAttention: can hopefully be used for sorting
    - LeakyReLU: with a large negative slope, s.t. the negative values are also used
    - AdaptedLayerNorm: A norm that normalizes the input to a given mean and variance; that of the input list
                        this is used to normalize the output of the MultiheadAttention to that of the input list,
                        helping the network to learn to sort the list
"""

import torch
from torch import nn


def mean_fn(x: torch.Tensor) -> torch.Tensor:
    """A convenience function wrapping torch.mean."""
    return torch.mean(x, dim=1, keepdim=True)


def var_fn(x: torch.Tensor) -> torch.Tensor:
    """A convenience function wrapping torch.var."""
    return torch.var(x, dim=1, keepdim=True)


def allclose(x: torch.Tensor, y: torch.Tensor) -> bool:
    """A convenience function wrapping torch.allclose."""
    return torch.allclose(x, y, rtol=1e-3, atol=1e-4)


class AdaptedLayerNorm(nn.Module):
    """
    Adapted Layer Normalization.
    
    Arguments:
        - mean: The mean of the input tensor
        - variance: The variance of the input tensor

    Returns:
        - The normalized input tensor
    """

    def __init__(self, mean: torch.Tensor, variance: torch.Tensor):
        super().__init__()
        self.mean = mean
        self.variance = variance

    def forward(self, x: torch.Tensor):
        # Normalize the mean of variance of x to 0 and 1, respectively
        x = (x - mean_fn(x)) / torch.sqrt(var_fn(x))

        # Scale and shift x to have the same mean and variance as the input tensor
        x = x * torch.sqrt(self.variance) + self.mean

        return x
    

def test_adaptedlayernorm() -> None:
    """Test the AdaptedLayerNorm class."""
    # Use a batch size of 16 and a embed_dim of 8
    batch_size = 16
    embed_dim = 8

    # Create a test tensor
    test_tensor = torch.randn(batch_size, embed_dim)

    # Get the batch-wise mean and variance of the tensor
    mean = mean_fn(test_tensor)
    variance = var_fn(test_tensor)

    # Create an AdaptedLayerNorm object
    adaptedlayernorm = AdaptedLayerNorm(mean, variance)

    # Create a second test tensor
    test_tensor2 = torch.randn(batch_size, embed_dim)

    # Normalize the second test tensor
    test_tensor2 = adaptedlayernorm(test_tensor2)

    # Check if the mean and variance of the second test tensor are equal to those of the first test tensor
    assert torch.allclose(mean_fn(test_tensor2), mean)
    assert torch.allclose(var_fn(test_tensor2), variance)

    print("AdaptedLayerNorm test passed.")


class MultiheadSelfAttention(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, batch_first=True)

    def forward(self, x):
        # x is of shape (batch_size, embed_dim),
        # but MultiheadAttention expects (batch_size, 1, embed_dim)
        x = x.reshape(x.shape[0], 1, x.shape[1])

        # Forward pass of self-attention, with batch_first=True
        x, _ = super().forward(x, x, x)

        # Shape back
        x = x.reshape(x.shape[0], x.shape[2])
        return x
    

class AttentionBlock(nn.Module):
    """
    An attention block that uses MultiheadAttention with residuals and LayerNorms.

    Arguments
    ---------

    negative_slope: The negative slope of the LeakyReLU activation function.
                    Type: float
                    Default: 0.5
    
    *args: Arguments for MultiheadAttention
    **kwargs: Keyword arguments for MultiheadAttention
    """

    def __init__(
            self, mean: torch.Tensor, variance: torch.Tensor, negative_slope: float = 0.5, *args, **kwargs
    ) -> None:
        super().__init__()
        self.multiheadattention = MultiheadSelfAttention(*args, **kwargs)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.adaptedlayernorm = AdaptedLayerNorm(mean, variance)

    def forward(self, x):
        # Apply the attention block
        update = self.multiheadattention(x)
        update = self.leakyrelu(update)
        update = self.adaptedlayernorm(update + x)  # Norm residual, because it is returned in the end

        return update
    

def test_attentionblock() -> None:
    """Test the AttentionBlock class."""
    # All tests are done with batch_size=16 and embed_dim=8
    batch_size = 16
    embed_dim = 8

    # Create a test tensor
    test_tensor = torch.randn(batch_size, embed_dim)

    # Get the batch-wise mean and variance of the tensor
    mean = mean_fn(test_tensor)
    variance = var_fn(test_tensor)

    # Create an AttentionBlock object
    attentionblock = AttentionBlock(mean, variance, 0.5, embed_dim=embed_dim, num_heads=2)

    # Forward pass the test tensor through the AttentionBlock
    result = attentionblock(test_tensor)

    # Check if the shape of the result is correct
    assert result.shape == test_tensor.shape

    # Check that the result is not equal to the input tensor
    assert not allclose(result, test_tensor)

    # Check that the mean and variance of the result are equal to those of the input tensor
    assert allclose(mean_fn(result), mean)
    assert allclose(var_fn(result), variance)
    print("AttentionBlock test passed.")


class MLP(nn.Module):
    """
    A simple MLP with a single hidden layer.

    Edits the residual stream. Applies AdaptedLayerNorm to the residual stream after editing.
    Uses LeakyReLU as activation function.

    Arguments
    ---------

    embed_dim (int): The dimension of the input and output.
                        Type: int

    mean (torch.Tensor): The mean of the input tensor.
                            Type: torch.Tensor

    variance (torch.Tensor): The variance of the input tensor.
                            Type: torch.Tensor

    expansion_factor (float): The expansion factor of the hidden layer.
                              Type: float

    negative_slope (float): The negative slope of the LeakyReLU activation function.
                            Type: float
    """
    
    def __init__(
            self, 
            embed_dim: int, 
            mean: torch.Tensor, 
            variance: torch.Tensor, 
            expansion_factor: float = 2.0,
            negative_slope: float = 0.5
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, int(embed_dim * expansion_factor))
        self.linear2 = nn.Linear(int(embed_dim * expansion_factor), embed_dim)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.adaptedlayernorm = AdaptedLayerNorm(mean, variance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the MLP
        update = self.linear1(x)
        update = self.leakyrelu(update)
        update = self.linear2(update)
        update = self.adaptedlayernorm(update + x)  # Norm residual, because it is returned in the end

        return update
    

def test_mlp() -> None:
    """Test the MLP class, similar to the AttentionBlock test."""
    # All tests are done with batch_size=16 and embed_dim=8
    batch_size = 16
    embed_dim = 8

    # Create a test tensor
    test_tensor = torch.randn(batch_size, embed_dim)

    # Get the batch-wise mean and variance of the tensor
    mean = mean_fn(test_tensor)
    variance = var_fn(test_tensor)

    # Create an MLP object
    mlp = MLP(embed_dim, mean, variance, 2.0, 0.5)

    # Forward pass the test tensor through the MLP
    result = mlp(test_tensor)

    # Check if the shape of the result is correct
    assert result.shape == test_tensor.shape

    # Check that the result is not equal to the input tensor
    assert not allclose(result, test_tensor)

    # Check that the mean and variance of the result are equal to those of the input tensor
    assert allclose(mean_fn(result), mean)
    assert allclose(var_fn(result), variance)
    print("MLP test passed.")



def run_tests() -> None:
    """Run all tests."""
    test_adaptedlayernorm()
    test_attentionblock()
    test_mlp()
    print("All tests passed.")


if __name__ == "__main__":
    run_tests()
