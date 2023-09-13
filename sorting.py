"""Sort a list of integers using MultiheadAttention with residuals and LayerNorms.

Important ideas:
    - MultiheadAttention: can hopefully be used for sorting
    - LeakyReLU: with a large negative slope, s.t. the negative values are also used
    - AdaptedLayerNorm: A norm that normalizes the input to a given mean and variance; that of the input list
                        this is used to normalize the output of the MultiheadAttention to that of the input list,
                        helping the network to learn to sort the list
"""

import argparse
import functools
import math
import os

import torch
from torch import nn
import torch.optim
from torch.optim import Adam
import wandb
import rich
import matplotlib.pyplot as plt
import numpy as np
import torchinfo


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

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor):
        # Normalize the mean of variance of x to 0 and 1, respectively
        x = (x - mean_fn(x)) / torch.sqrt(var_fn(x))

        # Scale and shift x to have the same mean and variance as the input tensor
        x = x * torch.sqrt(variance) + mean

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
    adaptedlayernorm = AdaptedLayerNorm()

    # Create a second test tensor
    test_tensor2 = torch.randn(batch_size, embed_dim)

    # Normalize the second test tensor
    test_tensor2 = adaptedlayernorm(test_tensor2, mean, variance)

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

    use_residual: Whether to use residuals in the AttentionBlocks and MLPs.
                    Type: bool
                    Default: True

    normalize: Whether to normalize the input to the network.
                Type: bool  
                Default: True
    
    *args: Arguments for MultiheadAttention
    **kwargs: Keyword arguments for MultiheadAttention
    """

    def __init__(
            self, 
            negative_slope: float = 0.5, 
            use_residual: bool = True, 
            normalize: bool = True,
            *args, **kwargs
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.normalize = normalize

        self.multiheadattention = MultiheadSelfAttention(*args, **kwargs)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.adaptedlayernorm = AdaptedLayerNorm()

    def forward(self, x, mean: torch.Tensor, variance: torch.Tensor):
        # Apply the attention block
        update = self.multiheadattention(x)
        update = self.leakyrelu(update)
        update = update + x if self.use_residual else update
        if self.normalize:
            # Norm residual, because it is returned in the end
            update = self.adaptedlayernorm(update, mean, variance)  

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
    attentionblock = AttentionBlock(0.5, embed_dim=embed_dim, num_heads=2)

    # Forward pass the test tensor through the AttentionBlock
    result = attentionblock(test_tensor, mean, variance)

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

    use_residuals (bool): Whether to use residuals in the AttentionBlocks and MLPs.
                            Type: bool

    mean (torch.Tensor): The mean of the input tensor.
                            Type: torch.Tensor

    variance (torch.Tensor): The variance of the input tensor.
                            Type: torch.Tensor

    expansion_factor (float): The expansion factor of the hidden layer.
                              Type: float

    negative_slope (float): The negative slope of the LeakyReLU activation function.
                            Type: float

    normalize (bool): Whether to normalize the input to the network.
                        Type: bool
    """
    
    def __init__(
            self, 
            embed_dim: int, 
            use_residuals: bool = True,
            expansion_factor: float = 2.0,
            negative_slope: float = 0.5,
            normalize: bool = True,
    ) -> None:
        super().__init__()
        self.use_residuals = use_residuals
        self.normalize = normalize

        self.linear1 = nn.Linear(embed_dim, int(embed_dim * expansion_factor))
        self.linear2 = nn.Linear(int(embed_dim * expansion_factor), embed_dim)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.adaptedlayernorm = AdaptedLayerNorm()

    def forward(
            self, x: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        # Apply the MLP
        update = self.linear1(x)
        update = self.leakyrelu(update)
        update = self.linear2(update)
        update = update + x if self.use_residuals else update
        if self.normalize:
            # Norm residual, because it is returned in the end
            update = self.adaptedlayernorm(update, mean, variance)  

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
    mlp = MLP(embed_dim, 2.0, 0.5)

    # Forward pass the test tensor through the MLP
    result = mlp(test_tensor, mean, variance)

    # Check if the shape of the result is correct
    assert result.shape == test_tensor.shape

    # Check that the result is not equal to the input tensor
    assert not allclose(result, test_tensor)

    # Check that the mean and variance of the result are equal to those of the input tensor
    assert allclose(mean_fn(result), mean)
    assert allclose(var_fn(result), variance)
    print("MLP test passed.")


class SortNet(nn.Module):
    """
    An ANN for sorting a list of integers.

    Arguments
    ---------

    embed_dim (int): The dimension of the input and output.
                        Type: int

    negative_slope (float): The negative slope of the LeakyReLU activation function.
                            Type: float

    expansion_factor (float): The expansion factor of the hidden layer.
                              Type: float

    num_layers (int): The number of layers in the ANN.
                      Type: int

    use_mlp (bool): Whether to use MLPs or AttentionBlocks.
                    Type: bool

    use_residuals (bool): Whether to use residuals in the AttentionBlocks and MLPs.
                            Type: bool

    normalize (bool): Whether to normalize the input to the network.
                        Type: bool
    """
    
    def __init__(
            self,
            embed_dim: int,
            negative_slope: float = 0.5,
            expansion_factor: float = 2.0,
            num_layers: int = 2,
            use_mlp: bool = False, 
            use_residuals: bool = True,
            normalize: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.negative_slope = negative_slope
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.use_mlp = use_mlp
        self.use_residuals = use_residuals
        self.normalize = normalize

        # Create the layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                AttentionBlock(
                    self.negative_slope, self.use_residuals, self.normalize, self.embed_dim, 2
                )
            )
            if self.use_mlp:
                self.layers.append(
                    MLP(
                        self.embed_dim, self.use_residuals, self.expansion_factor, self.negative_slope,
                        normalize=self.normalize
                    )
                )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the mean and variance of the input tensor
        mean = mean_fn(x)
        variance = var_fn(x)

        # Apply the layers
        for layer in self.layers:
            x = layer(x, mean, variance)

        return x
        

def test_sortnet() -> None:
    """Test the SortNet class."""
    # All tests are done with batch_size=16 and embed_dim=8
    batch_size = 16
    embed_dim = 8

    # Create a test tensor
    test_tensor = torch.randn(batch_size, embed_dim)

    # Create a SortNet object
    sortnet = SortNet(embed_dim, 0.5, 2.0, 2)

    # Forward pass the test tensor through the SortNet
    result = sortnet(test_tensor)

    # Check if the shape of the result is correct
    assert result.shape == test_tensor.shape

    # Check that the result is not equal to the input tensor
    assert not allclose(result, test_tensor)

    # Check that the mean and variance of the result are equal to those of the input tensor
    assert allclose(mean_fn(result), mean_fn(test_tensor))
    assert allclose(var_fn(result), var_fn(test_tensor))

    print("SortNet test passed.")


def validate(model: SortNet, niter: int, batch_size: int, seq_len: int, low: int, high: float) -> float:
    """Validate the model on a batch of data.
    
    Arguments
    ---------

    model (SortNet): The model to validate.
                        Type: SortNet

    niter (int): The number of iterations to validate.
                    Type: int

    batch_size (int): The batch size of the data.
                        Type: int

    seq_len (int): The sequence length of the data.
                    Type: int

    low (int): The lower bound of the data.
                Type: int

    high (int): The upper bound of the data.
                Type: int

    Returns
    -------
    The loss of the model on the data.
    """
    # Create a loss function
    loss_fn = nn.MSELoss() if hparams.loss == "mse" else nn.L1Loss()

    loss = 0.0
    for _ in range(niter):
        # Generate a batch of data
        data, y = generate_data(batch_size, seq_len, low, high)

        # Forward pass the data through the network
        result = model(data)

        # Compute the loss
        loss += loss_fn(result, y).item()

    return loss / niter


def generate_data(batch_size: int, seq_len: int, low: int = -100, high: int = 100) -> torch.Tensor:
    """Generate a batch of data for testing."""
    x = torch.randint(low, high, (batch_size, seq_len), dtype=torch.float)
    y, _ = torch.sort(x, dim=1)
    return x, y


def train_loop(hparams: argparse.Namespace) -> None:
    """
    The training loop for the SortNet.
    """
    wandb.init(project="sorting", config=vars(hparams), )

    # Create a SortNet object
    sortnet = SortNet(
        hparams.embed_dim,
        hparams.negative_slope,
        hparams.expansion_factor,
        hparams.num_layers,
        hparams.use_mlp
    )
    wandb.watch(sortnet)

    # Create a loss function
    loss_fn = nn.MSELoss() if hparams.loss == "mse" else nn.L1Loss()

    # Create an optimizer
    wdm = hparams.weight_decay_multiple
    wd = wdm * math.sqrt(1 / (hparams.num_epochs + torch.finfo(torch.float).eps))
    optimizer = Adam(sortnet.parameters(), lr=hparams.learning_rate, weight_decay=wd)

    # Create validation functions
    niter = 5
    validate_id = functools.partial(
        validate, 
        niter=niter,
        batch_size=hparams.batch_size, seq_len=hparams.embed_dim, 
        low=hparams.low, high=hparams.high
    )
    validate_ood = functools.partial(
        validate, 
        niter=niter,          
        batch_size=hparams.batch_size, seq_len=hparams.embed_dim, 
        low=hparams.high, high=hparams.high * 3
    )

    # Train the network
    train_losses = []
    val_losses_id = []
    val_losses_ood = []
    for epoch in range(hparams.num_epochs):
        # Generate a batch of data
        data, y = generate_data(hparams.batch_size, hparams.embed_dim, hparams.low, hparams.high)
        
        # Forward pass the data through the network
        result = sortnet(data)

        # Compute the loss
        loss = loss_fn(result, y)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        val_loss_id = validate_id(sortnet)
        val_loss_ood = validate_ood(sortnet)

        # Save the losses
        train_losses.append(loss.item())
        val_losses_id.append(val_loss_id)
        val_losses_ood.append(val_loss_ood)

        # Log the losses
        wandb.log("train_loss", loss.item())
        wandb.log("val_loss_id", val_loss_id)
        wandb.log("val_loss_ood", val_loss_ood)

        # Print the loss
        if epoch % 10 == 0:
            rich.print(
                f"Epoch {epoch} | Train-loss: {loss.item():.2f} | "
                f"ID-loss: {val_loss_id:.2f} | OOD-loss: {val_loss_ood:.2f}"
            )

    if hparams.plot_losses:
        plt.plot(np.arange(len(train_losses)), train_losses, label="Train loss")
        plt.plot(np.arange(len(val_losses_id)), val_losses_id, label="ID loss")
        # plt.plot(np.arange(len(val_losses_ood)), val_losses_ood, label="OOD loss")
        plt.legend()
        plt.show()

    if hparams.check_results:
        check_results(sortnet, hparams)


def check_results(sortnet: SortNet, hparams: argparse.Namespace) -> None:
    x, _ = generate_data(1, hparams.embed_dim, hparams.low, hparams.high)
    torchinfo.summary(sortnet, input_data=x)

    
    # Plot the relative error per element
    num_iters = 10
    diff = torch.empty(num_iters, hparams.embed_dim)
    for i in range(num_iters):
        x, y = generate_data(1, hparams.embed_dim, hparams.low, hparams.high)
        result = sortnet(x)

        diff[i] = ((y - result) / (y + result)).abs()

    diff = diff.mean(dim=0).reshape(-1).abs().detach().numpy()
    plt.plot(np.arange(len(diff)), diff)
    plt.show()


def get_hparams() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--test", action="store_true", help="Run tests")
    parser.add_argument("-e", "--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-a", "--low", type=int, default=-100, help="Lower bound of the data")
    parser.add_argument("-u", "--high", type=int, default=100, help="Upper bound of the data")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "-w", "--weight_decay_multiple", default=0.03, 
        help="Weight decay multiple. "
             "See https://arxiv.org/pdf/1711.05101.pdf for more information."
    )
    parser.add_argument("-d", "--embed_dim", type=int, default=100, help="Embedding dimension; equal to the sequence length")
    parser.add_argument("-n", "--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("-m", "--use_mlp", action="store_true", help="Use MLPs in addition to AttentionBlocks")
    parser.add_argument("-r", "--negative_slope", type=float, default=0.5, help="Negative slope of the LeakyReLU")
    parser.add_argument("-f", "--expansion_factor", type=float, default=3.0, help="Expansion factor of the MLP")

    parser.add_argument("-c", "--check_results", help="Check the model performance after training", action="store_true")
    parser.add_argument("-p", "--plot_losses", help="Plot the losses", action="store_true")

    parser.add_argument("--use_residual", action="store_true", help="Use residuals in the AttentionBlocks and MLPs")
    parser.add_argument("--normalize", action="store_true", help="Normalize the input to the network")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"], help="The loss function to use")

    hparams = parser.parse_args()
    rich.print(vars(hparams))
    return hparams


def run_tests() -> None:
    """Run all tests."""
    test_adaptedlayernorm()
    test_attentionblock()
    test_mlp()
    print("All tests passed.")


if __name__ == "__main__":
    hparams = get_hparams()

    if hparams.test:
        run_tests()
    else:
        train_loop(hparams)
