"""Sort a list of integers using an ANN.

"""

import argparse
import functools
import math
import os

import torch
from torch import nn
import torch.optim
from torch.optim import Adam
import rich
import matplotlib.pyplot as plt
import numpy as np
import torchinfo
import transformers
import wandb 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_fn(x: torch.Tensor) -> torch.Tensor:
    """A convenience function wrapping torch.mean."""
    return torch.mean(x, dim=1, keepdim=True)


def var_fn(x: torch.Tensor) -> torch.Tensor:
    """A convenience function wrapping torch.var."""
    return torch.var(x, dim=1, keepdim=True)


def allclose(x: torch.Tensor, y: torch.Tensor) -> bool:
    """A convenience function wrapping torch.allclose."""
    return torch.allclose(x, y, rtol=1e-3, atol=1e-4)


class MultiheadSelfAttention(nn.MultiheadAttention):
    def __init__(
            self, 
            dropout: float = 0.0, 
            bias: bool = True, 
            add_bias_kv: bool = False, 
            add_zero_attn: bool = False, 
            kdim: int = None, 
            vdim: int = None
    ):

        super().__init__(
            1, 1, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of shape (batch_size, embed_dim),
        # but MultiheadAttention expects (batch_size, seq_len=embed_dim, 1)
        x = x.reshape(x.shape[0], x.shape[1], 1)

        # Forward pass of self-attention, with batch_first=True
        x, _ = super().forward(x, x, x)

        # Shape back
        x = x.reshape(x.shape[0], x.shape[1])
        return x


class AttentionBlock(nn.Module):
    def __init__(
            self, 
            seq_len: int,
            dropout: float = 0.0, 
            bias: bool = True, 
            add_bias_kv: bool = False, 
            add_zero_attn: bool = False, 
            kdim: int = None, 
            vdim: int = None
    ) -> None:
        super().__init__()
        self.multiheadattention = MultiheadSelfAttention(
            dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim
        )
        self.layernorm = nn.LayerNorm(seq_len)

    def forward(self, x: torch.Tensor) -> None:
        # First norm input over sequence length 
        # so that the multiheadattention can learn to sort the input,
        # independent from its scale.
        update = self.layernorm(x)  
        update = self.multiheadattention(update)
        update = update + x
        return update


class MLP(nn.Module):
    def __init__(
        self, 
        seq_len: int, 
        expansion_factor: float = 3.0,
        negative_slope: float = 0.5,
    ) -> None:
        super().__init__()
        self.expand = nn.Linear(seq_len, int(seq_len * expansion_factor))
        self.project = nn.Linear(int(seq_len * expansion_factor), seq_len)
        self.layernorm = nn.LayerNorm(seq_len)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First norm input over sequence length
        # so that the MLP can learn to sort the input,
        # independent from its scale.
        update = self.layernorm(x)

        # Apply the MLP
        update = self.expand(update)
        update = self.activation(update)
        update = self.project(update)
        update = update + x
        return update


class SortNet(nn.Module):
    def __init__(
            self,
            seq_len: int,
            negative_slope: float = 0.5,
            expansion_factor: float = 3.0,
            num_layers: int = 2,
            use_mlp: bool = False,
            dropout: float = 0.0, 
            bias: bool = True, 
            add_bias_kv: bool = False, 
            add_zero_attn: bool = False, 
            kdim: int = None, 
            vdim: int = None
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for nlayer in range(num_layers):
            if nlayer % 2 == 0 or not use_mlp:
                self.layers.append(
                    AttentionBlock(
                        seq_len, 
                        dropout, 
                        bias, 
                        add_bias_kv, 
                        add_zero_attn, 
                        kdim, 
                        vdim,
                    )
                )
            else:
                self.layers.append(
                    MLP(
                        seq_len, 
                        expansion_factor, 
                        negative_slope,
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = mean_fn(x)
        variance = var_fn(x)
        for layer in self.layers:
            x = layer(x)

        # Normalize x again and then bring it to its original scale
        x = (x - mean_fn(x)) / torch.sqrt(var_fn(x) + torch.finfo(float).eps)
        x = x * torch.sqrt(variance + torch.finfo(float).eps) + mean
        return x


def validate(
    model: SortNet, 
    niter: int, 
    batch_size: int, 
    seq_len: int, 
    low: int, 
    high: float
) -> tuple[float, float]:
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
    The loss and top-1 accuracy of the model on the data.
    """
    # Create a loss function
    loss_fn = nn.MSELoss() if hparams.loss == "mse" else nn.L1Loss()

    loss = 0.0
    top1_acc = 0.0
    for _ in range(niter):
        # Generate a batch of data
        data, y = generate_data(batch_size, seq_len, low, high)

        # Forward pass the data through the network
        result = model(data)

        # Compute the loss
        loss += loss_fn(result, y).item()

        # Compute the top-1 accuracy
        result_rounded = torch.round(result)
        top1_correct = (result_rounded == y).sum().item()
        top1_acc += top1_correct / (batch_size * seq_len)

    return loss / niter, top1_acc / niter



def generatate_bimodal(batch_size: int, seq_len: int, low: int, high: int) -> torch.Tensor:
    # Bimodal distribution: choose two random centers for each sequence
    # Generate two mean-tensors and two std-tensors
    half_width = int(seq_len // 2)
    centers1 = torch.randint(low, high, (batch_size, 1), dtype=torch.float).repeat(1, half_width)
    centers2 = torch.randint(low, high, (batch_size, 1), dtype=torch.float).repeat(1, seq_len - half_width)
    std_dev1 = torch.randint(0, int(round((high-low / 5.0))), (batch_size, 1), dtype=torch.float).repeat(1, half_width)
    std_dev2 = torch.randint(0, int(round((high-low / 5.0))), (batch_size, 1), dtype=torch.float).repeat(1, seq_len - half_width)

    # Generate two normal distributions from the mean-tensors and std-tensors
    x1 = torch.normal(mean=centers1, std=std_dev1)
    x2 = torch.normal(mean=centers2, std=std_dev2)
    
    # Concatenate the two distributions.
    x = torch.cat([x1, x2], dim=1)

    return x


def generate_exponential(batch_size: int, seq_len: int, low: int, high: int) -> torch.Tensor:
    # Exponential distribution: generate data from an exponential distribution
    scale = (high - low) / 5.0  # Adjust the scale so that most points fall between low and high
    x = torch.distributions.exponential.Exponential(scale).sample((batch_size, seq_len))

    return x


def generate_clustered(batch_size: int, seq_len: int, low: int, high: int) -> torch.Tensor:
    # Clustered distribution: choose a random center for each sequence
    centers = torch.randint(low, high, (batch_size, 1), dtype=torch.float).repeat(1, seq_len)
    std_dev = (high - low) / 5.0  # Adjust the standard deviation so that most points fall between low and high
    x = torch.normal(mean=centers, std=std_dev)

    return x


def generate_data(
    batch_size: int, seq_len: int, low: int = -100, high: int = 100, dist_type: str = None
) -> torch.Tensor:
    """Generate a batch of data for testing."""
    
    # Split the batch into several parts and generate each part separately
    # so that each batch has a diverse set of distributions
    batch = []
    split_num = 4
    batch_size_partial = batch_size // split_num
    for idx in range(split_num):
        if idx == split_num - 1:
            batch_size_partial += batch_size % split_num

        # Randomly choose the type of distribution for this batch
        dist_type = np.random.choice(
            ['uniform', 'clustered', 'bimodal', 'exponential', 'half_normal']
        ) if dist_type is None else dist_type

        if dist_type == 'uniform':
            x_partial = torch.randint(low, high, (batch_size_partial, seq_len), dtype=torch.float)
        elif dist_type == 'clustered':
            x_partial = generate_clustered(batch_size_partial, seq_len, low, high)
        elif dist_type == 'bimodal':
            x_partial = generatate_bimodal(batch_size_partial, seq_len, low, high)
        else:  # 'exponential'
            x_partial = generate_exponential(batch_size_partial, seq_len, low, high)

        # Append to the batch
        batch.append(x_partial)

    # Concatenate the batch
    x = torch.cat(batch, dim=0)

    # Clip x so that all its elements are inside [low, high] (inclusive)
    x = torch.clamp(x, low, high)
    
    # Round to nearest integer
    x = torch.round(x)

    y, _ = torch.sort(x, dim=1)
    return x.to(DEVICE), y.to(DEVICE)



def train_loop(hparams: argparse.Namespace) -> None:
    """
    The training loop for the SortNet.
    """
    if DEVICE == "cpu":  # only log locally (where I dont have a proper GPU)
        wandb.init(project="sorting", config=vars(hparams), )

    # Create a SortNet object
    sortnet = SortNet(
        hparams.seq_len,
        hparams.negative_slope,
        hparams.expansion_factor,
        hparams.num_layers,
        hparams.use_mlp
    ).to(DEVICE)
    
    if DEVICE == "cpu":
        wandb.watch(sortnet, log_freq=10, log="all")

    # Create a loss function
    loss_fn = nn.MSELoss() if hparams.loss == "mse" else nn.L1Loss()

    # Create an optimizer
    wdm = hparams.weight_decay_multiple
    wd = wdm * math.sqrt(1 / (hparams.num_epochs + torch.finfo(torch.float).eps))
    optimizer = Adam(sortnet.parameters(), lr=hparams.learning_rate, weight_decay=wd)

    # Create a scheduler
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(20, hparams.num_epochs // 10),
        num_training_steps=hparams.num_epochs,
        num_cycles=hparams.num_cycles,
    )

    # Create validation functions
    niter = 5
    validate_id = functools.partial(
        validate, 
        niter=niter,
        batch_size=hparams.batch_size, seq_len=hparams.seq_len, 
        low=hparams.low, high=hparams.high
    )
    validate_ood = functools.partial(
        validate, 
        niter=niter,          
        batch_size=hparams.batch_size, seq_len=hparams.seq_len, 
        low=hparams.high, high=hparams.high * 3
    )

    # Train the network
    for epoch in range(hparams.num_epochs):
        # Generate a batch of data
        data, y = generate_data(hparams.batch_size, hparams.seq_len, hparams.low, hparams.high)
        
        # Forward pass the data through the network
        result = sortnet(data)

        # Compute the loss
        loss = loss_fn(result, y)

        # Backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        val_loss_id, val_acc_id = validate_id(sortnet)
        val_loss_ood, val_acc_ood = validate_ood(sortnet)

        # Log the losses
        if DEVICE == "cpu":
            wandb.log(
                {
                    "train_loss": loss.item(), 
                    "val_loss_id": val_loss_id, 
                    "val_loss_ood": val_loss_ood,
                    "val_acc_id": val_acc_id,
                    "val_acc_ood": val_acc_ood,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Print the loss
        if epoch % 10 == 0:
            rich.print(
                f"Epoch {epoch} | Train-loss: {loss.item():.2f} | "
                f"ID-loss: {val_loss_id:.2f} | OOD-loss: {val_loss_ood:.2f}"
                f"ID-acc: {val_acc_id:.2f} | OOD-acc: {val_acc_ood:.2f}"
            )

        if hparams.use_lr_scheduler:
            scheduler.step()

    if hparams.check_results:
        check_results(sortnet, hparams)


def check_results(sortnet: SortNet, hparams: argparse.Namespace) -> None:
    x, _ = generate_data(1, hparams.seq_len, hparams.low, hparams.high)
    torchinfo.summary(sortnet, input_data=x)

    
    # Plot the relative error per element
    num_iters = 10
    diff = torch.empty(num_iters, hparams.seq_len)
    for i in range(num_iters):
        x, y = generate_data(1, hparams.seq_len, hparams.low, hparams.high)
        result = sortnet(x)

        diff[i] = ((y - result) / (y + result)).abs()

    diff = diff.mean(dim=0).reshape(-1).abs().detach().numpy()
    plt.plot(np.arange(len(diff)), diff)
    plt.show()


def run_tests() -> None:
    """Run all tests."""
    test_adaptedlayernorm()
    test_attentionblock()
    test_mlp()
    print("All tests passed.")

def get_hparams() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--test", action="store_true", help="Run tests")
    parser.add_argument("-v", "--visualize_output", action="store_true", help="Visualize the output distribution of samples in a batch")
    parser.add_argument("-c", "--check_results", help="Check the model performance after training", action="store_true")

    parser.add_argument("--use_mlp", type=bool, default=True, help="Use MLPs in addition to AttentionBlocks")
    parser.add_argument("--use_residual", type=bool, default=True, help="Use residuals in the AttentionBlocks and MLPs")
    parser.add_argument("--normalize", type=bool, default=True, help="Normalize the input to the network")
    parser.add_argument("--use_lr_scheduler", type=bool, default=True, help="Use a learning rate scheduler")

    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--seq_len", type=int, default=100, help="The length of the lists.")

    parser.add_argument("--low", type=int, default=-100, help="Lower bound of the data")
    parser.add_argument("--high", type=int, default=100, help="Upper bound of the data")

    parser.add_argument("--learning_rate", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--negative_slope", type=float, default=0.5, help="Negative slope of the LeakyReLU")
    parser.add_argument("--expansion_factor", type=float, default=3.0, help="Expansion factor of the MLP")
    parser.add_argument("--num_cycles", type=int, default=1, help="Number of cycles for the scheduler")

    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1"], help="The loss function to use")
    parser.add_argument(
        "--weight_decay_multiple", default=0.03, 
        help="Weight decay multiple. "
             "See https://arxiv.org/pdf/1711.05101.pdf for more information."
    )

    hparams = parser.parse_args()
    rich.print(vars(hparams))
    return hparams

def visualize_output_distribution():
    """Visualize the output distribution of samples in a batch."""
    fig, axs = plt.subplots(3, 4)
    for y in range(3):
        for x in range(4):
            dist_type = np.random.choice(['uniform', 'clustered', 'bimodal', 'exponential'])
            batch_data, _ = generate_data(hparams.batch_size, hparams.seq_len, hparams.low, hparams.high, dist_type)
            axs[y, x].hist(batch_data[0], bins='auto')  # visualize first batch
            axs[y, x].set_title(f"{dist_type=}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    hparams = get_hparams()

    if hparams.test:
        run_tests()
    elif hparams.visualize_output:
        visualize_output_distribution()
    else:
        train_loop(hparams)


