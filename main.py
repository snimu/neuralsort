import argparse
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt


class SelfAttention(nn.Module):
    """Self attention layer.
    Want it to work with lists, don't want to use nn.MultiheadAttention
    because of issues with shapes.
    """
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_weights = torch.softmax(Q @ K.transpose(-2, -1) / (self.dim ** 0.5), dim=-1)
        return attention_weights @ V


class ScaledNorm(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.mean = 0.0
        self.std = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x
        # return (
        #     x * torch.sqrt(
        #         self.std**2 + torch.finfo(torch.float).eps
        #     )
        #     + self.mean
        # )

    def unittest(self, x: torch.Tensor) -> None:
        self.mean, self.std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        y = self.norm(x)

        print(y.std(), x.std())
        print(y.mean(), x.mean())
        assert torch.allclose(y.std(), x.std())
        assert torch.allclose(y.mean(), x.mean())


class AttentionBlock(nn.Module):
    def __init__(self, size: int, use_norm: bool, layer: int, negative_slope: float = 0.5):
        super().__init__()
        self.use_norm = use_norm
        self.layers = nn.Sequential()
        # Use norm after every layer
        #   s.t. the mean and std are preserved throughout the network
        self.layers.add_module(
            f"SelfAttention{layer}",
            SelfAttention(size)
        )
        self.layers.add_module(
            f"LeakyReLU{layer}",
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        if use_norm:
            self.layers.add_module(
                f"Norm{layer}_2",
                ScaledNorm(size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # SelfAttention requires (seq_len, batch, feature)
        x = self.layers(x)
        x = x.permute(1, 0, 2)
        return x

    def unittest(self, x: torch.Tensor) -> None:
        if self.use_norm:
            mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
            for module in self.modules():
                if not isinstance(module, ScaledNorm):
                    continue
                module.mean = mean
                module.std = std
            y = self.forward(x)
            assert torch.allclose(y.std(), x.std())
            assert torch.allclose(y.mean(), x.mean())


class MLPBlock(nn.Module):
    def __init__(
            self,
            size: int,
            use_norm: bool,
            layer: int,
            expansion: float,
            negative_slope: float = 0.5
    ):
        super().__init__()
        self.use_norm = use_norm
        self.layers = nn.Sequential()
        self.layers.add_module(
            f"Linear{layer}_1",
            nn.Linear(size, int(expansion * size))
        )
        self.layers.add_module(
            f"LeakyReLU{layer}_1",
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        if use_norm:
            self.layers.add_module(
                f"Norm{layer}_11",
                ScaledNorm(size)
            )

        self.layers.add_module(
            f"Linear{layer}_2",
            nn.Linear(int(expansion * size), size)
        )
        self.layers.add_module(
            f"LeakyReLU{layer}_2",
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        if use_norm:
            self.layers.add_module(
                f"Norm{layer}_21",
                ScaledNorm(size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def unittest(self, x: torch.Tensor) -> None:
        if self.use_norm:
            mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
            for module in self.modules():
                if not isinstance(module, ScaledNorm):
                    continue
                module.mean = mean
                module.std = std
            y = self.layers(x)
            assert torch.allclose(y.std(), x.std())
            assert torch.allclose(y.mean(), x.mean())


class SortNet(nn.Module):
    def __init__(
            self,
            size: int,
            num_layers: int,
            use_fc: bool,
            expansion: float = 2,
            use_norm: bool = True,
            use_residual: bool = True,
            negative_slope: float = 0.5
    ):
        assert num_layers % 2 == 0, "Number of layers must be even"
        super(SortNet, self).__init__()

        self.use_residual = use_residual
        self.use_norm = use_norm

        self.layers = nn.Sequential()
        for i in range(num_layers // 2):
            self.layers.add_module(
                f"AttentionBlock{i}",
                AttentionBlock(size, use_norm, i, negative_slope)
            )
            self.layers.add_module(
                f"MLPBlock{i+1}" if use_fc else f"AttentionBlock{i+1}",
                MLPBlock(size, use_norm, i+1, expansion, negative_slope)
                if use_fc
                else AttentionBlock(size, use_norm, i+1, negative_slope)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
        for module in self.modules():
            if not isinstance(module, ScaledNorm):
                continue
            module.mean = mean
            module.std = std

        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)

        return x

    def unittest(self, x: torch.Tensor) -> None:
        if self.use_norm:
            y = self.forward(x)
            assert torch.allclose(y.std(), x.std())
            assert torch.allclose(y.mean(), x.mean())


def generate_dataset(length, min_val, max_val, batch_size, device):
    unsorted_tensor = torch.randint(min_val, max_val, (batch_size, 1, length), device=device)
    sorted_tensor = torch.sort(unsorted_tensor, dim=2)[0]
    return unsorted_tensor, sorted_tensor


def training_loop(
        num_epochs,
        model,
        loss_fn,
        optimizer,
        scheduler,
        length,
        min_val,
        max_val,
        batch_size,
        loop: tqdm,
        device,
):
    metrics = {"epoch": [], "train_loss": [], "valid_loss": [], "train_accuracy": [], "valid_accuracy": []}
    loop.write(
        f"Training with settings: "
        f"{num_epochs=}, {batch_size=}, {length=}, {min_val=}, {max_val=}"
    )
    metrics_title = "epoc | phase | loss | accuracy"
    loop.write(f"\n{metrics_title}\n" + '-' * len(metrics_title))
    for epoch in range(num_epochs):
        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            inputs, labels = generate_dataset(length, min_val, max_val, batch_size, device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs.float())
                loss = loss_fn(outputs, labels.float())
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (outputs.round() == labels).sum().item()

            epoch_loss = running_loss / batch_size  # adjust for the total number of samples
            epoch_acc = running_corrects / batch_size  # adjust for the total number of samples

            loop.write(f"{epoch}/{num_epochs} | {phase} | {epoch_loss:.4f} | {epoch_acc:.4f}")

            metrics["epoch"].append(epoch)
            metrics[phase+"_loss"].append(epoch_loss)
            metrics[phase+"_accuracy"].append(epoch_acc)

    df = pd.DataFrame(metrics)
    filename = (
        f"training_metrics_min_{min_val}_max_{max_val}_length_{length}_num_epochs_{num_epochs}_use_fc_{model.use_fc}.csv"
    )
    df.to_csv(filename, index=False)
    return df


def plot_metrics(df, min_val, max_val, length, num_epochs, use_fc, save_plot):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df['epoch'], df['train_loss'], label='Train')
    ax1.plot(df['epoch'], df['valid_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(df['epoch'], df['train_accuracy'], label='Train')
    ax2.plot(df['epoch'], df['valid_accuracy'], label='Validation')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    fig.suptitle(f'Training Metrics: Min={min_val}, Max={max_val}, Length={length}, Epochs={num_epochs}, Use_FC={use_fc}')

    if save_plot:
        filename = f"plot_min_{min_val}_max_{max_val}_length_{length}_num_epochs_{num_epochs}_use_fc_{use_fc}.png"
        plt.savefig(filename)
    else:
        plt.show()


def unittest() -> None:
    norm = ScaledNorm(10)
    x = torch.randn(10, 10) * 3 + 10
    norm.unittest(x)

    attention = AttentionBlock(10, True, 0)
    attention.unittest(x)

    mlp = MLPBlock(10, True, 0, 2)
    mlp.unittest(x)

    model = SortNet(10, 2, True)
    model.unittest(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SortNet Training')
    parser.add_argument('--min_val', nargs='+', type=int, default=[0])
    parser.add_argument('--max_val', nargs='+', type=int, default=[100])
    parser.add_argument('--length', nargs='+', type=int, default=[10])
    parser.add_argument('--num_epochs', nargs='+', type=int, default=[100])
    parser.add_argument('--num_layers', nargs='+', type=int, default=[1])
    parser.add_argument('--negative_slope', nargs='+', type=float, default=[0.5])
    parser.add_argument('--use_residual', type=bool, default=True, help="Use residual connections")
    parser.add_argument('--use_fc', action='store_true', help="Use a fully connected layer")
    parser.add_argument('-p', '--plot', action='store_true', help="Plot the metrics")
    parser.add_argument('-s', '--save_plot', action='store_true', help="Save the plot")
    parser.add_argument('-u', '--unittest', action='store_true', help="Run unittests")

    args = parser.parse_args()

    if args.unittest:
        unittest()
        exit()

    all_values = list(itertools.product(args.min_val, args.max_val, args.length, args.num_epochs, args.num_layers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loop = tqdm(
        all_values,
        total=len(args.min_val) * len(args.max_val) * len(args.length) * len(args.num_epochs) * len(args.num_layers)
    )

    for (min_val, max_val, length, num_epochs, num_layers) in loop:
        input_dim = output_dim = length
        # num_heads = 1
        batch_size = 128
        model = SortNet(
            input_dim, output_dim, num_layers, args.use_fc, args.use_residual, args.negative_slope
        ).to(device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        df = training_loop(
            num_epochs,
            model,
            loss_fn,
            optimizer,
            scheduler,
            length,
            min_val,
            max_val,
            batch_size,
            loop,
            device,
        )
        if args.plot:
            plot_metrics(
                df, min_val, max_val, length, num_epochs, args.use_fc, args.save_plot
            )
