import argparse
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import matplotlib.pyplot as plt


class SortNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers, use_fc):
        super(SortNet, self).__init__()
        self.use_fc = use_fc
        self.layers = nn.Sequential(
            *[
                nn.MultiheadAttention(input_dim, num_heads)
                for _ in range(num_layers)
            ]
        )
        if self.use_fc:
            self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # MHA requires (seq_len, batch, feature)
        for layer in self.layers:
            x, _ = layer(x, x, x)
        if self.use_fc:
            output = self.fc(x)
            return output.permute(1, 0, 2)  # Return to original (batch, seq_len, feature)
        else:
            return x.permute(1, 0, 2)


def generate_dataset(length, min_val, max_val, size, device):
    unsorted_tensor = torch.randint(min_val, max_val, (size, length), device=device)
    sorted_tensor = torch.sort(unsorted_tensor, dim=1)[0]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SortNet Training')
    parser.add_argument('--min_val', nargs='+', type=int, default=[0])
    parser.add_argument('--max_val', nargs='+', type=int, default=[100])
    parser.add_argument('--length', nargs='+', type=int, default=[10])
    parser.add_argument('--num_epochs', nargs='+', type=int, default=[100])
    parser.add_argument('--use_fc', type=bool, default=True)
    parser.add_argument('--num_layers', nargs='+', type=int, default=[1])
    parser.add_argument('-p', '--plot', action='store_true', help="Plot the metrics")
    parser.add_argument('-s', '--save_plot', action='store_true', help="Save the plot")

    args = parser.parse_args()
    all_values = list(itertools.product(args.min_val, args.max_val, args.length, args.num_epochs, args.num_layers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loop = tqdm(
        all_values,
        total=len(args.min_val) * len(args.max_val) * len(args.length) * len(args.num_epochs) * len(args.num_layers)
    )

    for (min_val, max_val, length, num_epochs, num_layers) in loop:
        input_dim = output_dim = length
        num_heads = 1
        batch_size = 128
        model = SortNet(input_dim, output_dim, num_heads, num_layers, args.use_fc).to(device)
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
            plot_metrics(df, min_val, max_val, length, num_epochs, args.use_fc, args.save_plot)
