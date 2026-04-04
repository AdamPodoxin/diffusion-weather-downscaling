import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Plot training loss curve for model")

    p.add_argument(
        "--input",
        type=Path,
        default=Path("models/vqvae-trained-vanilla/losses/losses.csv"),
        help="Losses csv file",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plots/train_loss_curves/vqvae-trained-vanilla.png"),
        help="Output image file",
    )
    p.add_argument(
        "--model",
        type=str,
        default="4-Channel Vanilla VQVAE",
        help="Model name",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.input).set_index("epoch")

    best_val_loss_index = df["validation loss"].argmin()

    sns.set_theme()
    fig = sns.lineplot(data=df)
    fig.set_title(f"Losses for {args.model} model training")
    plt.axvline(best_val_loss_index, 0, 1, linestyle="--", color="green")

    fig.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    plt.savefig(args.output, bbox_inches="tight")
