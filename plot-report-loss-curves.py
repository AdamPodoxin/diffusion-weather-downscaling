import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path


def plot_and_save(df: pd.DataFrame, model_name: str, save_path: Path):
    grid = sns.FacetGrid(df, col="model", height=4, aspect=1)
    grid.map(sns.lineplot, "epoch", "train loss")
    grid.map(sns.lineplot, "epoch", "validation loss", linestyle="--", color="orange")
    grid.figure.suptitle(f"Train and validation losses for {model_name}s")
    grid.figure.subplots_adjust(top=0.85)
    grid.set_titles("{col_name}")
    grid.figure.axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # train_line = plt.Line2D([], [], color="blue", linestyle="-", label="Train loss")
    # val_line = plt.Line2D([], [], color="orange", linestyle="--", label="Validation loss")
    # grid.figure.legend(handles=[train_line, val_line], loc="upper center", ncol=2)

    plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    sns.set_theme()

    vqvae_vanilla_loss_df = pd.read_csv("models/vqvae-trained-vanilla/losses/losses.csv")
    vqvae_vanilla_loss_df["model"] = "Standard VQVAE"

    vqvae_psd_loss_df = pd.read_csv("models/vqvae-trained-psd/losses/losses.csv")
    vqvae_psd_loss_df["model"] = "PSD-VQVAE"

    full_vqvae_df = pd.concat([vqvae_vanilla_loss_df, vqvae_psd_loss_df])
    plot_and_save(full_vqvae_df, "VQVAE", Path("plots/report_loss_curves/vqvae.png"))

    unet_vanilla_loss_df = pd.read_csv("models/unet-trained-vanilla/losses/losses.csv")
    unet_vanilla_loss_df["model"] = "Standard UNet"

    unet_psd_loss_df = pd.read_csv("models/unet-trained-psd/losses/losses.csv")
    unet_psd_loss_df["model"] = "PSD-UNet"

    full_unet_df = pd.concat([unet_vanilla_loss_df, unet_psd_loss_df])
    plot_and_save(full_unet_df, "UNet", Path("plots/report_loss_curves/unet.png"))
