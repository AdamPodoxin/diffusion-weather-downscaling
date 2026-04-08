from pathlib import Path
import pandas as pd


SAVE_PATH = Path("evaluation/vqvae_loss_stats/normalized_vqvae_loss_stats.csv")

if __name__ == "__main__":
    vanilla_df = pd.read_csv("evaluation/vqvae_losses/vanilla.csv")
    vanilla_df["model"] = "Standard VQVAE"

    psd_df = pd.read_csv("evaluation/vqvae_losses/psd.csv")
    psd_df["model"] = "PSD-VQVAE"

    full_df = pd.concat([vanilla_df, psd_df])
    normalized_df = full_df[full_df["normalized"] == "normalized"]

    stats_df = normalized_df \
                .groupby(["model", "channel", "loss_type"])["loss_value"] \
                .aggregate(["median", "std"])
    
    stats_df.to_csv(SAVE_PATH)
