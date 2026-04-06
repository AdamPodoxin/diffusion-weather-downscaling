from pathlib import Path
import pandas as pd


SAVE_PATH = Path("evaluation/test_loss_stats/normalized_test_loss_stats.csv")

if __name__ == "__main__":
    vanilla_df = pd.read_csv("evaluation/losses/vanilla.csv")
    vanilla_df["pipeline"] = "standard-LDM (custom)"

    vanilla_noise_df = pd.read_csv("evaluation/losses/vanilla_noise.csv")
    vanilla_noise_df["pipeline"] = "standard-LDM (recommended)"

    psd_df = pd.read_csv("evaluation/losses/psd.csv")
    psd_df["pipeline"] = "PSD-LDM (custom)"

    psd_noise_df = pd.read_csv("evaluation/losses/psd_noise.csv")
    psd_noise_df["pipeline"] = "PSD-LDM (recommended)"

    full_df = pd.concat([vanilla_df, vanilla_noise_df, psd_df, psd_noise_df])
    normalized_df = full_df[full_df["normalized"] == "normalized"]

    stats_df = normalized_df \
                .groupby(["pipeline", "channel", "loss_type"])["loss_value"] \
                .aggregate(["mean", "sem"])
    
    stats_df.to_csv(SAVE_PATH)
