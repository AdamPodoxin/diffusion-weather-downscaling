import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure
from matplotlib.axes import Axes
from pathlib import Path


SAVE_PATH = Path("plots/test_losses")

PIPELINE_ORDER = ["vanilla", "vanilla (noise latents)", "psd", "psd (noise latents)"]
PIPELINE_TITLE_MAP = {
    "vanilla": "standard-LDM\n(custom)",
    "vanilla (noise latents)": "standard-LDM\n(recommended)", 
    "psd": "PSD-LDM\n(custom)", 
    "psd (noise latents)": "PSD-LDM\n(recommended)"
}

CHANNEL_ORDER = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]


if __name__ == "__main__":
    sns.set_theme(context="talk")

    vanilla_df = pd.read_csv("evaluation/losses/vanilla.csv")
    vanilla_df["pipeline"] = "vanilla"

    vanilla_noise_df = pd.read_csv("evaluation/losses/vanilla_noise.csv")
    vanilla_noise_df["pipeline"] = "vanilla (noise latents)"

    psd_df = pd.read_csv("evaluation/losses/psd.csv")
    psd_df["pipeline"] = "psd"

    psd_noise_df = pd.read_csv("evaluation/losses/psd_noise.csv")
    psd_noise_df["pipeline"] = "psd (noise latents)"

    full_df = pd.concat([vanilla_df, vanilla_noise_df, psd_df, psd_noise_df])
    normalized_df = full_df[full_df["normalized"] == "normalized"]
    normalized_mse_df = normalized_df[normalized_df["loss_type"] == "MSE"]

    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(36, 16),
        constrained_layout=True,
    )

    for ax, channel in zip(axes, CHANNEL_ORDER):
        data = normalized_mse_df[normalized_mse_df["channel"] == channel]

        ax: Axes = ax
        ax.set_title(channel, fontsize=24)

        sns.boxplot(
            ax=ax,
            data=data,
            x="pipeline",
            hue="pipeline",
            y="loss_value",
            showfliers=False,
            order=PIPELINE_ORDER,
        )

        ax.set_xticklabels([PIPELINE_TITLE_MAP[p] for p in PIPELINE_ORDER])

        lowest_median = data.groupby("pipeline")["loss_value"].median().min()
        ax.axhline(lowest_median, linestyle="--", color="black", alpha=0.5)
    
    fig.suptitle("MSE loss per channel for each pipeline", fontsize=32)

    plt.savefig("plots/presentation_test_losses/normalized_losses.png", bbox_inches="tight")
