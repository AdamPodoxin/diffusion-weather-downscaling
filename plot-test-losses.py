import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import SubFigure
from matplotlib.axes import Axes
from pathlib import Path


SAVE_PATH = Path("plots/test_loss_plots")

PIPELINE_ORDER = ["vanilla", "vanilla (noise latents)", "psd", "psd (noise latents)"]
CHANNEL_ORDER = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "mean_sea_level_pressure"]
LOSS_ORDER = ["MSE", "PSD"]


def plot_losses(df: pd.DataFrame, title: str, save_path: Path):
    fig = plt.figure(
        constrained_layout=True,
        figsize=(24, 12),
    )
    fig.suptitle(title, fontsize=20)

    subfigs: list[SubFigure] = fig.subfigures(
        nrows=2,
        ncols=1,
    )

    for subfig_index, subfig in enumerate(subfigs):
        loss_type = LOSS_ORDER[subfig_index]
        subfig.suptitle(f"{loss_type} Loss")

        axes: list[Axes] = subfig.subplots(nrows=1, ncols=len(CHANNEL_ORDER))

        for channel_index, channel_name in enumerate(CHANNEL_ORDER):
            ax = axes[channel_index]
            ax.set_title(f"Channel: {channel_name}")
            ax.ticklabel_format(style="plain")

            data = df[(df["channel"] == channel_name) & (df["loss_type"] == loss_type)]

            sns.boxplot(
                ax=ax,
                data=data,
                x="pipeline",
                hue="pipeline",
                y="loss_value",
                showfliers=False,
                order=PIPELINE_ORDER,
            )
    
    plt.savefig(save_path)


if __name__ == "__main__":
    sns.set_theme()

    vanilla_df = pd.read_csv("evaluation/losses/vanilla.csv")
    vanilla_df["pipeline"] = "vanilla"

    vanilla_noise_df = pd.read_csv("evaluation/losses/vanilla_noise.csv")
    vanilla_noise_df["pipeline"] = "vanilla (noise latents)"

    psd_df = pd.read_csv("evaluation/losses/psd.csv")
    psd_df["pipeline"] = "psd"

    psd_noise_df = pd.read_csv("evaluation/losses/psd_noise.csv")
    psd_noise_df["pipeline"] = "psd (noise latents)"

    full_df = pd.concat([vanilla_df, vanilla_noise_df, psd_df, psd_noise_df])
    denormalized_df = full_df[full_df["normalized"] == "denormalized"]
    normalized_df = full_df[full_df["normalized"] == "normalized"]

    plot_losses(
        df=denormalized_df, 
        title="Losses for denormalized outputs",
        save_path=SAVE_PATH / "denormalized.png",
    )

    plot_losses(
        df=normalized_df, 
        title="Losses for normalized outputs",
        save_path=SAVE_PATH / "normalized.png",
    )
