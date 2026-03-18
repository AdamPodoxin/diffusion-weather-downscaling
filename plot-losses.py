import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_name = sys.argv[3]

    df = pd.read_csv(input_file).set_index("epoch")

    best_val_loss_index = df["validation loss"].argmin()

    sns.set_theme()
    fig = sns.lineplot(data=df)
    fig.set_title(f"Losses for {model_name} model training")
    plt.axvline(best_val_loss_index, 0, 1, linestyle="--", color="green")
    
    plt.savefig(output_file, bbox_inches="tight")
