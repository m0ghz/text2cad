import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(metrics_file: Path, output_image: Path = None):
    """
    Reads a metrics.jsonl file and plots key metrics over steps.
    """
    if not metrics_file.exists():
        print(f"Error: File not found at {metrics_file}")
        return

    try:
        df = pd.read_json(metrics_file, lines=True)
        if df.empty:
            print("File is empty.")
            return

        # Define metrics to plot
        metrics_to_plot = [
            ("env/all/judge/score", "Judge Score (0-5)"),
            ("env/all/compile/success", "Compilation Success Rate (0-1)"),
            ("env/all/reward/total", "Total Reward"),
            ("optim/kl_sample_train_v1", "KL Divergence")
        ]

        # Filter only existing columns
        available_metrics = [(col, label) for col, label in metrics_to_plot if col in df.columns]

        if not available_metrics:
            print("No recognizable metrics found to plot.")
            print("Available columns:", df.columns.tolist())
            return

        # Create subplots
        num_metrics = len(available_metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=True)
        
        if num_metrics == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, available_metrics):
            ax.plot(df["step"], df[col], marker='o', linestyle='-', label=label)
            ax.set_ylabel(label)
            ax.set_title(f"{label} over Training Steps")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.xlabel("Step")
        plt.tight_layout()

        if output_image:
            plt.savefig(output_image)
            print(f"Plot saved to {output_image}")
        else:
            plt.show()

    except Exception as e:
        print(f"An error occurred while plotting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from jsonl file.")
    parser.add_argument("file_path", type=Path, help="Path to the metrics.jsonl file")
    parser.add_argument("--output", "-o", type=Path, default=Path("training_progress.png"), help="Output path for the plot image")
    args = parser.parse_args()

    plot_metrics(args.file_path, args.output)

