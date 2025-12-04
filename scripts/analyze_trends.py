import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_trends(metrics_file: Path):
    if not metrics_file.exists():
        print("Metrics file not found.")
        return

    df = pd.read_json(metrics_file, lines=True)
    if df.empty:
        print("Metrics file is empty.")
        return

    # Sort by step just in case
    df = df.sort_values('step')

    print(f"--- Analysis of {len(df)} steps ---")

    # Check Token Length Trend
    tokens = df['env/all/ac_tokens_per_turn']
    print("\n[Token Length]")
    print(tokens.head(10).to_string())
    if (tokens == 1024).all():
        print(">> ALERT: Token length hits max (1024) at ALL steps. Model is likely repeating/looping from the start.")
    elif (tokens == 1024).any():
        first_collapse = (tokens == 1024).idxmax()
        print(f">> ALERT: Model collapsed to max length at step {first_collapse}.")
    else:
        print(">> Token length seems within bounds.")

    # Check KL Divergence
    if 'optim/kl_sample_train_v1' in df.columns:
        kl = df['optim/kl_sample_train_v1']
        print("\n[KL Divergence]")
        print(kl.head(10).to_string())
        if kl.iloc[0] > 1.0:
             print(">> ALERT: High initial KL divergence. Initial policy might be too different or LR too high.")
    
    # Check Judge Score
    score = df['env/all/judge/score']
    print("\n[Judge Score]")
    print(score.head(10).to_string())
    
    # Check Rewards
    reward = df['env/all/reward/total']
    print("\n[Reward]")
    print(reward.head(10).to_string())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_trends(Path(sys.argv[1]))
    else:
        print("Usage: python script.py path/to/metrics.jsonl")

