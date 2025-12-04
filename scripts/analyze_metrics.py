import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_metrics(metrics_file: Path):
    """
    Reads a metrics.jsonl file and prints a summary analysis.
    """
    if not metrics_file.exists():
        print(f"Error: File not found at {metrics_file}")
        return

    print(f"--- Analyzing {metrics_file} ---")
    
    try:
        # Read the JSONL file into a pandas DataFrame
        df = pd.read_json(metrics_file, lines=True)
        
        if df.empty:
            print("File is empty.")
            return

        print(f"Total Records: {len(df)}")
        print("\n--- Column Summary ---")
        print(df.info())

        # numeric_cols = df.select_dtypes(include=['number']).columns
        # if not numeric_cols.empty:
        #     print("\n--- Numeric Statistics ---")
        #     print(df[numeric_cols].describe())
        
        # Specific analysis for expected CAD-RFT metrics if they exist
        if 'judge/score' in df.columns:
            print("\n--- Judge Score Stats ---")
            print(df['judge/score'].describe())
            
        if 'compile/success' in df.columns:
            success_rate = df['compile/success'].mean() * 100
            print(f"\n--- Compilation Success Rate ---")
            print(f"{success_rate:.2f}%")

        if 'reward' in df.columns:
             print("\n--- Reward Stats ---")
             print(df['reward'].describe())

        print("\n--- First 3 Rows ---")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"An error occurred while analyzing the file: {e}")
        # Fallback to raw line reading if pandas fails
        print("\nFalling back to raw text inspection of first 5 lines:")
        with open(metrics_file, 'r') as f:
            for _ in range(5):
                print(f.readline().strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze metrics.jsonl file.")
    parser.add_argument("file_path", type=Path, help="Path to the metrics.jsonl file")
    args = parser.parse_args()

    analyze_metrics(args.file_path)

