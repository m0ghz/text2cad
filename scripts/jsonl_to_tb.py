import argparse
import json
import time
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def sync_jsonl_to_tensorboard(jsonl_path: Path, log_dir: Path, interval: float = 2.0, offset_step: int = 0):
    """
    Continuously reads a jsonl file and writes new entries to TensorBoard.
    """
    if not jsonl_path.exists():
        print(f"Waiting for {jsonl_path} to be created...")
        while not jsonl_path.exists():
            time.sleep(interval)

    print(f"Syncing {jsonl_path} to TensorBoard logs at {log_dir} (offset_step={offset_step})")
    
    # Create the writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    processed_lines = 0
    
    try:
        while True:
            with open(jsonl_path, 'r') as f:
                # Skip lines we've already processed
                # Note: This is a simple implementation. Ideally we'd seek(), 
                # but seeking in text files can be tricky with encoding.
                # For typical training logs (kB/MBs), reading all lines is fast enough.
                lines = f.readlines()
                
                if len(lines) > processed_lines:
                    new_lines = lines[processed_lines:]
                    
                    for line in new_lines:
                        try:
                            data = json.loads(line)
                            step = data.get('step')
                            if step is None:
                                continue
                            
                            # Apply offset
                            step += offset_step
                                
                            # Log all numeric values
                            for key, value in data.items():
                                if isinstance(value, (int, float)) and key != 'step':
                                    writer.add_scalar(key, value, step)
                                    
                        except json.JSONDecodeError:
                            continue
                            
                    processed_lines = len(lines)
                    writer.flush()
                    print(f"Updated: {len(new_lines)} new records. Total: {processed_lines}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopping sync.")
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync metrics.jsonl to TensorBoard")
    parser.add_argument("jsonl_file", type=Path, help="Path to metrics.jsonl")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for TensorBoard logs")
    parser.add_argument("--offset-step", type=int, default=0, help="Add this offset to the step number")
    
    args = parser.parse_args()
    
    if args.log_dir is None:
        # Default to a 'tb_logs' folder next to the metrics file
        args.log_dir = args.jsonl_file.parent / "tb_logs"
        
    sync_jsonl_to_tensorboard(args.jsonl_file, args.log_dir, offset_step=args.offset_step)

