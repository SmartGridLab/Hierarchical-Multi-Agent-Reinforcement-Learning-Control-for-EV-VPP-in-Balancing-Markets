"""
Main training entry point.
Run with: `python Main.py`
"""

from training.train import train


if __name__ =="__main__":
    # Start training and collect the main returned artifacts.
    ag ,all_rewards ,perf ,ep_data ,work_dir =train ()
