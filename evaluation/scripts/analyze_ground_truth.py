#!/usr/bin/env python3
"""
Utility script to analyze ground truth CSV files structure.
"""
import pandas as pd
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_participant_demographics_ground_truth():
    """Load participant demographics ground truth data and return column structure."""
    ground_truth_path = os.path.join(PROJECT_ROOT, "evaluation", "data", "participant_demographics_ground_truth.csv")
    df = pd.read_csv(ground_truth_path)
    return df.columns.tolist()

def load_nv_task_ground_truth():
    """Load NV task ground truth data and return column structure."""
    ground_truth_path = os.path.join(PROJECT_ROOT, "evaluation", "data", "nv_task_ground_truth.csv")
    df = pd.read_csv(ground_truth_path)
    return df.columns.tolist()

def print_ground_truth_structures():
    """Print the column structures of both ground truth files."""
    print("Participant Demographics Ground Truth Structure:")
    print("---------------------------------------------")
    pd_cols = load_participant_demographics_ground_truth()
    for col in pd_cols:
        print(f"- {col}")
    
    print("\nNV Task Ground Truth Structure:")
    print("-----------------------------")
    nv_cols = load_nv_task_ground_truth()
    for col in nv_cols:
        print(f"- {col}")

if __name__ == "__main__":
    print_ground_truth_structures()
