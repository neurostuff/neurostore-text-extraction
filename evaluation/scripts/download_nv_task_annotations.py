"""
Script to download nv task annotations and create a ground truth CSV file.
The CSV file is saved in the evaluation/data directory.
"""

import os
from pathlib import Path
import pandas as pd

# Get paths relative to this script
script_dir = Path(__file__).parent.resolve()
annotations_path = (script_dir.parent / "data" / "labelbuddy-annotations").resolve()
output_path = (script_dir.parent / "data").resolve()

# Set the environment variable before importing labelrepo
os.environ["LABELBUDDY_ANNOTATIONS_REPO"] = str(annotations_path)

# Import after setting environment variable
from labelrepo.projects.nv_task import load_annotations  # noqa: E402
from utils import get_annotation_summary  # noqa: E402

# Create data directory if it doesn't exist
output_path.mkdir(exist_ok=True)

# Load and filter annotations
annotations = load_annotations()
annotations = annotations[annotations["annotator_name"] != "alice_chen"]

# Process annotations into summary format
annotations_summary = get_annotation_summary(annotations)

# Identify papers to exclude and papers with/without task names
exclude_idx = [_p for _p, v in annotations_summary.items() if v["Exclude"] is not None]

has_task_name = set(
    [
        _p
        for _p, v in annotations_summary.items()
        if v["TaskName"] and ("n/a" not in v["TaskName"])
    ]
) - set(exclude_idx)

has_task_noname = set(
    [
        _p
        for _p, v in annotations_summary.items()
        if not (v["TaskName"]) or "n/a" in v["TaskName"]
    ]
) - set(exclude_idx)

# Convert summary to DataFrame for saving
ground_truth = pd.DataFrame.from_dict(annotations_summary, orient="index")

# Save detailed ground truth
ground_truth.to_csv(output_path / "nv_task_ground_truth.csv", index=True)

# Print summary statistics
print("\nGround Truth Summary:")
print(f"Total Papers: {len(annotations_summary)}")
print(f"Papers to Exclude: {len(exclude_idx)}")
print(f"Papers with Task Names: {len(has_task_name)}")
print(f"Papers without Task Names: {len(has_task_noname)}")

print("\nAnnotations per section:")
section_counts = annotations.groupby("section").size()
print(section_counts)

print("\nLabel types:")
label_counts = annotations.groupby("label_name").size()
print(label_counts)
