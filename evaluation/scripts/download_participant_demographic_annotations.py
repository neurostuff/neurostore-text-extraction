"""
Script to download participant demographic annotations and create a ground truth CSV file.
The CSV file is saved in the evaluation/data directory.
"""

import os
from pathlib import Path

from labelrepo.projects.participant_demographics import get_participant_demographics

import pandas as pd

# Get paths relative to this script
script_dir = Path(__file__).parent.resolve()
annotations_path = (script_dir.parent / "data" / "labelbuddy-annotations").resolve()
output_path = (script_dir.parent / "data").resolve()

# Create data directory if it doesn't exist
output_path.mkdir(exist_ok=True)

# Set the environment variable
os.environ["LABELBUDDY_ANNOTATIONS_REPO"] = str(annotations_path)

subgroups = get_participant_demographics()

top_raters = ["Jerome_Dockes", "kailu_song", "calvin_surbey", "joon_hong", "ju-chi_yu"]

combined = []
for pmcid, group in subgroups.groupby("pmcid"):
    chosen = False
    for rater in top_raters:
        if rater in group.annotator_name.values:
            combined.append(group[group.annotator_name == rater])
            chosen = True
            break

    if not chosen:
        # If not choose rating with most counts
        chosen_rater = group.groupby("annotator_name")["count"].sum().idxmax()
        combined.append(group[group.annotator_name == chosen_rater])

combined = pd.concat(combined)

combined.to_csv(output_path / "participant_demographics_ground_truth.csv")
