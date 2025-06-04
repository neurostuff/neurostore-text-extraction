# Evaluation Framework

This directory contains the framework for evaluating text extraction against ground truth datasets.

## Participant Demographics Evaluation

### Overview
The participant demographics evaluation compares extracted demographic information to manually annotated ground truth data. Metrics are calculated for each demographic field:

- count (total participant count)
- male_count 
- female_count
- age statistics (mean, minimum, maximum, median)
- group name matching

### Directory Structure
```
evaluation/
├── data/
│   ├── participant_demographics_ground_truth.csv   # Ground truth annotations
│   ├── extractions/                               # Extractor outputs
│   │   └── test/                                  # Default test set 
│   └── results/                                   # Evaluation results
├── scripts/
│   └── run_participant_demographics_evaluation.py  # Evaluation runner
└── evaluate_participant_demographics.py           # Core evaluation logic
```

### Usage

1. Ground Truth Data
The ground truth data should be in CSV format with the following columns:
- pmcid: Study identifier
- group_name: Name of participant group (e.g. "healthy", "patients") 
- count: Total number of participants
- male_count: Number of male participants
- female_count: Number of female participants
- age_mean: Mean age
- age_minimum: Minimum age
- age_maximum: Maximum age
- age_median: Median age

2. Extractor Output
Extractor outputs should be JSON files with the following structure:
```json
{
    "pmcid": "12345",
    "groups": [
        {
            "group_name": "healthy",
            "count": 10,
            "male_count": 5,
            "female_count": 5,
            "age_mean": 25.0,
            "age_minimum": 18.0,
            "age_maximum": 35.0,
            "age_median": 24.0
        }
    ]
}
```

3. Running Evaluation

From command line:
```bash
# Run on default test set
python evaluation/scripts/run_participant_demographics_evaluation.py

# Run on specific test set
python evaluation/scripts/run_participant_demographics_evaluation.py --test-set custom_set

# Specify output directory
python evaluation/scripts/run_participant_demographics_evaluation.py --output-dir path/to/output
```

From Python:
```python
from evaluation.evaluate_participant_demographics import evaluate_extractions

results = evaluate_extractions(
    extraction_dir="evaluation/data/extractions/test",
    ground_truth_path="evaluation/data/participant_demographics_ground_truth.csv",
    output_path="evaluation/data/results/results.json"
)
```

### Output Format

The evaluation generates a JSON file with:
1. Per-study metrics
2. Overall metrics across all studies

Example output:
```json
{
    "studies": {
        "study1": {
            "count": 1.0,
            "male_count": 1.0,
            "female_count": 1.0,
            "age_mean": 0.95,
            "age_minimum": 1.0,
            "age_maximum": 0.92,
            "age_median": null,
            "group_names": 1.0
        }
    },
    "overall": {
        "count": 0.95,
        "male_count": 0.92,
        "female_count": 0.90,
        "age_mean": 0.88,
        "age_minimum": 0.93,
        "age_maximum": 0.87,
        "age_median": null,
        "group_names": 0.94
    }
}
```

### Metrics

Values indicate accuracy between 0 and 1:
- 1.0 = Perfect match
- 0.0 = No matches
- null = No data available

Numeric comparisons use a relative tolerance of 10% by default.
