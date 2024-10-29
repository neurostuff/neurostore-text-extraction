# neurostore-text-extraction

This repository contains pipelines and scripts for extracting features from text using Natural Language Processing (NLP), Large Language Models (LLMs), 
and other algorithms across thousands of articles in the NeuroStore database.

## Installation

To install the necessary dependencies, run:

    pip install -r requirements.txt


## Usage
### Running pipelines
Executable workflows in `pipelines/{pipeline_name}/run.py` will take as input standardized pubget-style text inputs (row row per article).


Run all available pipelines and harmonize outputs using CLI (todo)


### Pipeline outputs
Pipeline results are output to `data/outputs/{input_hash}/{pipeline_name}/{arghash-timestamp}`.
Outputs include extracted features `features.csv`, feature descriptions `descriptions.json`, and extraction information `info.json`.

Pipeline outputs are not stored as part of this repository.
See `ns-text-extraction-outputs` sub repository. 
