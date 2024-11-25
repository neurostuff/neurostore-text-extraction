# ns-text-extraction-workflows

This repository contains pipelines and scripts for extracting features from text using Natural Language Processing (NLP), Large Language Models (LLMs), 
and other algorithms across thousands of articles in the NeuroStore database.

## Installation

To install the necessary dependencies, run:

    pip install -r requirements.txt


## Usage

### Overview

Executable workflows in `pipelines/{pipeline_name}/run.py` will have a specific class that implements the `run` method.
The `run` method will take a `Dataset` object and an output directory as input, and will output extracted features to the output directory in the following format:

        # the pipeline info file contains configuration information about the pipeline
        output_dir/{pipeline_name}/{pipeline_version}/{input_hash}/pipeline_info.json
        # the study results file contains whatever extracted features from the study by the pipeline
        output_dir/{pipeline_name}/{pipeline_version}/{input_hash}/{study_id}/results.json
        # the study info file contains metadata about the inputs to the pipeline
        output_dir/{pipeline_name}/{pipeline_version}/{input_hash}/{study_id}/info.json

You will need to create a dataset object that contains the studies you want to process, and then pass that dataset object to the `run` method of the pipeline class.

Run all available pipelines and harmonize outputs using CLI (todo)

Pipelines can either be "dependent" or "independent".
Dependent pipelines are those whose outputs for each individual study depend on the outputs of other studies.
Independent pipelines are those whose outputs for each individual study do not depend on the outputs of other studies.

## Note(s) for self

#### Each study is independently processed

1) scenario 1: nothing changed
2) scenario 2: a study was added
3) scenario 3: a study was changed

`info.json` in the output directory
increment (value): 0
date: 2021-09-01

ns-pond: no hashing
we will hash based on the inputs to the pipeline and then store the hash in the info.json in the output directory.

have a place for the raw output of the API/external service. 
raw.json
and clean.json
clean function for a pipeline output, that can be used to clean the output of a pipeline

#### Each study is processed in the context of all other studies

Have a dev version
only include openaccess papers
pipeline name plus version then hash runs
pipeline/v1.0.0/hash_run-01

the hash is just the hash of the pipeline config


independent studies: copy over the studies that have been processed and havent been changed
independent studies: re-run the pipeline on studies that have been changed


## Notes

# study independent results:
/pipline_name/v1.0.0/conf-#000A/run-01/study-01/input.json
                                      /study-02/input.json
                                      /results.json

/pipline_name/v1.0.0/conf-#000A/run-02/study-03/ 

# study dependent results:
/pipline_name/v1.0.0/#sbqA_run-01/study-01
                                 /study-02
/pipline_name/v1.0.0/#sbqA_run-02/study-01
                                 /study-02
                                 /study-03

Re-Run study independent pipeline:
1. Update with new - create new directory with only updated studies
2. Force re-run for a given set of inputs (from a particular directory, we are not using inheritance here)

Re-Run study dependent pipeline:
1. Re-run all


after update:
database.study_results_table
id, study, conf, run:
0   01      #000A, 01
1   02      #000A, 01
2   03      #000A, 02


after re-run:
database.study_results_table
id, study, conf, run:
0   01      #000A, 01
1   02      #000A, 01
2   03      #000A, 02
3   01      #000A, 02
4   02      #000A, 02

## Tf-idf gets it's own unique table
## participant demographics get their own unique table


## have a table for feature names?
database.study_results_values_table
id, study_results_table_fk, feature(name), value, certainty


database.pipeline_table
id, pipline_name, pipline_description, version, study_dependent?, ace_compatiable?, pubget_compat?, Derivative
0, gpt3_embed, wat, 1.0.0, False, True, True, False
1, HDBSCABN, wat, 1.0.0, True, False, False, True
2, TF-IDF, wat, 1.0.0, True, False, True, False
3, embed_and_HDBSCAN, wat, 1.0.0, True, True, True, False

database.pipeline_configs_table
id, pipline_fk, configuration, configuration_hash, 
0, 0, {use_cheap_option: true}, #000A
1, 1, {dimensions: 10}, #XXXX

database.pipeline_run_table
id, pipline_fk, config_hash_fk, run_index, description, date


## TODO: how do I represent results in the database?
