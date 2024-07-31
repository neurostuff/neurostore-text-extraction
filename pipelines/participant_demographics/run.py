""" Extract participant demographics from HTML files. """
import os
from publang.extract import extract_from_text
from openai import OpenAI
from pathlib import Path
import json
import pandas as pd

import prompts
from .clean import clean_predictions

def _run(extraction_model, extraction_client, docs, output_dir, prompt_set='', **extract_kwargs):
    short_model_name = extraction_model.split('/')[-1]

    extract_kwargs.pop('search_query', None)

    outname = f"{prompt_set}_{short_model_name}"
    predictions_path = output_dir / f'{outname}.json'
    clean_predictions_path = output_dir / f'{outname}_clean.csv'

    # Extract
    predictions = extract_from_text(
        docs['text'].to_list(),
        model=extraction_model, client=extraction_client,
        **extract_kwargs
    )

    # Add PMCID to predictions
    for i, pred in enumerate(predictions):
        pred['pmcid'] = docs['pmcid'].iloc[i]

    json.dump(predictions, open(predictions_path, 'w'))

    clean_predictions(predictions).to_csv(
        clean_predictions_path, index=False
    )


def _load_client(model_name):
    if 'gpt' in model_name:
        client = OpenAI(api_key=os.getenv('MYOPENAI_API_KEY'))

    else:
        raise ValueError(f"Model {model_name} not supported")

    return client

def _load_prompt_config(prompt_set):
    return getattr(prompts, prompt_set)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--extraction-model', type=str, required=True,
        help='The model to use for extraction.'
    )
    parser.add_argument(
        '--docs-path', type=str, required=True,
        help='The path to the JSON file containing the documents.'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='The directory to save the output files.'
    )
    parser.add_argument(
        '--prompt-set', type=str, default='ZERO_SHOT_MULTI_GROUP_FTSTRICT_FC',
        help='The prompt set to use for the extraction.'
    )

    args = parser.parse_args()

    docs = pd.read_json(args.docs_path)

    extraction_client = _load_client(args.extraction_model)

    prompt_config = _load_prompt_config(args.prompt_set)

    output_dir = Path(args.output_dir)

    _run(
        args.extraction_model, extraction_client, docs, output_dir, prompt_set=args.prompt_set,
        **prompt_config
    )