__REQUIRES__ = 'participant_demographics'

import spacy
from scispacy.candidate_generation import CandidateGenerator
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from ns_pipelines.pipeline import IndependentPipeline

from spacy.language import Language
@Language.component("serialize_abbreviation")
def replace_abbrev_with_json(spacy_doc):
    # https://github.com/allenai/scispacy/issues/205#issuecomment-597273144
    # Code is modified to work as a component
    new_abbrevs = []
    for short in spacy_doc._.abbreviations:
        short_text = short.text
        short_start = short.start
        short_end = short.end
        long = short._.long_form
        long_text = long.text
        long_start = long.start
        long_end = long.end
        serializable_abbr = {"short_text": short_text, "short_start": short_start, "short_end": short_end,
                             "long_text": long_text, "long_start": long_start, "long_end": long_end}
        short._.long_form = None
        new_abbrevs.append(serializable_abbr)
    spacy_doc._.abbreviations = new_abbrevs
    return spacy_doc

def _load_spacy_model():
    nlp = spacy.load("en_core_sci_sm", enable=["tokenizer"])
    nlp.add_pipe("abbreviation_detector")
    nlp.add_pipe("serialize_abbreviation", after="abbreviation_detector")

    return nlp

def _resolve_abbreviations(target, abbreviations, start_char=None, end_char=None):
    for abrv in abbreviations:
        if abrv['short_text'] in target:
            # If start and end char are provided, only resolve abbreviations within the target
            if start_char is not None and end_char is not None:
                if not (abrv['start_char'] >= start_char and abrv['end_char'] <= end_char):
                    continue 
            target = target.replace(abrv['short_text'], abrv['long_text'])

    return target

def get_candidates(
        generator, target, abbreviations=None, start_char=None, end_char=None,
        k=30, threshold=0.5, no_definition_threshold=0.95, filter_for_definitions=True, max_entities_per_mention=5):
    """ Given a text and a target, return the UMLS entities that match the target
    Takes advantage of abbreciation detection from full text and entity linking to UMLS.
    """
    # First we need to resolve abbreciation in the target text
    if abbreviations is not None:
        target = _resolve_abbreviations(target, abbreviations, start_char=start_char, end_char=end_char)

    # Second we can use the CandidateGenerator to get the UMLS entities
    candidates = generator([target], k)[0]
    predicted = []
    for cand in candidates:
        score = max(cand.similarities)
        if (
            filter_for_definitions
            and generator.kb.cui_to_entity[cand.concept_id].definition is None
            and score < no_definition_threshold
        ):
            continue
        if score > threshold:
            name = cand.canonical_name if hasattr(cand, 'canonical_name') else cand.aliases[0]
            predicted.append((cand.concept_id, name, score))
    sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x[2])
    return target, sorted_predicted[: max_entities_per_mention]


def run_umls_extraction(preds, abbreviations=None):
    generator = CandidateGenerator(name='umls')

    print("Extracting UMLS entities")
    results = []
    for doc_preds in tqdm(preds):
        for ix, pred in enumerate(doc_preds['preds']):
            # Get the UMLS entities that match the targettarg
            start_char = pred['start_char'] if 'start_char' in pred else None
            end_char = pred['end_char'] if 'end_char' in pred else None
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                abrvs = abbreviations[ix] if abbreviations is not None else None
                resolved_target, target_ents = get_candidates(
                    generator, pred['diagnosis'], start_char=start_char, end_char=end_char,
                    abbreviations=abrvs)

                for ent in target_ents:
                    results.append({
                        "pmid": int(doc_preds['pmid']),
                        "diagnosis": resolved_target,
                        "umls_cui": ent[0],
                        "umls_name": ent[1],
                        "umls_prob": ent[2],
                        "count": pred['count'],
                        "group_ix": ix,
                        "start_char": start_char,
                        "end_char": end_char,
                    })

    return results


def _load_abbreviations(docs, n_workers=1, batch_size=20):
    nlp = _load_spacy_model()
    abbreviations = []
    print("Processing abbreviations")
    for i in tqdm(range(0, len(docs), batch_size)):
        batch_docs = docs[i:i+batch_size]
        batch_abbreviations = nlp.pipe(batch_docs, n_process=n_workers)
        for processed_doc in batch_abbreviations:
            abbreviations.append(processed_doc._.abbreviations)
    return abbreviations


def _load_preds(preds_path, docs_path=None, load_abbreviations=True, n_workers=1):
    """ Load the documents and predictions dataframes. 
    Both CSV files should have a 'pmid' column.
    They will be loaded into a single dictionary with the pmid as the key.
    """

    if load_abbreviations:
        if docs_path is None:
            raise ValueError(
                "Abbreviations require the documents to be loaded. provide the path to the documents CSV."
                )
        texts = pd.read_csv(docs_path)
    else:
        texts = None    

    preds = pd.read_csv(preds_path)
    
    combined = []
    docs = []

    for pmid, row in preds.groupby('pmid'):
        combined.append(
            {
                'pmid': pmid,
                'preds': row.to_dict(orient='records')
            }
        )

        if texts is not None:        
            docs.append(texts[texts['pmid'] == pmid]['body'].values[0])

    if load_abbreviations:
        abbreviations = _load_abbreviations(docs, n_workers)
    else:
        abbreviations = None

    return combined, abbreviations

def __main__(docs_path, preds_path, replace_abreviations=True, output_dir=None, n_workers=1, **kwargs):
    """ Run the participant demographics extraction pipeline. 

    Args:
        docs_path (str): The path to the csv file containing the documents.
        preds_path (str): The path to the csv file containing the participant demographics predictions.
        replace_appreviations (bool): Whether to replace abbreviations in the text before running the extraction.
        output_dir (str): The directory to save the output files.
        **kwargs: Additional keyword arguments to pass to the extraction function.
    """

    # Refactor to replace abbrevations while loading
    preds, abbreviations = _load_preds(preds_path, docs_path, replace_abreviations, n_workers)

    if abbreviations is not None and output_dir is not None:
        out_name = Path(preds_path).stem.replace('_clean', '_abrv')
        out_path = Path(output_dir) / f'{out_name}.json'
        json.dump(abbreviations, out_path.open('w'))

    results = run_umls_extraction(preds, abbreviations=abbreviations)

    results_df = pd.DataFrame(results)

    if output_dir is not None:
        out_name = Path(preds_path).stem.replace('_clean', '_umls')
        out_path = Path(output_dir) / f'{out_name}.csv'
        results_df.to_csv(out_path, index=False)

    return results_df
