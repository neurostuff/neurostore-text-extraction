__REQUIRES__ = 'participant_demographics'

import spacy
from scispacy.candidate_generation import CandidateGenerator
from scispacy.abbreviation import AbbreviationDetector
import pandas as pd
from tqdm import tqdm
import warnings
from multiprocessing import Pool

def _load_spacy_model():
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")

    generator = CandidateGenerator(name='umls')

    return nlp, generator


def get_candidates(
        generator, processed_doc, target, resolve_abbreviations=True, start_char=None, end_char=None,
        k=30, threshold=0.5, no_definition_threshold=0.95, filter_for_definitions=True, max_entities_per_mention=5):
    """ Given a text and a target, return the UMLS entities that match the target
    Takes advantage of abbreciation detection from full text and entity linking to UMLS.
    """
    generator = CandidateGenerator(name='umls')
    # First we need to resolve abbreciation in the target text
    if resolve_abbreviations:
        for abrv in processed_doc._.abbreviations:
            if abrv.text in target:
                # If start and end char are provided, only resolve abbreviations within the target
                if start_char is not None and end_char is not None:
                    if not (abrv.start_char >= start_char and abrv.end_char <= end_char):
                        continue 
                target = target.replace(abrv.text, abrv._.long_form.text)

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


def run_extraction(doc_preds):
    nlp, generator = _load_spacy_model()

    results = []
    for data in tqdm(doc_preds):
        doc_preds = data['preds']
        doc = data['body']
        pmid = data['pmid']

        processed_doc = nlp(doc['body'])
        for ix, pred in doc_preds.iterrows():
            # Get the UMLS entities that match the targettarg
            start_char = pred['start_char'] if 'start_char' in pred else None
            end_char = pred['end_char'] if 'end_char' in pred else None
            if pred['group_name'] == 'patients' and pd.isna(pred['diagnosis']) == False:
                resolved_target, target_ents = get_candidates(
                    generator, processed_doc, pred['diagnosis'], start_char=start_char, end_char=end_char)

                for ent in target_ents:
                    results.append({
                        "pmid": int(doc['pmid']),
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


def _load_docs_preds(docs_path, preds_path):
    """ Load the documents and predictions dataframes. 
    Both CSV files should have a 'pmid' column.
    They will be loaded into a single dictionary with the pmid as the key.
    """
    texts = pd.read_csv(docs_path)
    texts = dict(zip(texts['pmid'], texts['body']))

    preds = pd.read_csv(preds_path)
    
    combined = []
    for pmid, row in preds.groupby('pmid'):
        if pmid not in texts:
            warnings.warn(f"PMID {pmid} not found in documents, skipping")
            continue
        
        combined.append(
            {
                'pmid': pmid,
                'body': texts[pmid],
                'preds': row
            }
        )

    return combined

# Split data into chunks
def split_data(data, n_chunks):
    chunk_size = len(data) // n_chunks
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]

def __main__(docs_path, preds_path, output_dir=None, n_workers=1, **kwargs):
    """ Run the participant demographics extraction pipeline. 

    Args:
        docs_path (str): The path to the csv file containing the documents.
        preds_path (str): The path to the csv file containing the participant demographics predictions.
        output_dir (str): The directory to save the output files.
        **kwargs: Additional keyword arguments to pass to the extraction function.
    """

    doc_preds = _load_docs_preds(docs_path, preds_path)

    if n_workers > 1:
        # Split the data into chunks
        chunks = split_data(doc_preds, n_workers)

        # Create a pool of workers
        with Pool(n_workers) as pool:
            # Distribute the chunks to the workers
            results = pool.map(run_extraction, chunks)

        # Merge the results
        results = [item for sublist in results for item in sublist]

        # Print or use the merged results
        print(f"Processed {len(results)} chunks of data.")
    else:
        results = run_extraction(doc_preds)
        
    results_df = pd.DataFrame(results)

    if output_dir is not None:
        out_name = preds_path.replace('_clean', '_umls').stem
        out_path = output_dir / f'{out_name}.csv'
        results_df.to_csv(out_path, index=False)

    return results_df
