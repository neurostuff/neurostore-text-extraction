from typing import List, Dict, Union
import spacy
from spacy.language import Language


def normalize_string(input_string: str) -> str:
    """Normalize a string by removing leading/trailing whitespace and converting to lowercase.
    Args:
        input_string (str): The string to normalize.
    Returns:
        str: The normalized string.
    """
    clean_string = input_string.strip().title()
    if clean_string == "":
        return None
    if clean_string == "None":
        return None
    if clean_string == "Nan":
        return None
    if clean_string == "N/A":
        return None
    if clean_string == "Null":
        return None

    return clean_string


def load_abbreviations(
    text: str, model: Union[str, Language] = "en_core_sci_sm"
) -> List[Dict]:
    """Process text to extract abbreviations using spaCy with scispacy.
    Args:
        text (str): The text to process for abbreviations
        model (Union[str, Language]): SpaCy model name or loaded model.
            Defaults to "en_core_sci_sm".
    Returns:
        List[Dict]: List of abbreviation dictionaries, each containing:
            - short_text: The abbreviated form
            - short_start: Start position of short form
            - short_end: End position of short form
            - long_text: The expanded form
            - long_start: Start position of long form
            - long_end: End position of long form
    Example:
        >>> text = "Magnetic resonance imaging (MRI) is a medical imaging technique"
        >>> abbrevs = load_abbreviations(text)
        >>> print(abbrevs[0]['short_text'])  # 'MRI'
        >>> print(abbrevs[0]['long_text'])   # 'Magnetic resonance imaging'
    """
    try:
        if isinstance(model, str):
            try:
                nlp = spacy.load(model, disable=["parser", "ner"])
            except OSError:
                print(f"Downloading {model} model...")
                spacy.cli.download(model)
                nlp = spacy.load(model, disable=["parser", "ner"])
        else:
            nlp = model

        # Add abbreviation detector if not present
        if "abbreviation_detector" not in nlp.pipe_names:
            try:
                import scispacy.abbreviation  # noqa: F401

                nlp.add_pipe("abbreviation_detector")
            except ImportError as e:
                raise ImportError(
                    f"scispacy is required for abbreviation detection: {e}"
                )

        # Process the text
        doc = nlp(text)
        abbreviations = []

        # Extract and serialize abbreviations
        for short in doc._.abbreviations:
            long = short._.long_form
            abbreviations.append(
                {
                    "short_text": short.text,
                    "short_start": short.start_char,
                    "short_end": short.end_char,
                    "long_text": long.text,
                    "long_start": long.start_char,
                    "long_end": long.end_char,
                }
            )

        return abbreviations

    except Exception as e:
        print(f"Warning: Error processing abbreviations: {e}")
        return []


def resolve_abbreviations(
    target: str,
    abbreviations: List[Dict],
) -> str:
    """Resolve abbreviations in target text using a list of known abbreviations.

    Finds and expands all abbreviations in the text, but only processes each unique
    abbreviation once (using its first occurrence).

    Args:
        target (str): The text string that may contain abbreviations
        abbreviations (List[Dict]): List of abbreviation dictionaries from load_abbreviations()
    Returns:
        str: Text with abbreviations expanded to their full forms
    Example:
        >>> text = "The MRI showed abnormal MRI results. Both EEG and MRI indicated..."
        >>> abbrevs = load_abbreviations(
        ...     "Magnetic resonance imaging (MRI) and electroencephalogram (EEG)..."
        ... )
        >>> expanded = resolve_abbreviations(text, abbrevs)
        >>> print(expanded)
        >>> # Result: First MRI expanded, EEG expanded, subsequent MRIs unchanged
    """
    if not target or not abbreviations:
        return target

    # Track which abbreviations we've already processed
    processed_abbrevs = set()
    result = target

    # Find all abbreviations that appear in the target
    matching_abbrevs = [
        abrv
        for abrv in abbreviations
        if abrv["short_text"] in target and abrv["short_text"] not in processed_abbrevs
    ]

    # Process each unique abbreviation (first occurrence only)
    for abrv in matching_abbrevs:
        short_form = abrv["short_text"]
        if short_form not in processed_abbrevs:
            result = result.replace(short_form, abrv["long_text"])
            processed_abbrevs.add(short_form)

    return result
