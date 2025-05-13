from typing import List, Dict, Optional, Union, Any, Tuple

import pandas as pd
import spacy
from pydantic import BaseModel, Field
from scispacy.candidate_generation import CandidateGenerator
from spacy.language import Language
from tqdm import tqdm

from ns_extract.pipelines.base import IndependentPipeline, Extractor

# Import and register the scispacy abbreviation detector
import scispacy.abbreviation  # noqa: F401


class UMLSDiseaseSchema(BaseModel):
    """Schema for UMLS Disease extraction output for a single study"""

    pmid: int = Field(description="PubMed ID of the article")
    diagnosis: str = Field(description="Original diagnosis text")
    umls_entities: List[Dict[str, Union[str, float]]] = Field(
        description="List of UMLS entities found",
        example=[
            {
                "umls_cui": "C0011849",
                "umls_name": "Diabetes Mellitus",
                "umls_prob": 0.95,
            }
        ],
    )
    count: int = Field(description="Number of patients with this diagnosis")
    group_ix: int = Field(description="Group index from demographics")
    start_char: Optional[int] = Field(description="Start character position in text")
    end_char: Optional[int] = Field(description="End character position in text")


class BaseUMLSDiseaseSchema(BaseModel):
    groups: List[UMLSDiseaseSchema]


@Language.component("serialize_abbreviation")
def replace_abbrev_with_json(spacy_doc):
    # https://github.com/allenai/scispacy/issues/205#issuecomment-597273144
    new_abbrevs = []
    for short in spacy_doc._.abbreviations:
        short_text = short.text
        short_start = short.start
        short_end = short.end
        long = short._.long_form
        long_text = long.text
        long_start = long.start
        long_end = long.end
        serializable_abbr = {
            "short_text": short_text,
            "short_start": short_start,
            "short_end": short_end,
            "long_text": long_text,
            "long_start": long_start,
            "long_end": long_end,
        }
        short._.long_form = None
        new_abbrevs.append(serializable_abbr)
    spacy_doc._.abbreviations = new_abbrevs
    return spacy_doc


class UMLSDiseaseExtractor(Extractor, IndependentPipeline):
    """Extracts UMLS disease concepts from text using configurable spaCy language models.

    The extractor uses spaCy with scispacy components to identify and link disease mentions
    to UMLS concepts. It supports configurable language models and customizable pipeline
    components.

    Required Input Sources:
        - pubget or ace: For accessing text files
        - participant_demographics: For diagnosis information

    Required File Inputs:
        - text: Article full text file

    Required Pipeline Inputs:
        - participant_demographics: ["results"]

    Model Configuration:
        The extractor can be configured with different spaCy models through the model_name
        parameter. By default, it uses 'en_core_sci_sm', but other models like
        'en_core_sci_md' or 'en_core_sci_lg' can be used for potentially better accuracy
        at the cost of increased memory usage.

    Examples:
        # Using default small model
        extractor = UMLSDiseaseExtractor()

        # Using medium model for better accuracy
        extractor = UMLSDiseaseExtractor(model_name='en_core_sci_md')

    Pipeline Components:
        The extractor disables the 'parser' and 'ner' components by default for better
        performance. It automatically adds the 'abbreviation_detector' and
        'serialize_abbreviation' components if not already present.

    Backward Compatibility:
        The addition of the model_name parameter maintains backward compatibility with
        previous versions. Existing code using the default configuration will continue
        to work without modifications.
    """

    _version = "1.0.0"
    _output_schema = BaseUMLSDiseaseSchema
    _data_pond_inputs = {("pubget", "ace"): ("text",)}
    _input_pipelines = {("participant_demographics",): ("results",)}

    def __init__(
        self,
        k: int = 30,
        threshold: float = 0.5,
        no_definition_threshold: float = 0.95,
        filter_for_definitions: bool = True,
        max_entities_per_mention: int = 5,
        n_workers: int = 1,
        model_name: str = "en_core_sci_sm",
    ):
        """Initialize the UMLS Disease extractor.

        Args:
            k: Number of candidates to consider
            threshold: Minimum similarity threshold for matches
            no_definition_threshold: Higher threshold for concepts without definitions
            filter_for_definitions: Whether to apply stricter threshold for undefined concepts
            max_entities_per_mention: Maximum number of UMLS entities per mention
            n_workers: Number of workers for parallel processing
            model_name: Name of the spaCy model to load (default: en_core_sci_sm)
        """
        self.k = k
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention
        self.n_workers = n_workers
        self.model_name = model_name

        # Initialize spaCy and UMLS
        self.nlp = self._load_spacy_model()
        self.umls_generator = CandidateGenerator(name="umls")

        super().__init__()

    def _load_spacy_model(self):
        """Load spaCy model with abbreviation detection pipeline.

        This function handles downloading of the model if it's not already installed
        and verifies model compatibility. It also sets up required pipeline components.

        Returns:
            spacy.language.Language: Loaded spaCy model with configured pipeline

        Raises:
            ImportError: If there are issues downloading or loading the model
            ValueError: If the model is not compatible with required components
        """
        try:
            try:
                nlp = spacy.load(self.model_name, disable=["parser", "ner"])
            except OSError:
                print(f"Downloading {self.model_name} model...")
                spacy.cli.download(self.model_name)
                nlp = spacy.load(self.model_name, disable=["parser", "ner"])

            # Verify model compatibility with required components
            if "transformer" in nlp.pipe_names:
                raise ValueError(
                    f"Model {self.model_name} is a transformer model. "
                    "Please use a standard spaCy model like en_core_sci_sm/md/lg."
                )

            # Add registered components in correct order
            if "abbreviation_detector" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("abbreviation_detector")
                except Exception as e:
                    raise ValueError(
                        f"Failed to add abbreviation_detector to pipeline: {str(e)}"
                    )

            if "serialize_abbreviation" not in nlp.pipe_names:
                try:
                    nlp.add_pipe(
                        "serialize_abbreviation", after="abbreviation_detector"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to add serialize_abbreviation to pipeline: {str(e)}"
                    )

            return nlp

        except Exception as e:
            raise ImportError(
                f"Error loading spaCy model {self.model_name}: {str(e)}. "
                "Please ensure you have an internet connection and "
                "sufficient permissions to download models."
            ) from e

    def _load_abbreviations(self, texts: List[str]) -> List[List[Dict]]:
        """Process texts to extract abbreviations"""
        abbreviations = []
        print("Processing abbreviations")
        batch_size = 20
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_docs = texts[i: i + batch_size]
            batch_abbreviations = self.nlp.pipe(batch_docs, n_process=self.n_workers)
            for processed_doc in batch_abbreviations:
                abbreviations.append(processed_doc._.abbreviations)
        return abbreviations

    def _resolve_abbreviations(
        self,
        target: str,
        abbreviations: List[Dict],
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
    ) -> str:
        """Resolve abbreviations in target text"""
        for abrv in abbreviations:
            if abrv["short_text"] in target:
                if start_char is not None and end_char is not None:
                    if not (
                        abrv["start_char"] >= start_char
                        and abrv["end_char"] <= end_char
                    ):
                        continue
                target = target.replace(abrv["short_text"], abrv["long_text"])
        return target

    def get_candidates(
        self,
        target: str,
        abbreviations: Optional[List[Dict]] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
    ) -> Tuple[str, list]:
        """Get UMLS candidates for target text"""
        if abbreviations is not None:
            target = self._resolve_abbreviations(
                target, abbreviations, start_char=start_char, end_char=end_char
            )

        candidates = self.umls_generator([target], self.k)[0]
        predicted = []

        for cand in candidates:
            score = max(cand.similarities)
            if (
                self.filter_for_definitions
                and self.umls_generator.kb.cui_to_entity[cand.concept_id].definition
                is None
                and score < self.no_definition_threshold
            ):
                continue
            if score > self.threshold:
                name = (
                    cand.canonical_name
                    if hasattr(cand, "canonical_name")
                    else cand.aliases[0]
                )
                predicted.append(
                    {
                        "umls_cui": cand.concept_id,
                        "umls_name": name,
                        "umls_prob": float(score),
                    }
                )

        sorted_predicted = sorted(predicted, reverse=True, key=lambda x: x["umls_prob"])
        return target, sorted_predicted[: self.max_entities_per_mention]

    def _transform(self, inputs: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Transform input data into UMLS disease entities.

        Args:
            inputs: Dictionary containing study data with:
                - text: Article full text content
                - participant_demographics: Demographics results data
            **kwargs: Additional arguments

        Returns:
            List of UMLS disease entities found in the text
        """
        text = inputs["text"]
        demographics = inputs["participant_demographics"]

        if not demographics or "groups" not in demographics:
            return {}

        # Process text for abbreviations
        abbreviations = self._load_abbreviations([text])[0]

        # Extract UMLS entities
        study_results = []
        for group_ix, group in enumerate(demographics["groups"]):
            if not pd.isna(group.get("diagnosis")):
                resolved_target, target_ents = self.get_candidates(
                    group["diagnosis"], abbreviations=abbreviations
                )

                if target_ents:  # Only add results if entities were found
                    result = UMLSDiseaseSchema(
                        pmid=0,  # PMID not critical for UMLS extraction
                        diagnosis=resolved_target,
                        umls_entities=target_ents,
                        count=group["count"],
                        group_ix=group_ix,
                        start_char=None,
                        end_char=None,
                    ).model_dump()
                    study_results.append(result)

        return {"groups": study_results}
