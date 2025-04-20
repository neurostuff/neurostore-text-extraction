from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import pandas as pd
import json
from tqdm import tqdm

import spacy
from scispacy.candidate_generation import CandidateGenerator
from spacy.language import Language

from ns_extract.pipelines.base import DependentPipeline


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


class UMLSDiseaseExtractor(DependentPipeline):
    """Extracts UMLS disease concepts from text.

    Required Input Sources:
        - pubget or ace: For accessing text files
        - participant_demographics: For diagnosis information

    Required File Inputs:
        - text: Article full text file

    Required Pipeline Inputs:
        - participant_demographics: ["results"]
    """

    _version = "1.0.0"
    _output_schema = UMLSDiseaseSchema

    def __init__(
        self,
        inputs=("text",),
        input_sources=("pubget", "ace"),
        pipeline_inputs={"participant_demographics": ["results"]},
        k: int = 30,
        threshold: float = 0.5,
        no_definition_threshold: float = 0.95,
        filter_for_definitions: bool = True,
        max_entities_per_mention: int = 5,
        n_workers: int = 1,
    ):
        """Initialize the UMLS Disease extractor.

        Args:
            inputs: Input file types to process
            input_sources: Sources to accept inputs from
            pipeline_inputs: Dict mapping pipeline names to lists of required inputs
            k: Number of candidates to consider
            threshold: Minimum similarity threshold for matches
            no_definition_threshold: Higher threshold for concepts without definitions
            filter_for_definitions: Whether to apply stricter threshold for undefined concepts
            max_entities_per_mention: Maximum number of UMLS entities per mention
            n_workers: Number of workers for parallel processing
        """
        self.k = k
        self.threshold = threshold
        self.no_definition_threshold = no_definition_threshold
        self.filter_for_definitions = filter_for_definitions
        self.max_entities_per_mention = max_entities_per_mention
        self.n_workers = n_workers

        # Initialize spaCy and UMLS
        self.nlp = self._load_spacy_model()
        self.umls_generator = CandidateGenerator(name="umls")

        super().__init__(
            inputs=inputs, input_sources=input_sources, pipeline_inputs=pipeline_inputs
        )

    def _load_spacy_model(self):
        nlp = spacy.load("en_core_sci_sm", enable=["tokenizer"])
        nlp.add_pipe("abbreviation_detector")
        nlp.add_pipe("serialize_abbreviation", after="abbreviation_detector")
        return nlp

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
    ) -> tuple[str, list]:
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

    def execute(
        self, study_inputs: Dict[str, Any], **kwargs
    ) -> Dict[str, List[UMLSDiseaseSchema]]:
        """Run UMLS disease extraction pipeline for a study.

        Args:
            study_inputs: Dictionary containing:
                - text: Path to article full text file
                - results: Path to participant_demographics results file
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping study IDs to lists of UMLS disease results
        """
        # Get text from file
        with open(study_inputs["text"]) as f:
            text = f.read()

        # Load demographics results
        with open(study_inputs["results"]) as f:
            demographics = json.load(f)

        if not demographics or "groups" not in demographics:
            return {}

        # Process text for abbreviations
        abbreviations = self._load_abbreviations([text])[0]

        # Extract UMLS entities
        study_results = []
        for group_ix, group in enumerate(demographics["groups"]):
            if not pd.isna(group.get("diagnosis")):
                start_char = group.get("start_char")
                end_char = group.get("end_char")

                resolved_target, target_ents = self.get_candidates(
                    group["diagnosis"],
                    abbreviations=abbreviations,
                    start_char=start_char,
                    end_char=end_char,
                )

                if target_ents:  # Only add results if entities were found
                    result = UMLSDiseaseSchema(
                        pmid=group["pmid"],
                        diagnosis=resolved_target,
                        umls_entities=target_ents,
                        count=group.get("count", 1),
                        group_ix=group_ix,
                        start_char=start_char,
                        end_char=end_char,
                    ).model_dump()
                    study_results.append(result)

        return {kwargs["study_id"]: study_results} if study_results else {}
