from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
from ns_extract.pipelines.base import DependentPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import json


class TFIDFSchema(BaseModel):
    """Schema for TFIDF output"""

    terms: List[str] = Field(description="List of terms in vocabulary")
    tfidf_scores: Dict[str, float] = Field(
        description="Dictionary mapping terms to their TF-IDF scores"
    )


class TFIDFExtractor(DependentPipeline):
    """TFIDF extraction pipeline.

    Calculates TF-IDF scores for each document's terms.
    """

    _version = "1.0.0"
    _output_schema = TFIDFSchema

    def __init__(
        self,
        inputs=("text", "metadata"),
        input_sources=("pubget", "ace"),
        min_df=2,
        text_type: Literal["full_text", "abstract", "both"] = "full_text",
        vocabulary: Optional[Dict[str, int]] = None,
        custom_terms: Optional[List[str]] = None,
    ):
        """Initialize the TFIDF extractor.

        Args:
            inputs: Input file types to process
            input_sources: Sources to accept inputs from
            min_df: Minimum document frequency for terms
            text_type: Which text to use for TF-IDF calculation:
                      'full_text' - use only the full text
                      'abstract' - use only the abstract
                      'both' - concatenate abstract and full text
            vocabulary: Custom vocabulary dict mapping terms to indices
            custom_terms: List of terms to include in vocabulary
                        (alternative to vocabulary dict)
        """
        self.min_df = min_df
        self.text_type = text_type
        self.vocabulary = vocabulary
        self.custom_terms = custom_terms

        # Convert custom_terms list to vocabulary dict if provided
        if custom_terms is not None:
            self.vocabulary = {term: idx for idx, term in enumerate(custom_terms)}

        self.vectorizer = TfidfVectorizer(min_df=min_df, vocabulary=self.vocabulary)
        super().__init__(inputs=inputs, input_sources=input_sources)

    def get_text_content(self, text_file: str, metadata_file: str) -> str:
        """Get text content based on text_type setting.

        Args:
            text_file: Path to text file containing full text
            metadata_file: Path to metadata file containing abstract

        Returns:
            Text content to use for TF-IDF calculation
        """
        with open(text_file, "r") as f:
            full_text = f.read()

        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            abstract = metadata.get("abstract", "")

        if self.text_type == "full_text":
            return full_text
        elif self.text_type == "abstract":
            return abstract
        else:  # both
            return f"{abstract}\n{full_text}"

    def execute(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the TFIDF extraction pipeline.

        Args:
            processed_inputs: Dictionary containing all study inputs
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping study IDs to their TFIDF scores
        """
        # Load all texts
        study_texts = {}
        for study_id, study_inputs in processed_inputs.items():
            text_file = study_inputs["text"]
            metadata_file = study_inputs["metadata"]
            content = self.get_text_content(text_file, metadata_file)
            study_texts[study_id] = content

        # Get list of all texts in same order as study IDs
        study_ids = list(study_texts.keys())
        texts = [study_texts[study_id] for study_id in study_ids]

        # Calculate TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Create output dictionary
        study_tfidf_scores = {}
        for idx, study_id in enumerate(study_ids):
            # Get scores for this document
            doc_scores = tfidf_matrix[idx].toarray()[0]

            # Create dictionary of term -> score for non-zero entries
            term_scores = {
                term: score
                for term, score in zip(feature_names, doc_scores)
                if score > 0
            }

            study_tfidf_scores[study_id] = {
                "terms": list(term_scores.keys()),
                "tfidf_scores": term_scores,
            }

        return study_tfidf_scores
