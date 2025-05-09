from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional
from ns_extract.pipelines.base import DependentPipeline, Extractor
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFSchema(BaseModel):
    """Schema for TFIDF output"""

    terms: List[str] = Field(description="List of terms in vocabulary")
    tfidf_scores: Dict[str, float] = Field(
        description="Dictionary mapping terms to their TF-IDF scores"
    )


class TFIDFExtractor(Extractor, DependentPipeline):
    """TFIDF extraction pipeline.

    Calculates TF-IDF scores for each document's terms.
    """

    _version = "1.0.0"
    _output_schema = TFIDFSchema
    _data_pond_inputs = {("pubget", "ace"): ("text", "metadata")}
    _pipeline_inputs = {}

    def __init__(
        self,
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
        super().__init__()

    def get_text_content(self, text: str, metadata: dict) -> str:
        """Get text content based on text_type setting.

        Args:
            text: Full text content
            metadata: Metadata dictionary containing abstract

        Returns:
            Text content to use for TF-IDF calculation
        """
        abstract = metadata.get("abstract", "")

        if self.text_type == "full_text":
            return text
        elif self.text_type == "abstract":
            return abstract
        else:  # both
            return f"{abstract}\n{text}"

    def _transform(self, processed_inputs: dict, **kwargs) -> dict:
        """Run the TFIDF extraction pipeline.

        Args:
            processed_inputs: Dictionary containing all study inputs
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping study IDs to their TFIDF scores
        """
        # Process all texts
        study_texts = {}
        for study_id, study_inputs in processed_inputs.items():
            text = study_inputs["text"]
            metadata = study_inputs["metadata"]
            content = self.get_text_content(text, metadata)
            study_texts[study_id] = content

        # Get list of all texts in same order as study IDs
        study_ids = list(study_texts.keys())
        texts = [study_texts[study_id] for study_id in study_ids]

        # Calculate TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        # Create output dictionary matching schema format
        study_tfidf_scores = {}
        for idx, study_id in enumerate(study_ids):
            # Get scores for this document
            doc_scores = tfidf_matrix[idx].toarray()[0]

            # Create dictionary of term -> score for non-zero entries
            term_scores = {
                term: float(score)  # Convert numpy float to Python float
                for term, score in zip(feature_names, doc_scores)
                if score > 0
            }

            study_tfidf_scores[study_id] = {
                "terms": list(term_scores.keys()),
                "tfidf_scores": term_scores,
            }

        return study_tfidf_scores
