""" Extract participant demographics from articles. """

from contextlib import nullcontext
import logging

from .prompts import base_message
from .schemas import BaseDemographicsSchema
import pandas as pd
import numpy as np

from ns_extract.pipelines.api import APIPromptExtractor


class ParticipantDemographicsExtractor(APIPromptExtractor):
    """Participant demographics extraction pipeline."""

    _version = "1.1.0"
    _prompt = base_message
    _output_schema = BaseDemographicsSchema

    def post_process(self, results, study_inputs, **kwargs):
        # Clean known issues with GPT demographics result
        cleaned_results = super().post_process(results, study_inputs, **kwargs)
        for study_id, study_result in cleaned_results.items():
            meta_keys = ["pmid", "rank", "start_char", "end_char", "id"]
            meta_keys = [k for k in meta_keys if k in study_result]

            # Convert JSON to DataFrame
            df = pd.json_normalize(study_result, record_path=["groups"], meta=meta_keys)
            if df.empty:
                logging.warning(f"No groups found for study {study_id}")
                cleaned_results[study_id] = {}
                continue

            df.columns = df.columns.str.replace(" ", "_")

            # Check if the option exists in pandas
            has_no_silent_downcast = hasattr(pd.options, "future") and hasattr(
                pd.options.future, "no_silent_downcasting"
            )

            ctx = (
                pd.option_context("future.no_silent_downcasting", True)
                if has_no_silent_downcast
                else nullcontext()
            )
            with ctx:
                df = df.fillna(value=np.nan).infer_objects(copy=False)
            df["group_name"] = df["group_name"].fillna("healthy")

            # Drop rows where count is NA
            df = df[~pd.isna(df["count"])]

            # Set group_name to healthy if no diagnosis
            df.loc[
                (df["group_name"] != "healthy") & (pd.isna(df["diagnosis"])),
                "group_name",
            ] = "healthy"

            # Ensure minimum count is 0
            df["count"] = df["count"].clip(lower=0)

            # If no male count, substract count from female count columns
            ix_male_miss = (pd.isna(df["male_count"])) & ~(pd.isna(df["female_count"]))
            df.loc[ix_male_miss, "male_count"] = (
                df.loc[ix_male_miss, "count"] - df.loc[ix_male_miss, "female_count"]
            )

            df["male_count"] = df["male_count"].clip(lower=0)

            # Same for female count
            ix_female_miss = (pd.isna(df["female_count"])) & ~(
                pd.isna(df["male_count"])
            )
            df.loc[ix_female_miss, "female_count"] = (
                df.loc[ix_female_miss, "count"] - df.loc[ix_female_miss, "male_count"]
            )

            df["female_count"] = df["female_count"].clip(lower=0)

            # Replace missing values with None
            df = df.astype(object).where(pd.notna(df), None)
            df = df.where(pd.notnull(df), None)
            cleaned_results[study_id] = {"groups": df.to_dict(orient="records")}

        return cleaned_results
