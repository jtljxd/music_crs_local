import os
import json
import random
from datasets import load_dataset, concatenate_datasets


def _resolve_splits(ds, requested: list, dataset_name: str) -> list:
    """Return valid splits from ``requested``, falling back to all splits."""
    avail = list(ds.keys())
    valid = [s for s in requested if s in avail]
    if not valid:
        import logging
        logging.getLogger(__name__).warning(
            "None of the requested splits %s found in '%s' (available: %s). "
            "Falling back to all available splits.",
            requested, dataset_name, avail,
        )
        return avail
    skipped = [s for s in requested if s not in avail]
    if skipped:
        import logging
        logging.getLogger(__name__).warning(
            "Splits not found in '%s', skipping: %s", dataset_name, skipped
        )
    return valid


class UserProfileDB:
    def __init__(self,
            dataset_name: str,
            split_types: list[str],
        ):
        metadata_dataset = load_dataset(dataset_name)
        valid_splits = _resolve_splits(metadata_dataset, split_types, dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[s] for s in valid_splits])
        self.default_columns = ['user_id', 'age_group', 'gender', 'country_name']
        self.user_profiles = {item["user_id"]: item for item in metadata_concat_dataset}

    def id_to_profile(self, user_id: str):
        user_profile = self.user_profiles[user_id]
        return user_profile

    def id_to_profile_str(self, user_id: str):
        user_profile = self.user_profiles[user_id]
        profile_str = [f"{key}: {user_profile[key]}" for key in self.default_columns]
        return "\n".join(profile_str)
