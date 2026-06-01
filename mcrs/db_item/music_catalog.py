"""Music catalog database that loads track metadata from a HuggingFace dataset.

Supported metadata fields:
    track_id, ISRC, track_name, artist_name, album_name,
    tag_list, popularity, release_date, duration, artist_id, album_id
"""
import logging
import os
import json
from typing import Dict, List, Optional, Any

from datasets import load_dataset, concatenate_datasets

logger = logging.getLogger(__name__)


def _resolve_splits(ds, requested: List[str], dataset_name: str) -> List[str]:
    """Return valid splits from ``requested``, falling back to all splits if needed."""
    avail = list(ds.keys())
    valid = [s for s in requested if s in avail]
    if not valid:
        logger.warning(
            "None of the requested splits %s found in '%s' (available: %s). "
            "Falling back to all available splits.",
            requested, dataset_name, avail,
        )
        return avail
    skipped = [s for s in requested if s not in avail]
    if skipped:
        logger.warning(
            "Splits not found in '%s', skipping: %s", dataset_name, skipped
        )
    return valid


# All 11 metadata fields required by the task
ALL_TRACK_FIELDS = [
    "track_id",
    "ISRC",
    "track_name",
    "artist_name",
    "album_name",
    "tag_list",
    "popularity",
    "release_date",
    "duration",
    "artist_id",
    "album_id",
]

# Fields used for the text representation shown to the LLM
DEFAULT_CORPUS_TYPES = ["track_name", "artist_name", "album_name"]


class MusicCatalogDB:
    """Accessor for track metadata loaded from a Hugging Face dataset.

    All 11 standard fields are stored for every track:
        track_id, ISRC, track_name, artist_name, album_name,
        tag_list, popularity, release_date, duration, artist_id, album_id

    The ``corpus_types`` parameter controls which fields are included in
    the short text string returned by :py:meth:`id_to_metadata` (used by
    the LLM for response generation).
    """

    def __init__(
        self,
        dataset_name: str,
        split_types: List[str],
        corpus_types: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            dataset_name: Hugging Face dataset identifier.
            split_types: Dataset splits to load and concatenate.
            corpus_types: Metadata fields included in the short text
                representation. Defaults to
                ``["track_name", "artist_name", "album_name"]``.
        """
        self.corpus_types = corpus_types or DEFAULT_CORPUS_TYPES
        metadata_dataset = load_dataset(dataset_name)
        valid_splits = _resolve_splits(metadata_dataset, split_types, dataset_name)
        metadata_concat_dataset = concatenate_datasets(
            [metadata_dataset[s] for s in valid_splits]
        )
        # Store the full row; missing columns will be absent from the dict
        self.metadata_dict: Dict[str, Dict[str, Any]] = {
            item["track_id"]: item for item in metadata_concat_dataset
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def id_to_raw(self, track_id: str) -> Dict[str, Any]:
        """Return the raw metadata dict for a track.

        All available fields are returned (see :data:`ALL_TRACK_FIELDS`).
        Returns an empty dict if the track_id is unknown.
        """
        return self.metadata_dict.get(track_id, {})

    def id_to_metadata(self, track_id: str) -> str:
        """Return a short text representation for the LLM.

        Includes ``track_id`` plus the fields listed in ``self.corpus_types``.

        Args:
            track_id: Track identifier.

        Returns:
            A comma-separated ``field: value`` string, e.g.
            ``"track_id: T123, track_name: Shape of You, artist_name: Ed Sheeran"``.
        """
        metadata = self.metadata_dict.get(track_id, {})
        entity_str = f"track_id: {track_id}"
        for corpus_type in self.corpus_types:
            value = metadata.get(corpus_type, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            else:
                value = str(value).lower() if value is not None else ""
            entity_str += f", {corpus_type}: {value}"
        return entity_str

    def id_to_full_metadata_str(self, track_id: str) -> str:
        """Return a detailed text representation with ALL 11 fields.

        Useful for prompting or logging.
        """
        metadata = self.metadata_dict.get(track_id, {})
        lines = []
        for field in ALL_TRACK_FIELDS:
            value = metadata.get(field, "N/A")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"{field}: {value}")
        return "\n".join(lines)

    def get_field(self, track_id: str, field: str, default: Any = None) -> Any:
        """Retrieve a single metadata field for a track.

        Args:
            track_id: Track identifier.
            field: One of the standard metadata field names.
            default: Value returned when the field or track is missing.
        """
        return self.metadata_dict.get(track_id, {}).get(field, default)

    def __len__(self) -> int:
        return len(self.metadata_dict)

    def __contains__(self, track_id: str) -> bool:
        return track_id in self.metadata_dict
