"""Assemble the full custom_kg payload (nodes + counted edges + source chunk).

custom_kg needs at least one chunk whose ``source_id`` matches the source_id
referenced by every entity/relationship, otherwise LightRAG logs UNKNOWN
source warnings. We attach a single build-provenance chunk.
"""

from __future__ import annotations

from .schema import Record
from .taxonomy import Taxonomy
from .nodes import build_nodes, build_taxonomy_edges, SOURCE_ID
from .aggregate import (
    CoocConfig,
    build_cooccurrence_edges,
    build_cue_edges,
)


def build_custom_kg(
    records: list[Record],
    tax: Taxonomy,
    cfg: CoocConfig | None = None,
    *,
    include_cue_edges: bool = True,
) -> dict:
    cfg = cfg or CoocConfig()

    entities = build_nodes(tax)
    relationships = build_taxonomy_edges(tax)
    relationships += build_cooccurrence_edges(records, tax, cfg)
    if include_cue_edges:
        relationships += build_cue_edges(records, tax, cfg)

    chunk = {
        "content": (
            "Layer-1 emotion concept graph. Nodes from 분류표 (taxonomy); "
            f"edges counted from {len(records)} gold-label train records "
            f"at level={cfg.level}, min_count={cfg.min_count}."
        ),
        "source_id": SOURCE_ID,
    }

    return {
        "chunks": [chunk],
        "entities": entities,
        "relationships": relationships,
    }
