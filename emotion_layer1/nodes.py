"""분류표 -> custom_kg nodes (and taxonomy membership edges).

Produces:
  * entities: one per major emotion, minor emotion and nonverbal cue.
  * relationships: minor --BELONGS_TO--> major (taxonomy structure, NOT counted
    from data; it is schema, so injecting it is not a data leak).

These are the graph's points. Edges learned from labels (선1 etc.) are added
separately by ``aggregate.py``.
"""

from __future__ import annotations

from .taxonomy import ENTITY_TYPE, MAJOR_PREFIX, MINOR_PREFIX, CUE_PREFIX, Taxonomy

SOURCE_ID = "layer1"  # all taxonomy nodes/edges trace to the build chunk


def build_nodes(tax: Taxonomy) -> list[dict]:
    entities: list[dict] = []
    for kr, meta in tax.majors.items():
        entities.append({
            "entity_name": tax.major_node(kr),
            "entity_type": ENTITY_TYPE[MAJOR_PREFIX],
            "description": f"감정 대분류 '{kr}' ({meta['en']}).",
            "source_id": SOURCE_ID,
        })
    for kr, meta in tax.minors.items():
        entities.append({
            "entity_name": tax.minor_node(kr),
            "entity_type": ENTITY_TYPE[MINOR_PREFIX],
            "description": f"감정 소분류 '{kr}' ({meta['en']}), 대분류 {meta['major']}.",
            "source_id": SOURCE_ID,
        })
    for cue in tax.cues:
        entities.append({
            "entity_name": tax.cue_node(cue),
            "entity_type": ENTITY_TYPE[CUE_PREFIX],
            "description": f"비언어 cue '{cue}' (통제 어휘).",
            "source_id": SOURCE_ID,
        })
    return entities


def build_taxonomy_edges(tax: Taxonomy) -> list[dict]:
    edges: list[dict] = []
    for kr, meta in tax.minors.items():
        edges.append({
            "src_id": tax.minor_node(kr),
            "tgt_id": tax.major_node(meta["major"]),
            "description": f"소분류 '{kr}'는 대분류 '{meta['major']}'에 속한다.",
            "keywords": "BELONGS_TO,taxonomy",
            "weight": 1.0,
            "source_id": SOURCE_ID,
        })
    return edges
