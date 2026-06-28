"""Node source = the 분류표 (taxonomy). Data-driven, swappable.

Loads ``taxonomy.json`` and exposes:
  * the node sets (majors / minors / cues),
  * deterministic, collision-free node ids (type-prefixed),
  * label -> node-id resolution used by the loader and aggregator.

Nodes are DEFINED here, never mined from data. Replace taxonomy.json with
your real 분류표 (same shape) and every downstream step follows.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

_HERE = os.path.dirname(__file__)
TAXONOMY_PATH = os.path.join(_HERE, "taxonomy.json")

# node-id namespaces (keep emotion nodes collision-free and clearly typed)
MAJOR_PREFIX = "EMO_MAJOR"
MINOR_PREFIX = "EMO_MINOR"
CUE_PREFIX = "CUE"

ENTITY_TYPE = {
    MAJOR_PREFIX: "EmotionMajor",
    MINOR_PREFIX: "EmotionMinor",
    CUE_PREFIX: "NonverbalCue",
}


def node_id(prefix: str, local: str) -> str:
    return f"{prefix}:{local}"


@dataclass(frozen=True)
class Taxonomy:
    majors: dict[str, dict]          # major_kr -> {en}
    minors: dict[str, dict]          # minor_kr -> {en, major, major_en}
    cues: dict[str, dict]            # cue_id   -> {}

    # ---- node id helpers -------------------------------------------------
    def major_node(self, major_kr: str) -> str:
        return node_id(MAJOR_PREFIX, major_kr)

    def minor_node(self, minor_kr: str) -> str:
        return node_id(MINOR_PREFIX, minor_kr)

    def cue_node(self, cue_id: str) -> str:
        return node_id(CUE_PREFIX, cue_id)

    # ---- validation ------------------------------------------------------
    def has_major(self, major_kr: str) -> bool:
        return major_kr in self.majors

    def has_minor(self, minor_kr: str) -> bool:
        return minor_kr in self.minors

    def has_cue(self, cue_id: str) -> bool:
        return cue_id in self.cues

    def major_of(self, minor_kr: str) -> str | None:
        m = self.minors.get(minor_kr)
        return m["major"] if m else None


def load(path: str = TAXONOMY_PATH) -> Taxonomy:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    majors = {m["id"]: {"en": m.get("en", "")} for m in raw["majors"]}
    minors = {
        m["id"]: {
            "en": m.get("en", ""),
            "major": m["major"],
            "major_en": m.get("major_en", ""),
        }
        for m in raw["minors"]
    }
    cues = {c["id"]: {} for c in raw["cues"]}
    return Taxonomy(majors=majors, minors=minors, cues=cues)
