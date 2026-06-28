"""Layer-1 emotion concept graph builder.

Offline batch pipeline that turns gold emotion labels into a LightRAG
``custom_kg`` payload:

  * nodes  come from the *taxonomy* (분류표) — never extracted from data.
  * edges  are *counted* from gold labels (선1 emotion co-occurrence, etc.)
    with a minimum-frequency threshold; weak edges are dropped.

The taxonomy lives in ``taxonomy.json`` (seeded from the existing
emotion_engine rules) and is data-driven: swap that one file to use a
different 분류표. No LightRAG core file is touched — injection goes through
the public ``ainsert_custom_kg`` API.
"""

from . import taxonomy, schema, nodes, aggregate, build_kg  # noqa: F401
