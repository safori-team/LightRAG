"""End-to-end smoke test (offline, no API key).

Flow: load taxonomy + dummy gold labels -> validate -> split train ->
build custom_kg (nodes + 선1 + cue edges) -> inject into LightRAG (file
backend, mock embedding) -> query '슬픔' neighbors -> assert the
'슬픔'+'사랑' co-occurrence edge was learned and surfaces as a 보정 candidate.

Run:  python -m emotion_layer1.smoke
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from . import taxonomy as tx
from .schema import load_jsonl, validate, split_records
from .aggregate import CoocConfig
from .build_kg import build_custom_kg
from .inject import make_rag, inject
from . import infer

DUMMY = os.path.join(os.path.dirname(__file__), "data", "dummy.jsonl")


async def main() -> int:
    tax = tx.load()
    print(f"[taxonomy] majors={len(tax.majors)} minors={len(tax.minors)} cues={len(tax.cues)}")

    records = load_jsonl(DUMMY)
    rep = validate(records, tax)
    print(f"[validate] {rep.summary()}")
    assert rep.ok, f"label validation failed: {rep.summary()}"

    train = split_records(records, "train")
    print(f"[split] train={len(train)}")

    cfg = CoocConfig(level="minor", min_count=2)
    kg = build_custom_kg(train, tax, cfg)
    cooc = [r for r in kg["relationships"] if "CO_OCCURS" in r["keywords"]]
    cue = [r for r in kg["relationships"] if "CUE_INDICATES" in r["keywords"]]
    print(
        f"[build] entities={len(kg['entities'])} "
        f"taxonomy_edges={len(kg['relationships']) - len(cooc) - len(cue)} "
        f"선1={len(cooc)} cue선={len(cue)}"
    )
    for r in cooc:
        print(f"   선1: {r['src_id']} <-> {r['tgt_id']}  w={r['weight']}  | {r['description']}")

    with tempfile.TemporaryDirectory() as wd:
        rag = await make_rag(wd)
        await inject(rag, kg)
        print("[inject] custom_kg inserted")

        graph = rag.chunk_entity_relation_graph
        cands = await infer.cooccur_candidates(graph, tax, "슬픔")
        print(f"[query] '슬픔' 선1 candidates: {[(c.emotion, c.weight) for c in cands]}")
        assert any(c.emotion == "사랑" for c in cands), "expected 슬픔<->사랑 co-occurrence edge"

        missing = await infer.suggest_missing(graph, tax, ["슬픔"], "엄마 생각이 나서 보고 싶다")
        print(f"[보정] Gemini=['슬픔'] -> missing candidates: {[(c.emotion, c.weight) for c in missing]}")
        assert any(c.emotion == "사랑" for c in missing), "expected 사랑 suggested as missing"

    print("\nSMOKE OK ✅  (선1 집계 -> 주입 -> 조회 -> 보정후보 동작 확인)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
