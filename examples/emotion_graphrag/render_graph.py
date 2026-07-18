"""Render the built emotion graph to a single shareable HTML (vis-network).

No extra deps: reads the graphml, colours nodes by major category, styles
belongs_to vs co_occurs edges, embeds everything into one HTML file that opens
in any browser (vis-network is loaded from CDN).

Run (from this directory):  python render_graph.py
Output: emotion_graph.html
"""

from __future__ import annotations

import json
import os

import networkx as nx

import schema

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHML = os.path.join(HERE, "rag_storage", "emotion",
                       "graph_chunk_entity_relation.graphml")
OUT = os.path.join(HERE, "emotion_graph.html")

MAJOR_COLOR = {
    "HAPPY": "#F1C40F", "SAD": "#3498DB", "ANGRY": "#E74C3C",
    "ANXIETY": "#9B59B6", "SURPRISE": "#E67E22", "NEUTRAL": "#95A5A6",
}
MAJOR_KR = {m: schema.TAXONOMY[m]["kr"] for m in schema.TAXONOMY}


def major_of(node: str, attrs: dict) -> str:
    if attrs.get("entity_type") == "EmotionMajor" or node in schema.TAXONOMY:
        return node
    return schema.SUB2MAJOR.get(node, "NEUTRAL")


def build():
    g = nx.read_graphml(GRAPHML)

    nodes = []
    for n, a in g.nodes(data=True):
        maj = major_of(n, a)
        is_major = a.get("entity_type") == "EmotionMajor" or n in schema.TAXONOMY
        nodes.append({
            "id": n,
            "label": n,
            "title": a.get("description", n),
            "color": MAJOR_COLOR.get(maj, "#bbbbbb"),
            "shape": "hexagon" if is_major else "dot",
            "size": 30 if is_major else 14,
            "font": {"size": 20 if is_major else 12,
                     "bold": bool(is_major)},
            "group": maj,
        })

    edges = []
    for s, t, a in g.edges(data=True):
        kind = a.get("keywords", "")
        w = float(a.get("weight", 1.0))
        if kind == "belongs_to":
            edges.append({"from": s, "to": t, "color": {"color": "#d0d0d0"},
                          "width": 1, "dashes": True, "title": "belongs_to"})
        else:  # co_occurs
            edges.append({
                "from": s, "to": t,
                "color": {"color": "#2c7fb8", "opacity": min(1.0, 0.3 + w)},
                "width": 1 + w * 7,
                "title": f"co_occurs · weight={w:.2f} · {a.get('description','')}",
            })
    return nodes, edges


HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>감정 GraphRAG — 지식그래프</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{{margin:0;height:100%;font-family:sans-serif;background:#1e1e1e;color:#eee}}
  #net{{width:100%;height:100vh}}
  #legend{{position:absolute;top:12px;left:12px;background:#2b2b2bdd;padding:10px 14px;
           border-radius:8px;font-size:13px;line-height:1.7}}
  #legend b{{font-size:14px}}
  .sw{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:middle}}
  .hint{{position:absolute;bottom:12px;left:12px;background:#2b2b2bdd;padding:8px 12px;
         border-radius:8px;font-size:12px;color:#bbb}}
</style></head><body>
<div id="legend"><b>감정 지식그래프</b> (노드 {n_nodes} · 엣지 {n_edges})<br>{legend}
  <div style="margin-top:6px;color:#bbb">━ 실선(굵기=weight): 감정 공존(co_occurs)<br>┄ 점선: 소속(belongs_to)</div>
</div>
<div id="net"></div>
<div class="hint">노드 호버=한글 설명 · 엣지 호버=weight/근거 · 드래그/휠로 이동·확대</div>
<script>
  const nodes = new vis.DataSet({nodes});
  const edges = new vis.DataSet({edges});
  new vis.Network(document.getElementById('net'), {{nodes, edges}}, {{
    physics:{{stabilization:true, barnesHut:{{gravitationalConstant:-6000, springLength:130}}}},
    interaction:{{hover:true, tooltipDelay:100}},
    nodes:{{borderWidth:0}},
    edges:{{smooth:{{type:'continuous'}}}}
  }});
</script></body></html>"""


def main():
    nodes, edges = build()
    legend = "".join(
        f'<span class="sw" style="background:{c}"></span>{m} {MAJOR_KR[m]}<br>'
        for m, c in MAJOR_COLOR.items()
    )
    html = HTML.format(
        n_nodes=len(nodes), n_edges=len(edges), legend=legend,
        nodes=json.dumps(nodes, ensure_ascii=False),
        edges=json.dumps(edges, ensure_ascii=False),
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT}  ({len(nodes)} nodes, {len(edges)} edges)")


if __name__ == "__main__":
    main()
