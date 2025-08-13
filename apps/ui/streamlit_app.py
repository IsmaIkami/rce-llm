import os, re, json
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Tuple
from huggingface_hub import InferenceClient
from pint import UnitRegistry
from dateutil.parser import parse as dtparse
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# ---------------- theory objects ----------------
ureg = UnitRegistry()

@dataclass
class Atom:
    id: str
    type: str   # "entity" | "event" | "quantity" | "time"
    label: str
    attrs: Dict

@dataclass
class Relation:
    src: str
    dst: str
    rtype: str
    weight: float = 1.0

@dataclass
class PotentialGraph:
    atoms: List[Atom]
    relations: List[Relation]
    meta: Dict

# ---------------- parsers (prompt -> G^p) ----------------

CITY_LEX = {"paris","lyon","brussels","london","berlin"}

def parse_quantities(text: str) -> List[Atom]:
    ats = []
    for i, m in enumerate(re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z/\^°%]+)", text)):
        val, unit = m.group(1), m.group(2)
        unit_l = unit.lower()
        # keep speed/distance/time-ish units
        keep = any(x in unit_l for x in ["km/h","mph","km","m","cm","mm","h","s"])
        if keep:
            ats.append(Atom(id=f"q{i}", type="quantity", label=f"{val} {unit}",
                            attrs={"value": float(val), "unit": unit}))
    return ats

def parse_times(text: str) -> List[Atom]:
    ats = []
    for j, m in enumerate(re.finditer(r"(\d{1,2}h(?:\d{2})?|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2})", text)):
        ats.append(Atom(id=f"t{j}", type="time", label=m.group(1), attrs={"time": m.group(1)}))
    return ats

def parse_cities(text: str) -> List[Atom]:
    ats = []
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)
    seen = set()
    for w in words:
        wl = w.lower()
        if wl in CITY_LEX and wl not in seen:
            seen.add(wl)
            ats.append(Atom(id=f"c_{wl}", type="entity", label=w, attrs={"kind":"city"}))
    return ats

def parse_trains(text: str) -> List[Atom]:
    # create up to two generic trains T1/T2 when we see patterns "A train ... Another ..."
    trains = []
    if re.search(r"\btrain\b", text, flags=re.I):
        trains.append(Atom(id="train1", type="entity", label="Train 1", attrs={"kind":"train"}))
        if re.search(r"\banother\b|\bsecond\b", text, flags=re.I):
            trains.append(Atom(id="train2", type="entity", label="Train 2", attrs={"kind":"train"}))
    return trains

def build_relations(text: str, atoms: List[Atom]) -> List[Relation]:
    id_by_label = {a.label.lower(): a.id for a in atoms}
    id_by_type = {}
    for a in atoms:
        id_by_type.setdefault(a.type, []).append(a.id)

    rels: List[Relation] = []

    # naive from→to pattern "from X towards Y"
    m = re.search(r"\bfrom\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+(?:to|towards)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)", text, flags=re.I)
    if m and "train1" in id_by_type.get("entity", []):
        src, dst = m.group(1).lower(), m.group(2).lower()
        if f"c_{src}" in id_by_label.values() and f"c_{dst}" in id_by_label.values():
            rels.append(Relation("train1", id_by_label.get(src, f"c_{src}"), "depart_from", 1.0))
            rels.append(Relation("train1", id_by_label.get(dst, f"c_{dst}"), "towards", 1.0))

    # second leg "Another leaves Y ... towards X"
    m2 = re.search(r"\banother\b.*?\bleaves?\s+([A-Za-zÀ-ÖØ-öø-ÿ]+).*?\b(towards|to)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)", text, flags=re.I|re.S)
    if m2 and "train2" in id_by_type.get("entity", []):
        src2, dst2 = m2.group(1).lower(), m2.group(3).lower()
        if f"c_{src2}" in id_by_label.values() and f"c_{dst2}" in id_by_label.values():
            rels.append(Relation("train2", id_by_label.get(src2, f"c_{src2}"), "depart_from", 1.0))
            rels.append(Relation("train2", id_by_label.get(dst2, f"c_{dst2}"), "towards", 1.0))

    # attach times: first time to train1, second to train2
    times = [a for a in atoms if a.type=="time"]
    if times:
        rels.append(Relation("train1", times[0].id, "depart_time", 1.0))
    if len(times) >= 2 and any(a.id=="train2" for a in atoms if a.type=="entity"):
        rels.append(Relation("train2", times[1].id, "depart_time", 1.0))

    # attach speeds: first quantity that has /h to train1, next to train2
    speeds = [a for a in atoms if a.type=="quantity" and "/h" in a.attrs.get("unit","").lower()]
    if speeds:
        rels.append(Relation("train1", speeds[0].id, "speed", 1.0))
    if len(speeds) >= 2 and any(a.id=="train2" for a in atoms if a.type=="entity"):
        rels.append(Relation("train2", speeds[1].id, "speed", 1.0))

    return rels

def build_gp(text: str) -> PotentialGraph:
    atoms = []
    atoms += parse_trains(text)
    atoms += parse_cities(text)
    atoms += parse_times(text)
    atoms += parse_quantities(text)
    rels = build_relations(text, atoms)
    return PotentialGraph(atoms=atoms, relations=rels, meta={"text": text})

# ---------------- constraints & μ ----------------

def units_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    msgs = []
    ok = 1.0
    for r in gp.relations:
        if r.rtype == "speed":
            q = next(a for a in gp.atoms if a.id == r.dst)
            try:
                _ = ureg(q.attrs["value"]) * ureg(q.attrs["unit"])
                if "/h" not in q.attrs["unit"].lower():
                    ok = min(ok, 0.6); msgs.append(f"speed unit suspicious: {q.label}")
            except Exception:
                ok = min(ok, 0.5); msgs.append(f"unknown unit: {q.label}")
    return ok, msgs

def time_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    msgs, s = [], 1.0
    for r in gp.relations:
        if r.rtype == "depart_time":
            t = next(a for a in gp.atoms if a.id == r.dst)
            try: _ = dtparse(str(t.attrs["time"]))
            except Exception:
                s = min(s, 0.5); msgs.append(f"bad time: {t.label}")
    return s, msgs

def typing_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    sig = {"depart_from":("entity","entity"), "towards":("entity","entity"),
           "depart_time":("entity","time"), "speed":("entity","quantity")}
    msgs = []; s = 1.0
    by_id = {a.id:a for a in gp.atoms}
    for r in gp.relations:
        if r.rtype in sig:
            st, dt = by_id[r.src].type, by_id[r.dst].type
            if (st,dt) != sig[r.rtype]:
                s = min(s, 0.0); msgs.append(f"type mismatch {r.src}-{r.rtype}->{r.dst}")
    return s, msgs

def mu(gp: PotentialGraph) -> Tuple[float, Dict[str, float], List[str]]:
    u, mu_u = units_ok(gp); t, mu_t = time_ok(gp); ty, mu_ty = typing_ok(gp)
    # weights (tweakable)
    w = {"units":0.4, "time":0.2, "typing":0.4}
    score = w["units"]*u + w["time"]*t + w["typing"]*ty
    breakdown = {"units":u, "time":t, "typing":ty}
    msgs = mu_u + mu_t + mu_ty
    return score, breakdown, msgs

# ---------------- Ω actualization (greedy) ----------------
def actualize(gp: PotentialGraph) -> PotentialGraph:
    # tiny greedy: keep all nodes; drop relations that violate typing/time hard
    kept = []
    by_id = {a.id:a for a in gp.atoms}
    for r in gp.relations:
        if r.rtype == "depart_time":
            t = by_id[r.dst]
            try: dtparse(str(t.attrs["time"])); keep_time = True
            except Exception: keep_time = False
            if not keep_time: continue
        if r.rtype in {"depart_from","towards"}:
            if by_id[r.src].type != "entity" or by_id[r.dst].type != "entity": continue
        if r.rtype == "speed":
            q = by_id[r.dst]
            if "/h" not in q.attrs.get("unit","").lower(): continue
        kept.append(r)
    return PotentialGraph(atoms=gp.atoms, relations=kept, meta=gp.meta | {"omega": True})

# ---------------- visualization ----------------
def gp_to_pyvis(gp: PotentialGraph, highlight_omega: bool=False) -> str:
    G = nx.MultiDiGraph()
    for a in gp.atoms:
        G.add_node(a.id, label=f"{a.label}\n[{a.type}]", type=a.type)
    for r in gp.relations:
        G.add_edge(r.src, r.dst, label=r.rtype)

    net = Network(height="500px", width="100%", directed=True, bgcolor="#FFFFFF")
    color_map = {"entity":"#4F46E5","time":"#059669","quantity":"#DC2626","event":"#7C3AED"}
    for nid, data in G.nodes(data=True):
        net.add_node(nid, label=data["label"], color=color_map.get(data.get("type",""), "#64748B"))
    for u,v,data in G.edges(data=True):
        net.add_edge(u,v, label=data.get("label",""))

    # export to HTML
    net.repulsion(node_distance=160, spring_length=160)
    return net.generate_html()

def show_graph(title: str, gp: PotentialGraph):
    st.subheader(title)
    html = gp_to_pyvis(gp)
    components.html(html, height=520, scrolling=True)

# ---------------- candidate alignment ----------------
def lexicalizations(gp: PotentialGraph) -> List[str]:
    lex = []
    for a in gp.atoms:
        if a.type in {"entity","time","quantity"}:
            lex.append(str(a.label).lower())
    for r in gp.relations:
        lex.append(r.rtype.replace("_"," "))
    return list(dict.fromkeys(lex))

def alignment_score(text: str, gp: PotentialGraph) -> float:
    t = text.lower()
    lex = lexicalizations(gp)
    if not lex: return 1.0
    hits = sum(1 for x in lex if x in t)
    return hits / len(lex)

# ---------------- HF generation (robust) ----------------
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")

def generate_candidates(prompt: str, n: int = 2, max_new_tokens: int = 256) -> List[str]:
    if not HF_TOKEN:
        return [
            "I will compute the meeting point step by step, checking units and times.",
            "They meet roughly halfway along the route, depending on relative speeds and departure times."
        ]
    client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
    outs = []
    for _ in range(n):
        text = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        outs.append(str(text))
    return outs

# ---------------- streamlit ui ----------------
st.set_page_config(page_title="RCE LLM — Graph MVP", layout="wide")
st.title("RCE LLM — Graph MVP")

prompt = st.text_area(
    "enter a question/task",
    "A train leaves Paris at 15h towards Lyon at 300 km/h. Another leaves Lyon at 14:30 towards Paris at 250 km/h. Where do they meet?",
    height=110,
)

if st.button("run"):
    # 1) build potential graph G^p from prompt
    gp = build_gp(prompt)
    score_gp, br_gp, msgs_gp = mu(gp)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("potential graph Gᵖ (emergent)")
        st.json([a.__dict__ for a in gp.atoms])
        st.caption("relations")
        st.json([r.__dict__ for r in gp.relations])
        st.markdown(f"**μ(Gᵖ)** = {round(score_gp,3)}  |  breakdown: {br_gp}")
    with c2:
        show_graph("Gᵖ visualization", gp)

    # 2) actualize Ω*
    omega = actualize(gp)
    score_om, br_om, msgs_om = mu(omega)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("actualized context Ω*")
        st.json([r.__dict__ for r in omega.relations])
        st.markdown(f"**μ(Ω*)** = {round(score_om,3)}  |  breakdown: {br_om}")
        if msgs_om: st.warning("notes: " + "; ".join(msgs_om))
    with c4:
        show_graph("Ω* visualization (kept edges)", omega)

    st.markdown("---")
    st.subheader("generation (HF model) + RCE rerank")

    # 3) generate candidates
    candidates = generate_candidates(prompt, n=2)

    # 4) score each by μ on Ω* + alignment coverage
    table = []
    best_idx, best_val = 0, -1
    for i, y in enumerate(candidates):
        gp_y = build_gp(y)          # optional: build graph of the answer text
        mu_y, br_y, _ = mu(omega)   # evaluate coherence on Ω* (policy choice)
        cov = alignment_score(y, omega)
        val = 0.7*mu_y + 0.3*cov     # combine
        table.append((y, round(mu_y,3), round(cov,3), round(val,3)))
        if val > best_val: best_val, best_idx = val, i

    st.markdown("**best candidate (by μ + coverage):**")
    st.write(table[best_idx][0])
    st.caption(f"μ={table[best_idx][1]}  |  coverage={table[best_idx][2]}  |  score={table[best_idx][3]}")

    with st.expander("all candidates (with μ & coverage)"):
        for y, m, cov, val in table:
            st.write("---")
            st.write(y)
            st.caption(f"μ={m}  |  coverage={cov}  |  score={val}")
