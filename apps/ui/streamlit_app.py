# apps/ui/streamlit_app.py
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components

from pint import UnitRegistry
from dateutil.parser import parse as dtparse
from huggingface_hub import InferenceClient

import networkx as nx
from pyvis.network import Network

# ================================
# Theory objects
# ================================
ureg = UnitRegistry()

@dataclass
class Atom:
    id: str
    type: str   # "entity" | "event" | "quantity" | "time" | "class" | "location"
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

# ================================
# Normalization & Lexicons
# ================================
UNIT_ALIASES = {
    "km/h": "kilometer/hour",
    "kmh": "kilometer/hour",
    "kph": "kilometer/hour",
    "km/hr": "kilometer/hour",
    "km per hour": "kilometer/hour",
    "m/s": "meter/second",
    "ms-1": "meter/second",
    "mps": "meter/second",
    "mph": "mile/hour",
    "h": "hour",
    "hr": "hour",
    "s": "second",
}
BASIC_UNIT_MAP = {
    "km": "kilometer",
    "m": "meter",
    "mi": "mile",
    "sec": "second",
    "s": "second",
}

def normalize_unit(u: str) -> str:
    u0 = (u or "").strip().lower()
    if u0 in UNIT_ALIASES:
        return UNIT_ALIASES[u0]
    if u0 in BASIC_UNIT_MAP:
        return BASIC_UNIT_MAP[u0]
    # tolerate missing spaces around slash
    if "/" in u0 and " / " not in u0:
        u0 = u0.replace("/", " / ")
    return u0

TAXONOMY_LEX = {
    "cat": ["animal", "mammal", "feline"],
    "dog": ["animal", "mammal", "canine"],
    "sparrow": ["animal", "bird"],
    "paris": ["city", "location"],
}

# ================================
# Parsers (prompt -> G^p)
# ================================
def parse_quantities(text: str) -> List[Atom]:
    ats: List[Atom] = []
    for i, m in enumerate(re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z/\^°%]+)", text)):
        val, unit_raw = m.group(1), m.group(2)
        unit_norm = normalize_unit(unit_raw)
        keep = any(x in unit_norm for x in [
            "kilometer/hour", "mile/hour", "meter/second", "kilometer", "meter", "mile", "hour", "second"
        ])
        if keep:
            ats.append(
                Atom(
                    id=f"q{i}",
                    type="quantity",
                    label=f"{val} {unit_raw}",
                    attrs={"value": float(val), "unit": unit_norm, "unit_raw": unit_raw},
                )
            )
    return ats

def parse_times(text: str) -> List[Atom]:
    ats: List[Atom] = []
    for j, m in enumerate(re.finditer(r"(\d{1,2}h(?:\d{2})?|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2})", text)):
        ats.append(Atom(id=f"t{j}", type="time", label=m.group(1), attrs={"time": m.group(1)}))
    return ats

def parse_locations(text: str) -> List[Atom]:
    """
    Very light location detection:
    - Any capitalized word is a location candidate (quick demo; replace with NER later).
    """
    ats: List[Atom] = []
    seen = set()
    for w in re.findall(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]+\b", text):
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            ats.append(Atom(id=f"loc_{wl}", type="location", label=w, attrs={"kind": "place"}))
    return ats

def parse_agents(text: str) -> List[Atom]:
    """
    Detect two moving agents generically:
    - "A <noun> ... Another ..."  → agent1, agent2 (label agent1 by noun)
    - Else if motion cues present → generic agent1/agent2
    """
    ats: List[Atom] = []
    m = re.search(r"\b(?:a|an)\s+([A-Za-z\-]+)\b.*?\b(?:another|second)\b", text, flags=re.I|re.S)
    if m:
        label1 = m.group(1).capitalize()
        ats.append(Atom(id="agent1", type="entity", label=label1, attrs={"kind":"agent"}))
        ats.append(Atom(id="agent2", type="entity", label="Agent 2", attrs={"kind":"agent"}))
        return ats
    if re.search(r"\b(leaves?|depart|from|towards|to)\b", text, flags=re.I):
        ats.append(Atom(id="agent1", type="entity", label="Agent 1", attrs={"kind":"agent"}))
        ats.append(Atom(id="agent2", type="entity", label="Agent 2", attrs={"kind":"agent"}))
    return ats

def parse_is_a(text: str):
    atoms: List[Atom] = []
    rels: List[Relation] = []
    m = re.search(r"\b(?:a|an)?\s*([A-Za-z\-]+)\s+is\s+(?:a|an)?\s*([A-Za-z\-]+)", text, flags=re.I)
    if not m:
        return atoms, rels
    subj, comp = m.group(1).lower(), m.group(2).lower()
    a_subj = Atom(id=f"e_{subj}", type="entity", label=subj, attrs={"kind":"thing"})
    a_cls  = Atom(id=f"cls_{comp}", type="class", label=comp, attrs={"kind":"class"})
    atoms += [a_subj, a_cls]
    rels.append(Relation(src=a_subj.id, dst=a_cls.id, rtype="is_a", weight=1.0))
    if subj in TAXONOMY_LEX:
        for sup in TAXONOMY_LEX[subj]:
            sid = f"cls_{sup}"
            atoms.append(Atom(id=sid, type="class", label=sup, attrs={"kind":"class"}))
            rels.append(Relation(src=a_subj.id, dst=sid, rtype="is_a", weight=0.9))
    return atoms, rels

def build_relations_motion(text: str, atoms: List[Atom]) -> List[Relation]:
    """
    Generic motion relations:
      agent -> depart_from location
      agent -> towards location
      agent -> depart_time time
      agent -> speed quantity
    """
    id_by_label = {a.label.lower(): a.id for a in atoms}
    id_by_type: Dict[str, List[str]] = {}
    for a in atoms:
        id_by_type.setdefault(a.type, []).append(a.id)

    rels: List[Relation] = []

    # "from X to/towards Y" (first occurrence → agent1)
    m1 = re.search(r"\bfrom\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+(?:to|towards)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)", text, flags=re.I)
    if m1 and "agent1" in id_by_type.get("entity", []):
        src, dst = m1.group(1).lower(), m1.group(2).lower()
        loc_src = id_by_label.get(src, f"loc_{src}")
        loc_dst = id_by_label.get(dst, f"loc_{dst}")
        rels.append(Relation("agent1", loc_src, "depart_from", 1.0))
        rels.append(Relation("agent1", loc_dst, "towards", 1.0))

    # second leg "... another ... leaves Y ... towards X" → agent2
    m2 = re.search(r"\b(?:another|second)\b.*?\bleaves?\s+([A-Za-zÀ-ÖØ-öø-ÿ]+).*?\b(?:towards|to)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)",
                   text, flags=re.I|re.S)
    if m2 and "agent2" in id_by_type.get("entity", []):
        src2, dst2 = m2.group(1).lower(), m2.group(2).lower()
        loc_src2 = id_by_label.get(src2, f"loc_{src2}")
        loc_dst2 = id_by_label.get(dst2, f"loc_{dst2}")
        rels.append(Relation("agent2", loc_src2, "depart_from", 1.0))
        rels.append(Relation("agent2", loc_dst2, "towards", 1.0))

    # first/second time & speed
    times = [a for a in atoms if a.type == "time"]
    if times and "agent1" in id_by_type.get("entity", []):
        rels.append(Relation("agent1", times[0].id, "depart_time", 1.0))
    if len(times) >= 2 and "agent2" in id_by_type.get("entity", []):
        rels.append(Relation("agent2", times[1].id, "depart_time", 1.0))

    speeds = [a for a in atoms if a.type == "quantity" and ("/" in a.attrs.get("unit","") or "hour" in a.attrs.get("unit","") or "second" in a.attrs.get("unit",""))]
    if speeds and "agent1" in id_by_type.get("entity", []):
        rels.append(Relation("agent1", speeds[0].id, "speed", 1.0))
    if len(speeds) >= 2 and "agent2" in id_by_type.get("entity", []):
        rels.append(Relation("agent2", speeds[1].id, "speed", 1.0))

    return rels

def build_gp(text: str) -> PotentialGraph:
    atoms: List[Atom] = []
    rels: List[Relation] = []

    atoms += parse_agents(text)
    atoms += parse_locations(text)
    atoms += parse_times(text)
    atoms += parse_quantities(text)

    a_tax, r_tax = parse_is_a(text)
    atoms += a_tax
    rels += r_tax

    rels += build_relations_motion(text, atoms)
    return PotentialGraph(atoms=atoms, relations=rels, meta={"text": text})

# ================================
# Quantity helpers (robust)
# ================================
def make_quantity(value: float, unit_str: str):
    """
    Robustly construct a pint quantity from (value, unit_str).
    Tries multiple normalizations before giving up.
    """
    candidates = []
    u0 = normalize_unit(unit_str)
    candidates.append(u0)
    # also try replacing ' per ' with '/'
    candidates.append(u0.replace(" per ", " / "))
    # and collapsing spaces around slashes
    candidates.append(u0.replace(" / ", "/"))
    # final: expand short forms if any remain
    for short, full in BASIC_UNIT_MAP.items():
        if u0 == short:
            candidates.append(full)

    last_exc = None
    for u in candidates:
        try:
            return value * ureg(u)
        except Exception as e:
            last_exc = e
            continue
    raise last_exc if last_exc else ValueError(f"Bad unit: {unit_str}")

# ================================
# Constraints & μ
# ================================
def units_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    msgs: List[str] = []
    ok = 1.0
    by_id = {a.id: a for a in gp.atoms}
    for r in gp.relations:
        if r.rtype == "speed":
            q = by_id[r.dst]
            try:
                qty = make_quantity(q.attrs["value"], q.attrs["unit"])
                # must be a velocity (dimension L/T)
                if not qty.check("[length] / [time]"):
                    ok = min(ok, 0.5); msgs.append(f"not a speed unit: {q.label}")
                # heuristic: prefer per-hour or per-second units (fine either way)
                u = q.attrs["unit"]
                if ("hour" not in u) and ("second" not in u) and ("/" not in u):
                    ok = min(ok, 0.7); msgs.append(f"suspicious speed unit: {q.label}")
            except Exception:
                ok = min(ok, 0.5); msgs.append(f"unknown unit: {q.attrs.get('unit_raw', q.label)}")
    return ok, msgs

def time_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    msgs: List[str] = []
    s = 1.0
    by_id = {a.id: a for a in gp.atoms}
    for r in gp.relations:
        if r.rtype == "depart_time":
            t = by_id[r.dst]
            try:
                _ = dtparse(str(t.attrs["time"]))
            except Exception:
                s = min(s, 0.5); msgs.append(f"bad time: {t.label}")
    return s, msgs

def typing_ok(gp: PotentialGraph) -> Tuple[float, List[str]]:
    sig = {
        "depart_from": ("entity","location"),
        "towards": ("entity","location"),
        "depart_time": ("entity","time"),
        "speed": ("entity","quantity"),
        "is_a": ("entity","class"),
    }
    msgs: List[str] = []
    s = 1.0
    by_id = {a.id: a for a in gp.atoms}
    for r in gp.relations:
        if r.rtype in sig:
            st, dt = by_id[r.src].type, by_id[r.dst].type
            if (st, dt) != sig[r.rtype]:
                s = min(s, 0.0); msgs.append(f"type mismatch {r.src}-{r.rtype}->{r.dst}")
    return s, msgs

def mu(gp: PotentialGraph) -> Tuple[float, Dict[str, float], List[str]]:
    u, mu_u = units_ok(gp)
    t, mu_t = time_ok(gp)
    ty, mu_ty = typing_ok(gp)
    weights = {"units": 0.4, "time": 0.2, "typing": 0.4}
    score = weights["units"]*u + weights["time"]*t + weights["typing"]*ty
    breakdown = {"units": u, "time": t, "typing": ty}
    msgs = mu_u + mu_t + mu_ty
    return score, breakdown, msgs

# ================================
# Ω actualization (greedy, task-aware)
# ================================
def detect_task(text: str) -> str:
    if re.search(r"\b[aA]n?\s+[A-Za-z\-]+\s+is\s+", text): return "taxonomy"
    if re.search(r"\b(leaves?|depart|from|towards|to|km/h|mph|m/s)\b", text, flags=re.I): return "motion"
    return "general"

def actualize(gp: PotentialGraph) -> PotentialGraph:
    task = detect_task(gp.meta.get("text",""))
    kept: List[Relation] = []
    by_id = {a.id: a for a in gp.atoms}
    for r in gp.relations:
        # universal type checks
        if r.rtype in {"depart_from","towards"}:
            if by_id[r.src].type != "entity" or by_id[r.dst].type != "location": continue
        if r.rtype == "depart_time":
            try: dtparse(str(by_id[r.dst].attrs.get("time"))); keep_time = True
            except Exception: keep_time = False
            if not keep_time: continue
        if r.rtype == "speed":
            try:
                qty = make_quantity(by_id[r.dst].attrs["value"], by_id[r.dst].attrs["unit"])
                if not qty.check("[length] / [time]"):  # drop non-velocity quantities
                    continue
            except Exception:
                continue

        if task == "taxonomy" and r.rtype == "is_a":
            kept.append(r)
        elif task == "motion" and r.rtype in {"depart_from","towards","depart_time","speed"}:
            kept.append(r)
        elif task == "general":
            kept.append(r)

    return PotentialGraph(atoms=gp.atoms, relations=kept, meta=gp.meta | {"omega": True, "task": task})

# ================================
# Visualization
# ================================
def gp_to_pyvis(gp: PotentialGraph) -> str:
    G = nx.MultiDiGraph()
    for a in gp.atoms:
        G.add_node(a.id, label=f"{a.label}\n[{a.type}]", type=a.type)
    for r in gp.relations:
        G.add_edge(r.src, r.dst, label=r.rtype)

    net = Network(height="500px", width="100%", directed=True, bgcolor="#FFFFFF")
    color_map = {"entity":"#4F46E5","time":"#059669","quantity":"#DC2626","event":"#7C3AED","class":"#0EA5E9","location":"#16A34A"}
    for nid, data in G.nodes(data=True):
        net.add_node(nid, label=data["label"], color=color_map.get(data.get("type",""), "#64748B"))
    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data.get("label",""))
    net.repulsion(node_distance=160, spring_length=160)
    return net.generate_html()

def show_graph(title: str, gp: PotentialGraph):
    st.subheader(title)
    html = gp_to_pyvis(gp)
    components.html(html, height=520, scrolling=True)

# ================================
# Rendering & Alignment
# ================================
def lexicalizations(gp: PotentialGraph) -> List[str]:
    lex = []
    for a in gp.atoms:
        if a.type in {"entity","time","quantity","class","location"}:
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

# ----- helpers for motion numeric solver -----
def parse_clock_to_hours(t: str) -> Optional[float]:
    try:
        dt = dtparse(str(t))
        return dt.hour + dt.minute/60.0 + dt.second/3600.0
    except Exception:
        m = re.match(r"^(\d{1,2})h(\d{2})?$", str(t))
        if m:
            h = int(m.group(1)); mnt = int(m.group(2) or 0)
            return h + mnt/60.0
    return None

def speed_to_kmh(a: Atom) -> Optional[float]:
    try:
        qty = make_quantity(a.attrs["value"], a.attrs["unit"])
        return float(qty.to("kilometer/hour").magnitude)
    except Exception:
        return None

def first_distance_quantity(atoms: List[Atom]) -> Optional[ureg.Quantity]:
    for a in atoms:
        if a.type != "quantity":
            continue
        u = (a.attrs.get("unit") or "").lower()
        if ("hour" in u) or ("/" in u) or ("second" in u):  # skip speeds
            try:
                qty = make_quantity(a.attrs["value"], a.attrs["unit"])
                if qty.check("[length] / [time]"):
                    continue
            except Exception:
                pass
        try:
            qty = make_quantity(a.attrs["value"], a.attrs["unit"])
            if qty.check("[length]"):
                return qty
        except Exception:
            pass
    return None

def collect_motion_params(omega: PotentialGraph):
    by_id = {a.id: a for a in omega.atoms}
    rels_by_src: Dict[str, List[Relation]] = {}
    for r in omega.relations:
        rels_by_src.setdefault(r.src, []).append(r)

    agents = [a.id for a in omega.atoms if a.id in {"agent1","agent2"}]
    v, t = {}, {}
    for aid in agents:
        for r in rels_by_src.get(aid, []):
            if r.rtype == "speed":
                sv = speed_to_kmh(by_id[r.dst])
                if sv is not None: v[aid] = sv
            elif r.rtype == "depart_time":
                th = parse_clock_to_hours(by_id[r.dst].attrs.get("time"))
                if th is not None: t[aid] = th

    D_qty = first_distance_quantity(omega.atoms)
    D_km = float(D_qty.to("kilometer").magnitude) if D_qty is not None else None
    return v.get("agent1"), v.get("agent2"), t.get("agent1"), t.get("agent2"), D_km

def solve_meeting(omega: PotentialGraph) -> Optional[str]:
    v1, v2, t1, t2, D = collect_motion_params(omega)
    if v1 is None or v2 is None or t1 is None or t2 is None:
        return None
    if D is None:
        return ("I need the distance between the two locations to compute the meeting time. "
                "Add something like 'distance 465 km' to the prompt.")

    delta = abs(t1 - t2)
    d0 = (v1 * delta) if (t1 < t2) else (v2 * delta)
    rel = v1 + v2
    if rel <= 0: return None
    tau = (D - d0) / rel
    if tau < 0: return None

    t_meet_from_a1 = tau if t1 >= t2 else (delta + tau)
    s1 = v1 * t_meet_from_a1
    frac = min(max(s1 / D, 0.0), 1.0)
    return (f"They meet {tau:.2f} h after the later departure. "
            f"From Agent 1's origin, the meeting point is at ~{s1:.1f} km "
            f"({frac:.0%} of the {D:.0f} km segment).")

def render_answer(omega: PotentialGraph) -> str:
    task = omega.meta.get("task","general")
    if task == "taxonomy":
        isa = [r for r in omega.relations if r.rtype == "is_a"]
        if isa:
            by_id = {a.id: a for a in omega.atoms}
            r = sorted(isa, key=lambda e: -e.weight)[0]
            subj = by_id[r.src].label
            cls  = by_id[r.dst].label
            art = "an" if cls[0].lower() in "aeiou" else "a"
            return f"{subj.capitalize()} is {art} {cls}."
    if task == "motion":
        solved = solve_meeting(omega)
        if solved: return solved
        return ("Two agents depart with given times and speeds in opposite directions. "
                "Add a distance (e.g., 'distance 465 km') for a numeric meeting point.")
    return ""  # general fallback

# ================================
# HF generation (robust)
# ================================
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")

def generate_candidates(prompt: str, n: int = 2, max_new_tokens: int = 256) -> List[str]:
    if not HF_TOKEN:
        return [
            "I will compute the meeting point step by step, checking units and times.",
            "They meet roughly halfway along the route, depending on relative speeds and departure times.",
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

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="RCE LLM — Graph MVP", layout="wide")
st.title("RCE LLM — Graph MVP (theory-guided coherence)")

with st.expander("how this demo works"):
    st.markdown(
        "- extracts atoms (entities, times, quantities, classes, locations) and relations → **potential graph Gᵖ**\n"
        "- selects a coherent subgraph **Ω*** (actualization) via typing/unit/time checks\n"
        "- computes a coherence score **μ** with a per-constraint breakdown\n"
        "- renders a **final answer** from Ω* when possible (taxonomy & generic motion)\n"
        "- also generates LM candidates and **reranks** them by μ + alignment with Ω*\n"
        "\nSet `HF_TOKEN` in Streamlit → Settings → Secrets to get real LM generations."
    )

prompt = st.text_area(
    "enter a question/task",
    "A bus leaves Paris at 15h towards Lyon at 300 km/h. Another leaves Lyon at 14:30 towards Paris at 250 km/h. "
    "Distance 465 km. Where do they meet?\n\n"
    "Try also: 'a cat is an ...' or 'a cat is a ?'",
    height=160,
)

if st.button("run"):
    # 1) build potential graph G^p
    gp = build_gp(prompt)
    score_gp, br_gp, msgs_gp = mu(gp)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("potential graph Gᵖ (emergent)")
        st.json([a.__dict__ for a in gp.atoms])
        st.caption("relations")
        st.json([r.__dict__ for r in gp.relations])
        st.markdown(f"**μ(Gᵖ)** = {round(score_gp,3)}  |  breakdown: {br_gp}")
        if msgs_gp:
            st.caption("notes: " + "; ".join(msgs_gp))
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
        if msgs_om:
            st.warning("notes: " + "; ".join(msgs_om))
    with c4:
        show_graph("Ω* visualization (kept edges)", omega)

    # 3) render a final answer from Ω* (structure-driven)
    st.markdown("---")
    st.markdown("### Final answer (from Ω*)")
    ans = render_answer(omega)
    if ans:
        st.success(ans)
    else:
        st.info("No direct Ω*-rendered answer; see LM candidates below.")

    # 4) LM candidates + rerank by μ + coverage
    st.subheader("generation (HF model) + RCE rerank")
    candidates = generate_candidates(prompt, n=2)

    table = []
    best_idx, best_val = 0, -1.0
    for i, y in enumerate(candidates):
        mu_y, _, _ = mu(omega)      # coherence aligned with Ω*
        cov = alignment_score(y, omega)
        val = 0.7*mu_y + 0.3*cov
        table.append((y, round(mu_y,3), round(cov,3), round(val,3)))
        if val > best_val:
            best_val, best_idx = val, i

    st.markdown("**best candidate (by μ + coverage):**")
    st.write(table[best_idx][0])
    st.caption(f"μ={table[best_idx][1]}  |  coverage={table[best_idx][2]}  |  score={table[best_idx][3]}")

    with st.expander("all candidates (with μ & coverage)"):
        for y, m, cov, val in table:
            st.write("---")
            st.write(y)
            st.caption(f"μ={m}  |  coverage={cov}  |  score={val}")
