# apps/ui/streamlit_app.py
import os, re, ast, math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None  # optional

# -------------------------
# Core data types
# -------------------------
@dataclass
class Atom:
    id: str          # e.g., n1
    type: str        # entity | class | quantity | time | location | op
    label: str       # surface form
    attrs: Dict[str, Any]

@dataclass
class Edge:
    src: str
    dst: str
    rtype: str       # is_a | has_unit | convert_to | depart_time | speed | from | to | mentions | computes
    weight: float = 1.0

@dataclass
class Graph:
    atoms: List[Atom]
    edges: List[Edge]
    meta: Dict[str, Any]

# ============================================================
# Ultra-light “understanding”: extract atoms + relations
# ============================================================
UNIT_MAP = {
    # length
    "km": ("length", 1000.0, "m"),
    "m":  ("length", 1.0, "m"),
    "cm": ("length", 0.01, "m"),
    "mm": ("length", 0.001, "m"),
    "mi": ("length", 1609.344, "m"),
    # time
    "h": ("time", 3600.0, "s"),
    "hr": ("time", 3600.0, "s"),
    "min": ("time", 60.0, "s"),
    "s": ("time", 1.0, "s"),
    # speed
    "km/h": ("speed", 1000.0/3600.0, "m/s"),
    "kph": ("speed", 1000.0/3600.0, "m/s"),
    "m/s": ("speed", 1.0, "m/s"),
    "mph": ("speed", 1609.344/3600.0, "m/s"),
}

TAXONOMY = {
    "cat": ["animal", "mammal", "feline"],
    "dog": ["animal", "mammal", "canine"],
    "sparrow": ["animal", "bird"],
    "paris": ["city", "location"],
}

def parse_numbers_units(text: str) -> List[Atom]:
    atoms = []
    for i, m in enumerate(re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z/]+)", text)):
        val, unit = m.group(1), m.group(2).lower()
        if unit in UNIT_MAP:
            atoms.append(Atom(id=f"q{i}", type="quantity", label=f"{val} {unit}",
                              attrs={"value": float(val), "unit": unit}))
    return atoms

def parse_times(text: str) -> List[Atom]:
    ats = []
    # 15h, 14:30, 2024-01-01 etc.
    for j, m in enumerate(re.finditer(r"\b(\d{1,2}h(?:\d{2})?|\d{1,2}:\d{2})\b", text)):
        ats.append(Atom(id=f"t{j}", type="time", label=m.group(1), attrs={"raw": m.group(1)}))
    return ats

def parse_entities_classes(text: str) -> Tuple[List[Atom], List[Edge]]:
    atoms, edges = [], []
    # X is (a|an) Y
    m = re.search(r"\b(?:a|an)?\s*([A-Za-z\-]+)\s+is\s+(?:a|an)?\s*([A-Za-z\-?]+)", text, flags=re.I)
    if m:
        x = m.group(1).lower()
        y = m.group(2).lower().strip("?")
        a = Atom(id=f"e_{x}", type="entity", label=x, attrs={})
        c = Atom(id=f"c_{y}", type="class", label=y, attrs={})
        atoms += [a, c]
        edges.append(Edge(a.id, c.id, "is_a"))
        # known supersets
        for sup in TAXONOMY.get(x, []):
            sc = Atom(id=f"c_{sup}", type="class", label=sup, attrs={})
            atoms.append(sc)
            edges.append(Edge(a.id, sc.id, "is_a", 0.9))
    return dedup_atoms(atoms), edges

def parse_motion(text: str) -> Tuple[List[Atom], List[Edge]]:
    atoms, edges = [], []
    if re.search(r"\b(from|towards|leaves?|depart)\b", text, flags=re.I):
        a1 = Atom(id="agent1", type="entity", label="Agent 1", attrs={})
        a2 = Atom(id="agent2", type="entity", label="Agent 2", attrs={})
        atoms += [a1, a2]
        # from X to Y
        m = re.search(r"\bfrom\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)\s+(?:to|towards)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)", text, flags=re.I)
        if m:
            L1 = Atom(id=f"loc_{m.group(1).lower()}", type="location", label=m.group(1), attrs={})
            L2 = Atom(id=f"loc_{m.group(2).lower()}", type="location", label=m.group(2), attrs={})
            atoms += [L1, L2]
            edges += [Edge("agent1", L1.id, "from"), Edge("agent1", L2.id, "to")]
        # “another leaves … to …” for agent2
        m2 = re.search(r"\banother\b.*?\bleaves?\s+([A-Za-zÀ-ÖØ-öø-ÿ]+).*?\b(?:towards|to)\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)", text, flags=re.I|re.S)
        if m2:
            L3 = Atom(id=f"loc_{m2.group(1).lower()}", type="location", label=m2.group(1), attrs={})
            L4 = Atom(id=f"loc_{m2.group(2).lower()}", type="location", label=m2.group(2), attrs={})
            atoms += [L3, L4]
            edges += [Edge("agent2", L3.id, "from"), Edge("agent2", L4.id, "to")]
    return dedup_atoms(atoms), edges

def dedup_atoms(atoms: List[Atom]) -> List[Atom]:
    seen, out = set(), []
    for a in atoms:
        key = (a.id, a.type, a.label)
        if key not in seen:
            seen.add(key); out.append(a)
    return out

def build_graph(text: str) -> Graph:
    atoms: List[Atom] = []
    edges: List[Edge] = []
    # parse blocks
    q = parse_numbers_units(text); atoms += q
    t = parse_times(text); atoms += t
    a_ec, e_ec = parse_entities_classes(text); atoms += a_ec; edges += e_ec
    a_mo, e_mo = parse_motion(text); atoms += a_mo; edges += e_mo

    # link quantities to agents as speeds (first two)
    speeds = [a for a in atoms if a.type == "quantity" and UNIT_MAP.get(a.attrs["unit"], ("", "", ""))[0] == "speed"]
    if speeds:
        if any(a.id == "agent1" for a in atoms): edges.append(Edge("agent1", speeds[0].id, "speed"))
    if len(speeds) >= 2 and any(a.id == "agent2" for a in atoms):
        edges.append(Edge("agent2", speeds[1].id, "speed"))

    # attach times to agents (first two)
    times = [a for a in atoms if a.type == "time"]
    if times and any(a.id == "agent1" for a in atoms): edges.append(Edge("agent1", times[0].id, "depart_time"))
    if len(times) >= 2 and any(a.id == "agent2" for a in atoms): edges.append(Edge("agent2", times[1].id, "depart_time"))

    return Graph(atoms=dedup_atoms(atoms), edges=edges, meta={"text": text})

# ============================================================
# Coherence μ (very small, generic)
# ============================================================
def mu(graph: Graph) -> Tuple[float, Dict[str, float], List[str]]:
    msgs = []
    # typing checks (coarse)
    by_id = {a.id: a for a in graph.atoms}
    score = 1.0; w = {"typing":0.5, "units":0.3, "time":0.2}
    s_type = 1.0
    for e in graph.edges:
        if e.rtype in {"from","to"}:
            if by_id.get(e.src, Atom("", "", "", {})).type != "entity" or by_id.get(e.dst, Atom("", "", "", {})).type != "location":
                s_type = min(s_type, 0.0); msgs.append("type mismatch in motion edge")
        if e.rtype == "speed":
            dst = by_id.get(e.dst)
            if not dst or UNIT_MAP.get(dst.attrs.get("unit",""), ("", "", ""))[0] != "speed":
                s_type = min(s_type, 0.0); msgs.append("speed without speed unit")
        if e.rtype == "depart_time":
            if by_id.get(e.dst, Atom("", "", "", {})).type != "time":
                s_type = min(s_type, 0.0); msgs.append("depart_time not time")
        if e.rtype == "is_a":
            if by_id.get(e.src, Atom("", "", "", {})).type != "entity" or by_id.get(e.dst, Atom("", "", "", {})).type != "class":
                s_type = min(s_type, 0.0); msgs.append("is_a typing error")

    # units presence – if speed edges exist, units must be valid
    s_units = 1.0
    for e in graph.edges:
        if e.rtype == "speed":
            q = by_id.get(e.dst)
            u = q.attrs.get("unit") if q else None
            if u not in UNIT_MAP:
                s_units = min(s_units, 0.5); msgs.append(f"unknown unit: {q.label if q else '??'}")

    # time parse presence
    s_time = 1.0
    # (light: existence already checked by typing; detailed parsing omitted)

    score = w["typing"]*s_type + w["units"]*s_units + w["time"]*s_time
    return score, {"typing":s_type, "units":s_units, "time":s_time}, msgs

# ============================================================
# Resolvers (task-agnostic set)
# ============================================================
def detect_task(text: str) -> str:
    if re.search(r"\bis\s+(?:a|an|\w+\??)$", text.strip(), flags=re.I): return "taxonomy"
    if re.search(r"\bconvert\b|\bto\b\s+[A-Za-z/]+$", text.strip(), flags=re.I): return "convert"
    if re.search(r"[0-9][0-9\+\-\*/\s\(\)\.]*=?\s*$", text.strip()): return "arithmetic"
    if re.search(r"\b(from|towards|leaves?|depart)\b", text, flags=re.I): return "motion"
    return "general"

# ---- helpers
def to_base(value: float, unit: str) -> Tuple[str, float]:
    kind, scale, base = UNIT_MAP[unit]
    return kind, value * scale, base

def from_base(kind: str, base_value: float, target_unit: str) -> float:
    k2, scale, base = UNIT_MAP[target_unit]
    assert k2 == kind and base_value is not None
    return base_value / scale

def parse_clock_to_hours(s: str) -> Optional[float]:
    # 15h or 14:30
    m = re.match(r"^(\d{1,2})h(\d{2})?$", s)
    if m:
        h = int(m.group(1)); mnt = int(m.group(2) or 0); return h + mnt/60.0
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) + int(m.group(2))/60.0
    return None

# ---- taxonomy
def resolve_taxonomy(graph: Graph) -> Optional[Dict[str, Any]]:
    for e in graph.edges:
        if e.rtype == "is_a":
            x = next(a for a in graph.atoms if a.id == e.src)
            y = next(a for a in graph.atoms if a.id == e.dst)
            art = "an" if y.label[0].lower() in "aeiou" else "a"
            return {"type":"taxonomy", "text": f"{x.label.capitalize()} is {art} {y.label}."}
    return None

# ---- conversions like: "convert 5 km to m"
def resolve_conversion(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"convert\s+(\d+(?:\.\d+)?)\s*([A-Za-z/]+)\s+to\s+([A-Za-z/]+)", text, flags=re.I)
    if not m: return None
    value, u_from, u_to = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
    if u_from not in UNIT_MAP or u_to not in UNIT_MAP: return {"type":"convert", "error":"unknown unit"}
    k1, base_v, base_unit = to_base(value, u_from)
    if UNIT_MAP[u_to][0] != k1: return {"type":"convert", "error":"incompatible units"}
    out = from_base(k1, base_v, u_to)
    return {"type":"convert", "text": f"{value} {u_from} = {out:.6g} {u_to}"}

# ---- arithmetic like: "12*(3+4)-5"
SAFE_OPS = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.FloorDiv}
def eval_arith(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Num): return n.n
        if isinstance(n, ast.BinOp) and type(n.op) in SAFE_OPS:
            return _eval(n.left) + 0 if isinstance(n.op, ast.UAdd) else \
                   (_eval(n.left) - 0 if isinstance(n.op, ast.USub) else None)
        if isinstance(n, ast.BinOp):
            if type(n.op) not in SAFE_OPS: raise ValueError("op not allowed")
            a, b = _eval(n.left), _eval(n.right)
            if   isinstance(n.op, ast.Add): return a + b
            elif isinstance(n.op, ast.Sub): return a - b
            elif isinstance(n.op, ast.Mult): return a * b
            elif isinstance(n.op, ast.Div): return a / b
            elif isinstance(n.op, ast.FloorDiv): return a // b
            elif isinstance(n.op, ast.Pow): return a ** b
            elif isinstance(n.op, ast.Mod): return a % b
        if isinstance(n, ast.UnaryOp) and type(n.op) in SAFE_OPS: return +_eval(n.operand) if isinstance(n.op, ast.UAdd) else -_eval(n.operand)
        raise ValueError("bad expression")
    return float(_eval(node))

def resolve_arithmetic(text: str) -> Optional[Dict[str, Any]]:
    expr = re.sub(r"[^\d\+\-\*\/\(\)\.\s]", "", text)
    if not re.search(r"\d", expr): return None
    try:
        val = eval_arith(expr)
        return {"type":"arithmetic", "text": f"{expr.strip()} = {val:g}"}
    except Exception:
        return {"type":"arithmetic", "error":"invalid expression"}

# ---- motion: meeting time if speeds + times + distance present in text
def find_quant(graph: Graph, kind: str) -> List[Atom]:
    return [a for a in graph.atoms if a.type=="quantity" and UNIT_MAP[a.attrs["unit"]][0] == kind]

def resolve_motion(graph: Graph) -> Optional[Dict[str, Any]]:
    speeds = find_quant(graph, "speed")
    if len(speeds) < 2: return None
    times = [a for a in graph.atoms if a.type=="time"]
    if len(times) < 2: return {"type":"motion", "error":"need two departure times"}
    # optional distance
    lengths = find_quant(graph, "length")
    D = None
    if lengths:
        # take the first length and convert to meters
        D = to_base(lengths[0].attrs["value"], lengths[0].attrs["unit"])[1]  # in meters

    # speeds to m/s
    v1 = to_base(speeds[0].attrs["value"], speeds[0].attrs["unit"])[1]
    v2 = to_base(speeds[1].attrs["value"], speeds[1].attrs["unit"])[1]
    t1 = parse_clock_to_hours(times[0].label)
    t2 = parse_clock_to_hours(times[1].label)
    if t1 is None or t2 is None: return {"type":"motion", "error":"bad time format"}

    if D is None:
        return {"type":"motion", "needs_more_info":True,
                "text":"Add a distance like 'distance 465 km' to compute the numeric meeting point."}

    # Convert D to km & speeds to km/h for human rendering
    D_km = D / 1000.0
    v1_kmh = v1 * 3.6
    v2_kmh = v2 * 3.6
    delta = abs(t1 - t2)  # hours
    d0 = v1_kmh*delta if t1 < t2 else v2_kmh*delta
    tau = (D_km - d0) / (v1_kmh + v2_kmh)
    if tau < 0: return {"type":"motion", "error":"inconsistent inputs"}
    later_clock = max(t1, t2) + tau
    hh = int(later_clock) % 24
    mm = int(round((later_clock - int(later_clock))*60)) % 60
    s1 = (v1_kmh * (tau if t1 >= t2 else delta + tau))
    return {"type":"motion", "text": f"Meeting around {hh:02d}:{mm:02d}. From Agent 1 origin: ~{s1:.1f} km of {D_km:.0f} km."}

# ---- general fallback: LM (optional)
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")

def lm_answer(prompt: str) -> str:
    if not (InferenceClient and HF_TOKEN):
        return "I can’t compute an exact answer with the given info. Please add more details or enable the LM (set HF_TOKEN in Streamlit secrets)."
    try:
        client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
        return client.text_generation(prompt, max_new_tokens=256, temperature=0.5)
    except Exception:
        return "Language model call failed. Try again or remove LM dependency."

# ---- orchestrator
def resolve(graph: Graph) -> Dict[str, Any]:
    text = graph.meta["text"]
    task = detect_task(text)
    out: Optional[Dict[str, Any]] = None
    if task == "taxonomy":
        out = resolve_taxonomy(graph)
    elif task == "convert":
        out = resolve_conversion(text)
    elif task == "arithmetic":
        out = resolve_arithmetic(text)
    elif task == "motion":
        out = resolve_motion(graph)

    if out and "text" in out:
        return {"task": task, "answer": out["text"], "details": out}
    # fallback to LM or graceful message
    return {"task": task, "answer": lm_answer(text), "details": out or {"note":"no deterministic resolver matched"}}

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="RCE mini-engine MVP", layout="wide")
st.title("RCE mini-engine — universal MVP")

with st.expander("What this does"):
    st.markdown(
        "- builds a **tiny graph** from any prompt (entities, classes, quantities, times, locations)\n"
        "- computes a small **coherence score μ** (typing/units/time presence)\n"
        "- runs **generic resolvers**: taxonomy, conversion, arithmetic, motion\n"
        "- always outputs a **final answer** (or a precise ‘missing info’ message)\n"
        "- optional LM fallback (set `HF_TOKEN` secret)"
    )

prompt = st.text_area("Ask anything", "convert 5 km to m\n\nTry:\n- a cat is an ...\n- 12*(3+4)-5\n- A car leaves Paris at 15h towards Lyon at 300 km/h. Another leaves Lyon at 14:30 towards Paris at 250 km/h. Distance 465 km. Where do they meet?",
                      height=160)

if st.button("Run"):
    # 1) graph
    G = build_graph(prompt)
    score, br, msgs = mu(G)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Graph (atoms)")
        st.json([a.__dict__ for a in G.atoms])
        st.caption("Edges")
        st.json([e.__dict__ for e in G.edges])
        st.markdown(f"**μ** = {score:.2f}  |  breakdown: {br}")
        if msgs: st.caption("notes: " + "; ".join(msgs))
    with c2:
        st.subheader("Graphviz (simple)")
        # Render as DOT without external libs
        dot = "digraph G {\nrankdir=LR;\n"
        for a in G.atoms:
            dot += f'"{a.id}" [label="{a.label}\\n[{a.type}]"];\n'
        for e in G.edges:
            dot += f'"{e.src}" -> "{e.dst}" [label="{e.rtype}"];\n'
        dot += "}\n"
        st.graphviz_chart(dot)

    st.markdown("---")
    # 2) resolution
    res = resolve(G)
    st.subheader("Final answer")
    st.success(res["answer"])
    with st.expander("resolution details"):
        st.json(res)
