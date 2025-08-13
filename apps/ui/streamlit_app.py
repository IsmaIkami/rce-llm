import os, re
import streamlit as st
from dataclasses import dataclass
from typing import List, Dict
from huggingface_hub import InferenceClient
from pint import UnitRegistry
from dateutil.parser import parse as dtparse

# ---------- tiny RCE pieces (MVP) ----------
ureg = UnitRegistry()

@dataclass
class Atom:
    id: str
    type: str   # "quantity" | "time" | ...
    label: str
    attrs: Dict

@dataclass
class PotentialGraph:
    atoms: List[Atom]
    relations: List[Dict]
    meta: Dict
    
def simple_parse(text: str) -> PotentialGraph:
    atoms, rels = [], []
    # quantities like "300 km/h", "5 km", "2.5 m"
    for i, m in enumerate(re.finditer(r"(\d+(?:\.\d+)?)\s*([A-Za-z\/\^°%]+)", text)):
        val, unit = m.group(1), m.group(2)
        if any(x in unit.lower() for x in ["km", "m", "cm", "mm", "kg", "g", "h", "s", "mph", "km/h", "°c"]):
            atoms.append(Atom(id=f"q{i}", type="quantity",
                              label=f"{val} {unit}",
                              attrs={"value": float(val), "unit": unit}))
    # time stamps like "15h", "2024-01-01", "14:30"
    for j, m in enumerate(re.finditer(r"(\d{1,2}h(?:\d{2})?|\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2})", text)):
        atoms.append(Atom(id=f"t{j}", type="time",
                          label=m.group(1), attrs={"time": m.group(1)}))
    return PotentialGraph(atoms=atoms, relations=[], meta={"text": text})


def units_consistency_score(gp: PotentialGraph) -> float:
    """soft score: 1.0 if all quantities have compatible dimensions when they share the same unit word,
    else penalize. (toy heuristic for MVP)"""
    dims = []
    for a in gp.atoms:
        if a.type != "quantity":
            continue
        try:
            q = ureg(a.attrs["value"]) * ureg(a.attrs["unit"])
            dims.append(q.dimensionality)
        except Exception:
            # unknown unit → small penalty
            dims.append(None)
    if not dims:
        return 1.0
    # if any is None → reduce score
    if any(d is None for d in dims):
        return 0.6
    # if mixed dimensionalities in a short answer → small penalty
    same = all(d == dims[0] for d in dims)
    return 1.0 if same else 0.7

def time_parse_ok(gp: PotentialGraph) -> float:
    ok = 1.0
    for a in gp.atoms:
        if a.type == "time":
            try:
                _ = dtparse(str(a.attrs["time"]))
            except Exception:
                ok = min(ok, 0.5)
    return ok

def coherence_mu(gp: PotentialGraph) -> float:
    # weighted average of toy constraints
    u = units_consistency_score(gp)
    t = time_parse_ok(gp)
    return 0.6 * u + 0.4 * t

# ---------- HF inference client ----------
HF_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Streamlit secrets

def generate_candidates(prompt: str, n: int = 2, max_new_tokens: int = 256) -> List[str]:
    if not HF_TOKEN:
        return [("⚠️ Set HF_TOKEN in Streamlit secrets for generation.",)]
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
        outs.append(text)
    return outs

def rce_rerank(prompt: str) -> Dict:
    # parse the prompt (optional), but most importantly parse each candidate output
    candidates = generate_candidates(prompt, n=2)
    scored = []
    for y in candidates:
        gp = simple_parse(y)
        scored.append((y, coherence_mu(gp)))
    best = max(scored, key=lambda x: x[1]) if scored else ("", 0.0)
    return {"best": best[0], "best_score": best[1], "all": scored}

# ---------- streamlit UI ----------
st.set_page_config(page_title="RCE LLM — MVP", layout="wide")
st.title("RCE LLM — MVP (theory-guided coherence)")

with st.expander("how this demo works"):
    st.markdown(
        "- parses quantities/time from the text\n"
        "- scores simple coherence (units/time)\n"
        "- calls a HF model to generate 2 candidates\n"
        "- reranks by coherence μ and shows the winner\n"
        "\nSet `HF_TOKEN` in Streamlit → Settings → Secrets.\n"
    )

prompt = st.text_area(
    "enter a question or task",
    "A train leaves Paris at 15h towards Lyon at 300 km/h. Another leaves Lyon at 14:30 towards Paris at 250 km/h. Where do they meet?",
    height=120,
)

if st.button("run"):
    # show parser on the prompt
    gp_in = simple_parse(prompt)
    st.subheader("parsed atoms from prompt")
    st.json([a.__dict__ for a in gp_in.atoms])
    st.write("prompt coherence μ:", round(coherence_mu(gp_in), 3))

    # generate + rerank
    st.subheader("generation (HF model) + RCE rerank")
    result = rce_rerank(prompt)
    st.markdown("**best candidate (by μ):**")
    st.write(result["best"])
    st.caption(f"μ(best) = {round(result['best_score'], 3)}")

    with st.expander("all candidates (with μ)"):
        for txt, s in result["all"]:
            st.write("---")
            st.write(txt)
            st.caption(f"μ = {round(s, 3)}")

    st.info("this is a minimal rce; the full version adds proper graph Ω actualization, richer constraints, and a learned μ.")
