"""
Three improvements to the toy session rerank (quince_session_rerank.py), tested side by side:

    A  baseline      MPNet on name + description (the guarded rerank as it stands)
    B  + attributes  append a season / formality / occasion line per item (quince_item_tags.py)
    C  + BGE         swap in bge-small-en-v1.5, a query-to-passage retrieval encoder,
                     with its query instruction prefix
    D  + cross-enc   at search time, rescore the query's top 30 with a cross-encoder
                     (ms-marco-MiniLM-L-6-v2) and store those scores in the session
    E  set model     the masked set model (quince_set_model.py) reads the session as a menu
                     (search = header, clicks and views = listed products) and predicts the
                     missing product; one combined anchor instead of click + query

Two evaluations:
  1. The wedding session from the doc, scored on the hand-labeled outfit set. Caveat: the
     attribute tags and the outfit labels were both written by hand, so B-D are partly
     circular on this one.
  2. Many sessions (fairer). Intents come from the keyword taxonomy (quince_taxonomy.py),
     which reads only the original product text, not the attribute tags. Each session
     clicks two items and views one from the intent, searches the intent's query, and we
     check whether a held-out item from the same intent (the "next click") lands in the
     reranked top 10. Also reported: query recall@10, how well the search alone finds the
     intent's items.

Usage:
    .venv/bin/python quince_rerank_improvements.py
"""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from quince_catalog import PRODUCTS
from quince_embeddings import embed
from quince_item_tags import attribute_line
from quince_session_rerank import (CATS, IDX, OUTFIT, LINEN_HOME, SESSION, SHOPPERS, SKUS, W_CLICK, W_VIEW,
                                   ROW, offline_scores, session_rerank)
from quince_taxonomy import TAXONOMY

BGE = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "
CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_TOP = 30

# intent -> (query, rule over the keyword taxonomy). Rules read only the original text.
INTENTS = {
    "lounge":     ("cozy loungewear for lazy weekends", lambda t: "Lounge & Sleep" in t.occasions),
    "cold":       ("something warm for a cold winter", lambda t: "Cold Weather" in t.occasions),
    "travel":     ("what to pack for a trip", lambda t: "Travel" in t.occasions),
    "bedding":    ("new sheets and bedding for the bedroom", lambda t: t.type == "Bedding"),
    "earrings":   ("earrings", lambda t: t.type == "Earrings"),
    "necklaces":  ("a necklace", lambda t: t.type == "Necklaces"),
    "pearl":      ("classic pearl jewelry", lambda t: "Pearl" in t.materials),
    "diamond":    ("lab grown diamonds", lambda t: "Diamond" in t.materials),
    "silk":       ("silk pieces", lambda t: "Silk" in t.materials),
    "leather":    ("leather handbag", lambda t: "Leather" in t.materials and t.department == "Bags"),
    "furniture":  ("furniture for the living room", lambda t: t.type in ("Furniture", "Decor")),
    "mens":       ("men's clothes", lambda t: t.audience == "Men"),
    "sweaters":   ("a soft sweater", lambda t: t.type == "Sweaters"),
}


def item_text(p, attrs: bool) -> str:
    base = f"{p.name}. {p.description}"
    return f"{base} {attribute_line(p.sku)}" if attrs else base


@lru_cache(maxsize=None)
def bge() -> SentenceTransformer:
    return SentenceTransformer(BGE)


@lru_cache(maxsize=None)
def cross() -> CrossEncoder:
    return CrossEncoder(CROSS)


class Variant:
    def __init__(self, name: str, encoder: str, attrs: bool, use_cross: bool):
        self.name, self.encoder, self.attrs, self.use_cross = name, encoder, attrs, use_cross
        self.texts = [item_text(p, attrs) for p in PRODUCTS]
        self.V = self._encode(self.texts, query=False)

    def _encode(self, texts, query: bool) -> np.ndarray:
        if self.encoder == "mpnet":
            return embed("mpnet", texts)
        texts = [BGE_QUERY + t for t in texts] if query else texts
        return bge().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def query_scores(self, query: str) -> np.ndarray:
        """Standardized query-to-item scores, as stored in session state at search time."""
        q = self._encode([query], query=True)[0]
        s = self.V @ q
        s = (s - s.mean()) / s.std()
        if not self.use_cross:
            return s
        top = np.argsort(-s)[:CE_TOP]
        ce = cross().predict([(query, self.texts[i]) for i in top])
        out = np.full(len(s), -3.0)                      # outside the top 30: far from the query
        out[top] = 3.0 * (ce - ce.min()) / (ce.max() - ce.min() + 1e-9)  # 0..3, above the floor
        return (out - out.mean()) / out.std()

    def click_scores(self, session: dict) -> np.ndarray:
        """Click anchor (views at half weight), z-scored, limited to the clicked categories."""
        w = [(s, W_CLICK) for s in session["clicks"]] + [(s, W_VIEW) for s in session["views"]]
        click = sum(wt * self.V[IDX[s]] for s, wt in w)
        c = self.V @ (click / np.linalg.norm(click))
        c = (c - c.mean()) / c.std()
        cats = {CATS[IDX[s]] for s, _ in w}
        c[~np.isin(CATS, list(cats))] = c.min()
        return c

    def sims(self, session: dict) -> np.ndarray:
        """anchors × items, guarded mode: per-anchor z-scores, click anchor limited to clicked categories."""
        return np.stack([self.click_scores(session), self.query_scores(session["query"])])


class SetModelVariant:
    """System E with the same interface as Variant. Its single anchor fills both rows, so
    session_rerank's max over anchors is just the set-model score."""
    name = "E set model"
    texts = None

    def query_scores(self, query: str) -> np.ndarray:
        from quince_set_model import session_scores
        s = session_scores([], query)
        return (s - s.mean()) / s.std()

    def sims(self, session: dict) -> np.ndarray:
        from quince_set_model import session_scores
        s = session_scores([IDX[x] for x in session["clicks"] + session["views"]], session["query"])
        s = (s - s.mean()) / s.std()
        return np.stack([s, s])


def rerank_all(v: Variant, session: dict, lam: float):
    sims = v.sims(session)
    anc = {"click": None, "query": None}
    seen = set(session["clicks"]) | set(session["views"])
    return [session_rerank(offline_scores(aff, seed), sims, anc, seen, lam)[1]
            for seed, aff in enumerate(SHOPPERS.values())]


def wedding_eval(v: Variant, lam: float) -> tuple[float, float, list[str]]:
    rows = rerank_all(v, SESSION, lam)
    outfit = np.mean([[SKUS[i] in OUTFIT for i in r] for r in rows])
    linen = np.mean([[SKUS[i] in LINEN_HOME for i in r] for r in rows])
    q = v.query_scores(SESSION["query"])
    return outfit, linen, [PRODUCTS[i].name for i in np.argsort(-q)[:5]]


def intent_sets() -> dict[str, tuple[str, list[int]]]:
    out = {}
    for name, (query, rule) in INTENTS.items():
        items = [i for i, t in enumerate(TAXONOMY) if rule(t)]
        if len(items) >= 4:
            out[name] = (query, items)
    return out


def multi_session_eval(v: Variant, lam: float, sessions_per_intent: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    hits, rr, qrec, per_intent = [], [], [], {}
    for name, (query, items) in intent_sets().items():
        q = v.query_scores(query)
        top10 = set(np.argsort(-q)[:10])
        qrec.append(len(top10 & set(items)) / min(10, len(items)))
        h_int = []
        for _ in range(sessions_per_intent):
            pick = rng.choice(items, size=4, replace=False)
            session = {"clicks": [SKUS[pick[0]], SKUS[pick[1]]], "views": [SKUS[pick[2]]], "query": query}
            target = pick[3]
            for row in rerank_all(v, session, lam):
                hit = target in row
                hits.append(hit); h_int.append(hit)
                rr.append(1 / (row.index(target) + 1) if hit else 0.0)
        per_intent[name] = np.mean(h_int)
    return np.mean(hits), np.mean(rr), np.mean(qrec), per_intent


def no_session_baseline(sessions_per_intent: int = 10, seed: int = 0) -> float:
    """Hit@10 of the held-out item in the offline shortlist alone (λ = 0, no session)."""
    rng = np.random.default_rng(seed)
    hits = []
    for _, (_, items) in intent_sets().items():
        for _ in range(sessions_per_intent):
            pick = rng.choice(items, size=4, replace=False)
            seen = {SKUS[i] for i in pick[:3]}
            for s, aff in enumerate(SHOPPERS.values()):
                r = offline_scores(aff, s)
                row = [i for i in np.argsort(-r) if SKUS[i] not in seen][:ROW]
                hits.append(pick[3] in row)
    return float(np.mean(hits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.6)
    args = ap.parse_args()

    variants = [Variant("A baseline (MPNet)", "mpnet", False, False),
                Variant("B + attributes", "mpnet", True, False),
                Variant("C + BGE", "bge", True, False),
                Variant("D + cross-encoder", "bge", True, True)]
    if (Path(__file__).parent / "outputs" / "set_model.pt").exists():
        variants.append(SetModelVariant())

    print(f"λ = {args.lam}.  Intents: " + ", ".join(f"{k} ({len(v[1])})" for k, v in intent_sets().items()))

    print("\n1. Wedding session (hand labels; B-D partly circular, see docstring)")
    print(f"   {'variant':<22} {'outfit@10':>9} {'linen-home@10':>13}   top 3 for 'summer wedding guest outfit'")
    for v in variants:
        o, l, top = wedding_eval(v, args.lam)
        print(f"   {v.name:<22} {o:>9.0%} {l:>13.0%}   " + " | ".join(top[:3]))

    print(f"\n2. Many sessions ({len(intent_sets())} intents × 10 sessions × 5 shoppers), taxonomy labels")
    print(f"   no session at all: next-click hit@10 {no_session_baseline():.0%}")
    print(f"   {'variant':<22} {'hit@10':>7} {'MRR':>6} {'query recall@10':>16}")
    per = {}
    for v in variants:
        h, m, qr, per[v.name] = multi_session_eval(v, args.lam)
        print(f"   {v.name:<22} {h:>7.0%} {m:>6.3f} {qr:>16.0%}")

    print("\n   next-click hit@10 by intent")
    names = list(per)
    print(f"   {'intent':<11}" + "".join(f"{n.split()[0]:>6}" for n in names))
    for intent in intent_sets():
        print(f"   {intent:<11}" + "".join(f"{per[n][intent]:>6.0%}" for n in names))

    t0 = time.perf_counter()
    for _ in range(5):
        cross().predict([(SESSION["query"], variants[3].texts[i]) for i in range(CE_TOP)])
    print(f"\nCross-encoder latency, {CE_TOP} pairs on this laptop's CPU: "
          f"{(time.perf_counter() - t0) / 5 * 1000:.0f} ms per search (runs once at search time, not per page load)")


if __name__ == "__main__":
    main()
