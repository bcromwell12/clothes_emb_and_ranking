"""
Session re-rank as a service for the explorer's "Additional actions > Re-rank".

Wraps the toy session rerank (quince_session_rerank.py) and its four system variants
(quince_rerank_improvements.py), plus a fifth system built on the masked set model
(quince_set_model.py). The explorer runs it in three steps: a session (clicked and viewed
products and/or search terms), a first pass (first_pass below: search results or a mock
shopper's "Picked for you" list), then rerank() of that first pass with the session.

Scoring is the session rerank as written: the list (top 30) plus the nearest items to each
anchor, scored by (1 - λ)·z(list score) + λ·z(similarity to the closer anchor), with the
first two slots pinned. Either anchor can be missing: no clicks means search only, no
search means clicks only.

Systems A-D have two anchors, clicks and search. System E has one: the set model reads the
whole session as a menu (search as the header, clicked products as the visible items) and
predicts the missing product, so clicks and search are combined before scoring.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from quince_catalog import PRODUCTS
from quince_rerank_improvements import Variant
from quince_session_rerank import (IDX, K_SHORTLIST, M_RETRIEVE, PINNED, ROW, SHOPPERS, SKUS, offline_scores,
                                   session_rerank, z)

SYSTEMS = {
    "A": ("A · MPNet", "mpnet", False, False, "MPNet on name + description."),
    "B": ("B · MPNet + attributes", "mpnet", True, False, "Adds a season / formality / occasion line to each product."),
    "C": ("C · BGE", "bge", True, False, "bge-small retrieval encoder with its query prefix, plus attributes."),
    "D": ("D · BGE + cross-encoder", "bge", True, True, "Rescores the search's top 30 with a cross-encoder."),
}
SET_SYSTEM = ("E · Set model", "Masked set model: the session is a menu (search = header, clicks = listed "
              "products) and the model predicts what else belongs on it.")
if (Path(__file__).parent / "outputs" / "set_model.pt").exists():
    SYSTEMS["E"] = (SET_SYSTEM[0], "set", False, False, SET_SYSTEM[1])
# The explorer offers only the two best on next-click hit@10 in quince_rerank_improvements.py
# (106-product catalog: D 66%, C 64%; B 62%, A 61%, E 59%). The rest stay there for that comparison.
ACTIVE = {"D": ("BGE + cross-encoder", "Best on next clicks (66% hit@10) and on finding search matches (89% recall@10). "
                                       "Rescores the search's top 30 with a cross-encoder."),
          "C": ("BGE", "Close second on next clicks (64% hit@10), simpler and faster. bge-small retrieval encoder "
                       "with its query prefix, plus the season / formality / occasion tags.")}
DEFAULT_SYSTEM, DEFAULT_LAM = "D", 0.6


@lru_cache(maxsize=None)
def variant(key: str) -> Variant:
    label, encoder, attrs, use_cross, _ = SYSTEMS[key]
    return Variant(label, encoder, attrs, use_cross)


def options() -> dict:
    return {"systems": [{"key": k, "label": label, "desc": desc} for k, (label, desc) in ACTIVE.items()],
            "shoppers": list(SHOPPERS), "defaultSystem": DEFAULT_SYSTEM, "defaultLam": DEFAULT_LAM,
            "pinned": PINNED, "row": ROW}


def anchor_scores(system: str, clicks: list[str], views: list[str], queries: list[str]) -> tuple[list[str], np.ndarray]:
    """Anchor names and an anchors x products matrix of z-scored similarities.
    Several search terms share one search anchor: the average of their standardized scores."""
    if system == "E":
        from quince_set_model import session_scores
        return ["session"], z(session_scores([IDX[s] for s in clicks + views], ", ".join(queries)))[None]
    v = variant(system)
    names, rows = [], []
    if clicks or views:
        names.append("click")
        rows.append(v.click_scores({"clicks": clicks, "views": views}))
    if queries:
        names.append("query")
        rows.append(z(np.mean([v.query_scores(q) for q in queries], axis=0)))
    return names, np.stack(rows)


def score_parts(ranker: np.ndarray, sims: np.ndarray, seen: set[str], lam: float):
    """The candidates and the two halves of their score, exactly as session_rerank computes them."""
    shortlist = [i for i in np.argsort(-ranker) if SKUS[i] not in seen][:K_SHORTLIST]
    retrieved = {int(i) for row in sims for i in np.argsort(-row)[:M_RETRIEVE + len(seen)]}
    cand = np.array(sorted(set(shortlist) | {i for i in retrieved if SKUS[i] not in seen}))
    return cand, (1 - lam) * z(ranker[cand]), lam * z(sims[:, cand].max(0)), sims[:, cand].argmax(0)


def rerank(system: str, clicks: list[str], queries: list[str] | str, lam: float,
           order: list[str] | None = None, shopper: str | None = None, views: list[str] | None = None) -> dict:
    if system not in ACTIVE:
        raise ValueError(f"Unknown system {system!r}. Choose one of {', '.join(ACTIVE)}.")
    clicks = [s for s in clicks if s in IDX]
    views = [s for s in (views or []) if s in IDX and s not in clicks]
    queries = [queries] if isinstance(queries, str) else list(queries or [])
    queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
    if not clicks and not views and not queries:
        raise ValueError("The session is empty: add a search term or a clicked item first.")
    names, sims = anchor_scores(system, clicks, views, queries)

    if order is not None:  # the explorer's list: higher score for earlier positions
        ranker = np.full(len(PRODUCTS), -float(len(PRODUCTS)))
        for pos, sku in enumerate(s for s in order if s in IDX):
            ranker[IDX[sku]] = -pos
    elif shopper in SHOPPERS:
        ranker = offline_scores(SHOPPERS[shopper], list(SHOPPERS).index(shopper))
    else:
        raise ValueError("Nothing to re-rank: build a first pass (or name a shopper) first.")

    seen = set(clicks) | set(views)
    before, after, tags = session_rerank(ranker, sims, dict.fromkeys(names), seen, lam)
    cand, list_part, sess_part, which = score_parts(ranker, sims, seen, lam)
    # position in the list as the shopper saw it, i.e. without the products already in the session
    source_rank = {int(i): r + 1 for r, i in enumerate(i for i in np.argsort(-ranker) if SKUS[i] not in seen)}
    parts = {int(i): {"list": round(float(a), 3), "session": round(float(b), 3), "anchor": names[w]}
             for i, a, b, w in zip(cand, list_part, sess_part, which)}
    item = lambda i: {"sku": SKUS[i], "i": int(i)}
    return {
        "system": ACTIVE[system][0], "anchors": names, "lam": lam, "pinned": PINNED,
        "before": [item(i) for i in before],
        "after": [{**item(i), **parts[int(i)], "anchor": str(tags[i]), "was": source_rank[int(i)],
                   "pinned": r < PINNED and int(i) == int(before[r])} for r, i in enumerate(after)],
        # every candidate with its score, best first: what a later stage (the Ranker) can choose from
        "candidates": sorted(({**item(i), **parts[int(i)], "was": source_rank[int(i)]} for i in cand),
                             key=lambda c: -(c["list"] + c["session"])),
    }


def first_pass(source: str, queries: list[str], clicks: list[str], shopper: str | None = None,
               views: list[str] | None = None) -> dict:
    """Step 2 of the explorer's flow: the list before any session re-rank.

    search    products by MPNet similarity (name + description, like system A) to the search
              terms, each term z-scored then averaged; or to the clicked items if no terms
    shopper   a mock shopper's offline "Picked for you" scores
    Products already clicked or viewed are left out, as the re-rank does."""
    clicks = [s for s in clicks if s in IDX]
    seen = set(clicks) | {s for s in (views or []) if s in IDX}
    queries = [q.strip() for q in queries if q and q.strip()]
    if source == "search":
        V = variant("A").V
        if queries:
            scores = np.mean([variant("A").query_scores(q) for q in queries], axis=0)
            how = "MPNet similarity to your search terms"
        elif clicks:
            c = V[[IDX[s] for s in clicks]].mean(0)
            scores = V @ (c / np.linalg.norm(c))
            how = "MPNet similarity to your clicked items (no search terms)"
        else:
            raise ValueError("Add a search term or a clicked item first.")
    elif shopper in SHOPPERS:
        scores = offline_scores(SHOPPERS[shopper], list(SHOPPERS).index(shopper))
        how = f"{shopper}'s offline Picked for you scores"
    else:
        raise ValueError("Choose search results or a shopper's Picked for you list.")
    order = [SKUS[i] for i in np.argsort(-scores) if SKUS[i] not in seen]
    return {"order": order, "how": how}


def warm() -> None:
    """Load the offered systems once so the first click in the explorer is quick."""
    for key in ACTIVE:
        anchor_scores(key, [], [], ["warm up"])
    variant("A")  # the first pass's search encoder
