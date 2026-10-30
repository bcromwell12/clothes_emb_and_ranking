"""
Toy session rerank: does "encode the clicks and the search with a BERT-family encoder,
then rerank each shopper's offline shortlist by cosine" actually work?

Setup (all made up, on the 100-product mock catalog):
  - Five shoppers, each with offline category affinities (what nightly MF / the ranker
    learned). Their offline ranker score is log(affinity) + popularity + noise, and the
    top K becomes their precomputed "Picked for you" shortlist.
  - Everybody then does the SAME session (Section 4.6 trace): clicks two linen pieces,
    searches "summer wedding guest outfit", views two items.
  - The session pass builds two anchors (click anchor, query anchor), adds the nearest
    items to each anchor to the shortlist, and blends z(ranker) with z(sim to closer anchor).

Checks:
  1. Before/after rows for each shopper, tagged by category and which anchor pulled them.
  2. Session relevance of the top 10, against a hand-labeled outfit set fixed up front.
  3. Personalization kept: rows across shoppers should get closer, not identical.
  4. Placebo session (random clicks + an unrelated query) should NOT pull toward outfits.
  5. Variants (raw cosine / per-anchor z-score / + category guard), lambda sweep, and
     a raw bert-base vs MPNet comparison (--encoder).

Usage:
    .venv/bin/python quince_session_rerank.py
    .venv/bin/python quince_session_rerank.py --encoder bert-base
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from quince_catalog import PRODUCTS, PRODUCTS_BY_SKU
from quince_embeddings import embed

SKUS = [p.sku for p in PRODUCTS]
IDX = {s: i for i, s in enumerate(SKUS)}
CATS = np.array([p.category for p in PRODUCTS])
SHORT = {"cashmere_sweaters": "CSH", "clothing": "CLO", "bags": "BAG", "furniture_bed": "HOM", "jewelry": "JWL"}

# Offline category affinities ("assume we have some categories for everybody").
SHOPPERS = {
    "cashmere loyalist": {"cashmere_sweaters": .60, "clothing": .20, "bags": .10, "furniture_bed": .05, "jewelry": .05},
    "home nester":       {"cashmere_sweaters": .10, "clothing": .05, "bags": .05, "furniture_bed": .70, "jewelry": .10},
    "jewelry gifter":    {"cashmere_sweaters": .10, "clothing": .10, "bags": .10, "furniture_bed": .05, "jewelry": .65},
    "bag buyer":         {"cashmere_sweaters": .10, "clothing": .15, "bags": .60, "furniture_bed": .05, "jewelry": .10},
    "cold start":        {c: .20 for c in SHORT},  # no history: popularity only, i.e. the bestseller list
}

# The shared session. Clicks count more than views (4.6 decision: clicks are the more useful data).
SESSION = {
    "clicks": ["CLO-003", "CLO-004"],          # linen wide-leg pants, linen shirt
    "query": "summer wedding guest outfit",
    "views": ["CLO-009", "CLO-007"],           # Tencel midi dress, silk slip skirt
}
PLACEBO = {"clicks": ["HOM-011", "BAG-013"], "query": "home office storage", "views": ["CSH-020"]}
W_CLICK, W_VIEW = 1.0, 0.5

# Hand labels, fixed before looking at any results: items a summer wedding guest with a
# linen streak would plausibly want. Linen *bedding* is tracked separately as the obvious
# failure mode (same material word, wrong intent).
OUTFIT = {"CLO-001", "CLO-002", "CLO-014", "CLO-015", "CLO-020", "BAG-014", "BAG-018",
          "JWL-001", "JWL-002", "JWL-003", "JWL-005", "JWL-006", "JWL-010", "JWL-015"}
LINEN_HOME = {"HOM-001", "HOM-007", "HOM-013", "HOM-014"}

K_SHORTLIST, M_RETRIEVE, ROW, PINNED = 30, 10, 10, 2


def item_vectors(encoder: str) -> np.ndarray:
    return embed(encoder, [f"{p.name}. {p.description}" for p in PRODUCTS])


def offline_scores(affinity: dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pop = np.log([p.review_count for p in PRODUCTS])
    pop = (pop - pop.mean()) / pop.std()
    return np.log([affinity[c] for c in CATS]) + 0.3 * pop + 0.2 * rng.standard_normal(len(PRODUCTS))


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def anchors(session: dict, V: np.ndarray, encoder: str) -> dict[str, np.ndarray]:
    w = [(s, W_CLICK) for s in session["clicks"]] + [(s, W_VIEW) for s in session["views"]]
    click = sum(wt * V[IDX[s]] for s, wt in w)
    return {"click": click / np.linalg.norm(click), "query": embed(encoder, [session["query"]])[0]}


def anchor_sims(V: np.ndarray, anc: dict, session: dict, mode: str) -> np.ndarray:
    """anchors × items similarity.
    raw:      plain cosine. Item-item cosines (~0.8) dwarf query-item ones (~0.3), so the
              click anchor always wins the max and the search never steers anything.
    zscore:   standardize each anchor's cosines over the catalog first (what the doc specifies).
    guarded:  zscore, plus the click anchor only boosts the clicked items' categories, so
              "linen" clicks on clothing don't pull in linen duvets.
    """
    sims = np.stack([V @ a for a in anc.values()])
    if mode == "raw":
        return sims
    sims = (sims - sims.mean(1, keepdims=True)) / sims.std(1, keepdims=True)
    if mode == "guarded":
        clicked_cats = {PRODUCTS_BY_SKU[s].category for s in session["clicks"] + session["views"]}
        sims[0, ~np.isin(CATS, list(clicked_cats))] = sims[0].min()
    return sims


def session_rerank(ranker: np.ndarray, sims: np.ndarray, anc: dict, seen: set[str], lam: float):
    """Shortlist ∪ nearest-M per anchor, scored by (1-λ)·z(ranker) + λ·z(sim to closer anchor)."""
    shortlist = [i for i in np.argsort(-ranker) if SKUS[i] not in seen][:K_SHORTLIST]
    retrieved = {int(i) for row in sims for i in np.argsort(-row)[:M_RETRIEVE + len(seen)]}
    cand = np.array(sorted(set(shortlist) | {i for i in retrieved if SKUS[i] not in seen}))
    best = sims[:, cand].max(0)
    which = np.array(list(anc))[sims[:, cand].argmax(0)]
    score = (1 - lam) * z(ranker[cand]) + lam * z(best)
    order = cand[np.argsort(-score)]
    pinned = shortlist[:PINNED]                                   # first two slots stay put
    row = pinned + [i for i in order if i not in pinned][:ROW - PINNED]
    tag = dict(zip(cand, which))
    return shortlist[:ROW], row, {i: tag.get(i, "-") for i in row}


def relevance(row) -> tuple[float, float]:
    names = [SKUS[i] for i in row]
    return np.mean([s in OUTFIT for s in names]), np.mean([s in LINEN_HOME for s in names])


def jaccard_mean(rows: list[list[int]]) -> float:
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(rows, 2)]))


def run(V, encoder, session, lam, verbose, mode="guarded"):
    anc = anchors(session, V, encoder)
    sims = anchor_sims(V, anc, session, mode)
    seen = set(session["clicks"]) | set(session["views"])
    befores, afters, stats = [], [], []
    for seed, (name, aff) in enumerate(SHOPPERS.items()):
        before, after, tags = session_rerank(offline_scores(aff, seed), sims, anc, seen, lam)
        befores.append(before); afters.append(after)
        stats.append((name, *relevance(before), *relevance(after)))
        if verbose:
            print(f"\n=== {name} ===")
            print(f"  {'before (offline shortlist)':<48} after (session rerank, λ={lam})")
            for b, a in zip(before, after):
                pb, pa = PRODUCTS[b], PRODUCTS[a]
                mark = "*" if pa.sku in OUTFIT else ("!" if pa.sku in LINEN_HOME else " ")
                print(f"  {pb.sku} {pb.name[:38]:<40}  {mark}{pa.sku} {pa.name[:38]:<40} [{tags[a]}]")
    return befores, afters, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="mpnet", choices=["mpnet", "bert-base"])
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--mode", default="guarded", choices=["raw", "zscore", "guarded"])
    args = ap.parse_args()
    V = item_vectors(args.encoder)

    anc = anchors(SESSION, V, args.encoder)
    print(f"Encoder: {args.encoder}.  Nearest items to each anchor:")
    for k, a in anc.items():
        top = np.argsort(-(V @ a))[:6]
        print(f"  {k:<6} " + ", ".join(f"{SKUS[i]} {PRODUCTS[i].name[:24]} ({V[i] @ a:.2f})" for i in top))

    befores, afters, stats = run(V, args.encoder, SESSION, args.lam, verbose=True, mode=args.mode)
    print("\n* = hand-labeled outfit item, ! = linen home item (material match, wrong intent)")

    print(f"\n{'shopper':<18} {'outfit@10 before':>16} {'after':>6} {'linen-home before':>18} {'after':>6}")
    for name, ob, lb, oa, la in stats:
        print(f"{name:<18} {ob:>16.0%} {oa:>6.0%} {lb:>18.0%} {la:>6.0%}")
    print(f"\nRow overlap across shoppers (mean pairwise Jaccard): before {jaccard_mean(befores):.2f}, "
          f"after {jaccard_mean(afters):.2f}")
    print("Category mix after rerank:")
    for (name, *_), row in zip(stats, afters):
        mix = {SHORT[c]: int((CATS[row] == c).sum()) for c in SHORT}
        print(f"  {name:<18} " + "  ".join(f"{k}:{v}" for k, v in mix.items() if v))

    print("\nVariant comparison (mean over shoppers): outfit@10 / linen-home@10 / rows steered by query anchor")
    for mode in ["raw", "zscore", "guarded"]:
        _, a, s = run(V, args.encoder, SESSION, args.lam, verbose=False, mode=mode)
        anc = anchors(SESSION, V, args.encoder)
        sims = anchor_sims(V, anc, SESSION, mode)
        q_share = np.mean([(sims[:, r].argmax(0) == 1).mean() for r in a])
        print(f"  {mode:<8} outfit {np.mean([x[3] for x in s]):>4.0%}   linen-home {np.mean([x[4] for x in s]):>4.0%}   "
              f"query-anchor slots {q_share:>4.0%}")

    _, p_after, p_stats = run(V, args.encoder, PLACEBO, args.lam, verbose=False, mode=args.mode)
    print(f"\nMode: {args.mode}.  Placebo session ({PLACEBO['query']!r} + oak bed / briefcase clicks): outfit@10 after = "
          f"{np.mean([s[3] for s in p_stats]):.0%} (real session: {np.mean([s[3] for s in stats]):.0%})")

    print("\nλ sweep (session weight): mean outfit@10 / mean overlap with original shortlist / cross-shopper Jaccard")
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        b, a, s = run(V, args.encoder, SESSION, lam, verbose=False, mode=args.mode)
        keep = np.mean([len(set(x) & set(y)) / ROW for x, y in zip(b, a)])
        print(f"  λ={lam:.1f}  outfit {np.mean([r[3] for r in s]):>4.0%}   kept {keep:>4.0%}   jaccard {jaccard_mean(a):.2f}")


if __name__ == "__main__":
    main()
