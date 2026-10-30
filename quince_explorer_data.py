"""
Export the saved description embeddings to explorer/explorer_data.js for the
interactive explorer page (explorer/index.html).

Run quince_embeddings.py first so outputs/embeddings_<model>.npy exist.

Usage:
    .venv/bin/python quince_explorer_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from quince_catalog import CATEGORIES, PRODUCTS
from quince_embeddings import MODELS

# encoders shown in the explorer; the tuned space appears once quince_set_model.py has been run
EXTRA = {"mpnet-set": "MPNet + masked set model adapter"}


def encoder_keys() -> list[str]:
    return list(MODELS) + [k for k in EXTRA if (ROOT / "outputs" / f"embeddings_{k}.npy").exists()]

ROOT = Path(__file__).parent
OUT = ROOT / "explorer" / "explorer_data.js"


def model_payload(key: str, labels: np.ndarray) -> dict:
    emb = np.load(ROOT / "outputs" / f"embeddings_{key}.npy")
    sim = emb @ emb.T
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=0)
    xy = reducer.fit_transform(emb)
    lo, hi = xy.min(0), xy.max(0)
    xy = (xy - lo) / (hi - lo)  # normalize to [0, 1]
    # keep the fitted projection so the server can place new queries on the same map
    joblib.dump({"umap": reducer, "lo": lo, "hi": hi}, ROOT / "outputs" / f"umap_{key}.joblib")

    sweep = {k: silhouette_score(emb, KMeans(k, n_init=10, random_state=0).fit_predict(emb), metric="cosine")
             for k in range(2, 13)}
    best_k = max(sweep, key=sweep.get)
    km5 = KMeans(5, n_init=10, random_state=0).fit_predict(emb)
    km_best = KMeans(best_k, n_init=10, random_state=0).fit_predict(emb)

    neighbors = []
    for i in range(len(emb)):
        order = [j for j in np.argsort(-sim[i]) if j != i][:10]
        neighbors.append(order)
    purity = np.mean([np.mean(labels[n[:5]] == labels[i]) for i, n in enumerate(neighbors)])

    cats = list(CATEGORIES)
    cat_sim = []
    for a in cats:
        row = []
        for b in cats:
            block = sim[np.ix_(labels == a, labels == b)]
            if a == b:
                block = block[~np.eye(len(block), dtype=bool)]
            row.append(round(float(block.mean()), 3))
        cat_sim.append(row)

    off = sim[~np.eye(len(sim), dtype=bool)]
    return {
        "label": MODELS[key][0] if key in MODELS else EXTRA[key],
        "xy": np.round(xy, 4).tolist(),
        "sim": np.round(sim, 3).tolist(),
        "neighbors": [[int(j) for j in n] for n in neighbors],
        "kmeans5": km5.tolist(),
        "kmeansBest": km_best.tolist(),
        "bestK": int(best_k),
        "silhouette": {int(k): round(float(v), 3) for k, v in sweep.items()},
        "catSim": cat_sim,
        "stats": {
            "cosMin": round(float(off.min()), 3),
            "cosMean": round(float(off.mean()), 3),
            "cosMax": round(float(off.max()), 3),
            "purity5": round(float(purity), 3),
            "ari5": round(float(adjusted_rand_score(labels, km5)), 3),
        },
    }


def facets() -> dict[str, list[str]]:
    """Taxonomy facets per product (first value where a product has several), for coloring the map."""
    from quince_taxonomy import TAXONOMY
    return {"type": [t.type for t in TAXONOMY], "material": [t.materials[0] for t in TAXONOMY],
            "occasion": [t.occasions[0] for t in TAXONOMY], "audience": [t.audience for t in TAXONOMY],
            "price": [t.price_band for t in TAXONOMY]}


def main() -> None:
    labels = np.array([p.category for p in PRODUCTS])
    data = {
        "categories": [{"key": k, "label": v["label"]} for k, v in CATEGORIES.items()],
        "products": [{"sku": p.sku, "name": p.name, "category": p.category, "subcategory": p.subcategory,
                      "price": p.price, "compareAt": p.compare_at_price, "unitCost": p.unit_cost,
                      "marginPct": p.margin_pct, "marginDollars": p.margin_dollars, "materials": p.materials,
                      "description": p.description} for p in PRODUCTS],
        "facets": facets(),
        "models": {key: model_payload(key, labels) for key in encoder_keys()},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("window.QUINCE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
