"""
Embed the Quince catalog descriptions with BERT-family encoders and explore the space.

Pipeline:
  1. Dump every product description into a Word doc (outputs/quince_descriptions.docx)
  2. Embed descriptions with each encoder in MODELS
  3. Cosine similarity: intra- vs inter-category, nearest neighbors, surprising pairs
  4. Clustering: KMeans silhouette sweep + k=5 vs true categories (ARI / NMI)
  5. Plots: UMAP projection and sorted similarity heatmap
  6. Markdown report: outputs/report.md

Usage:
    .venv/bin/python quince_embeddings.py
    .venv/bin/python quince_embeddings.py --text name+description
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from transformers import AutoModel, AutoTokenizer

from quince_catalog import CATEGORIES, PRODUCTS

OUT = Path(__file__).parent / "outputs"

# key -> (hf model id, kind). "raw" = vanilla BERT with mean pooling,
# "st" = sentence-transformers model (BERT-family, fine-tuned for similarity).
MODELS = {
    "bert-base": ("bert-base-uncased", "raw"),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "st"),
}

CAT_COLORS = {
    "cashmere_sweaters": "#c2185b",
    "clothing": "#1976d2",
    "bags": "#8d6e63",
    "furniture_bed": "#388e3c",
    "jewelry": "#f9a825",
}


# ---------------------------------------------------------------------------
# 1. Word doc
# ---------------------------------------------------------------------------

def write_docx(df: pd.DataFrame, path: Path) -> None:
    doc = Document()
    doc.add_heading("Quince Catalog - Product Descriptions", level=0)
    for cat, group in df.groupby("category", sort=False):
        doc.add_heading(CATEGORIES[cat]["label"], level=1)
        for row in group.itertuples():
            p = doc.add_paragraph()
            p.add_run(f"{row.sku}  {row.name}").bold = True
            p.add_run(f"  (${row.price:,.0f})")
            doc.add_paragraph(row.description)
    doc.save(path)


# ---------------------------------------------------------------------------
# 2. Embeddings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_encoder(key: str):
    """Load (and cache) the tokenizer/model for a MODELS key, so repeat queries are fast."""
    model_id, kind = MODELS[key]
    if kind == "raw":
        return AutoTokenizer.from_pretrained(model_id), AutoModel.from_pretrained(model_id).eval()
    return SentenceTransformer(model_id)


def embed_raw_bert(key: str, texts: list[str]) -> np.ndarray:
    """Vanilla BERT: mean-pool the last hidden layer over non-padding tokens."""
    tok, model = load_encoder(key)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tok(texts[i:i + 16], padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())
    return np.vstack(out)


def embed(key: str, texts: list[str]) -> np.ndarray:
    if MODELS[key][1] == "raw":
        emb = embed_raw_bert(key, texts)
    else:
        emb = load_encoder(key).encode(texts, batch_size=16, show_progress_bar=False)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)  # unit-norm => dot product = cosine


# ---------------------------------------------------------------------------
# 3. Similarity analysis
# ---------------------------------------------------------------------------

def similarity_stats(sim: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Mean cosine similarity between every pair of categories (diagonal excludes self-pairs)."""
    cats = list(CATEGORIES)
    table = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for a in cats:
        for b in cats:
            block = sim[np.ix_(labels == a, labels == b)]
            if a == b:
                block = block[~np.eye(len(block), dtype=bool)]
            table.loc[a, b] = block.mean()
    return table


def nearest_neighbors(sim: np.ndarray, df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        order = np.argsort(-sim[i])
        order = order[order != i][:k]
        rows.append({
            "sku": df.sku[i], "name": df.name[i], "category": df.category[i],
            "neighbors": [(df.name[j], df.category[j], round(float(sim[i, j]), 3)) for j in order],
            "neighbor_purity": float(np.mean(df.category.values[order] == df.category[i])),
        })
    return pd.DataFrame(rows)


def cross_category_pairs(sim: np.ndarray, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most similar pairs of products that live in *different* categories."""
    iu = np.triu_indices(len(df), k=1)
    pairs = pd.DataFrame({"i": iu[0], "j": iu[1], "sim": sim[iu]})
    pairs = pairs[df.category.values[pairs.i] != df.category.values[pairs.j]]
    pairs = pairs.nlargest(n, "sim")
    return pd.DataFrame({
        "a": df.name.values[pairs.i], "a_cat": df.category.values[pairs.i],
        "b": df.name.values[pairs.j], "b_cat": df.category.values[pairs.j],
        "cosine": pairs.sim.round(3).values,
    })


# ---------------------------------------------------------------------------
# 4. Clustering
# ---------------------------------------------------------------------------

def clustering(emb: np.ndarray, labels: np.ndarray, df: pd.DataFrame) -> dict:
    sweep = {k: silhouette_score(emb, KMeans(k, n_init=10, random_state=0).fit_predict(emb), metric="cosine")
             for k in range(2, 13)}
    km5 = KMeans(5, n_init=10, random_state=0).fit_predict(emb)
    agg5 = AgglomerativeClustering(5, metric="cosine", linkage="average").fit_predict(emb)
    best_k = max(sweep, key=sweep.get)
    km_best = KMeans(best_k, n_init=10, random_state=0).fit_predict(emb)

    def cluster_members(assign: np.ndarray) -> dict[int, list[str]]:
        return {c: df.name[assign == c].tolist() for c in np.unique(assign)}

    return {
        "silhouette_sweep": sweep,
        "best_k": best_k,
        "kmeans5_ari": adjusted_rand_score(labels, km5),
        "kmeans5_nmi": normalized_mutual_info_score(labels, km5),
        "agg5_ari": adjusted_rand_score(labels, agg5),
        "agg5_nmi": normalized_mutual_info_score(labels, agg5),
        "kmeans5_crosstab": pd.crosstab(pd.Series(labels, name="category"), pd.Series(km5, name="cluster")),
        "kmeans5": km5,
        "kmeans_best_members": cluster_members(km_best),
    }


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_umap(emb: np.ndarray, df: pd.DataFrame, clusters: np.ndarray, title: str, path: Path) -> None:
    xy = umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine", random_state=0).fit_transform(emb)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for cat, color in CAT_COLORS.items():
        m = df.category.values == cat
        axes[0].scatter(xy[m, 0], xy[m, 1], c=color, label=CATEGORIES[cat]["label"], s=45, alpha=0.85)
    for i, name in enumerate(df.name):
        axes[0].annotate(name.replace("Mongolian Cashmere ", "MC ").replace("Italian Leather ", "IL ")[:28],
                         xy[i], fontsize=5.5, alpha=0.7)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{title}: colored by true category")
    sc = axes[1].scatter(xy[:, 0], xy[:, 1], c=clusters, cmap="tab10", s=45)
    axes[1].legend(*sc.legend_elements(), title="KMeans k=5", fontsize=8)
    axes[1].set_title(f"{title}: colored by KMeans cluster")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_heatmap(sim: np.ndarray, df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim, cmap="viridis")
    bounds = np.cumsum(df.category.value_counts(sort=False).values)[:-1] - 0.5
    for b in bounds:
        ax.axhline(b, color="white", lw=0.8); ax.axvline(b, color="white", lw=0.8)
    ticks = np.arange(10, 100, 20)
    ax.set_xticks(ticks, [CATEGORIES[c]["label"] for c in CATEGORIES], rotation=30, fontsize=8)
    ax.set_yticks(ticks, [CATEGORIES[c]["label"] for c in CATEGORIES], fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(f"{title}: pairwise cosine similarity (sorted by category)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------

def run_model(key: str, df: pd.DataFrame, texts: list[str]) -> tuple[str, dict]:
    labels = df.category.values
    emb = embed(key, texts)
    np.save(OUT / f"embeddings_{key}.npy", emb)
    sim = emb @ emb.T

    cat_sim = similarity_stats(sim, labels)
    intra = np.diag(cat_sim.values).mean()
    inter = cat_sim.values[~np.eye(5, dtype=bool)].mean()
    off_diag = sim[~np.eye(len(sim), dtype=bool)]
    nn = nearest_neighbors(sim, df)
    cross = cross_category_pairs(sim, df)
    cl = clustering(emb, labels, df)

    plot_umap(emb, df, cl["kmeans5"], key, OUT / f"umap_{key}.png")
    plot_heatmap(sim, df, key, OUT / f"heatmap_{key}.png")

    summary = {
        "model": MODELS[key][0], "dim": emb.shape[1],
        "cos_min": off_diag.min(), "cos_mean": off_diag.mean(), "cos_max": off_diag.max(),
        "intra": intra, "inter": inter, "gap": intra - inter,
        "nn_purity@5": nn.neighbor_purity.mean(),
        "kmeans5_ARI": cl["kmeans5_ari"], "kmeans5_NMI": cl["kmeans5_nmi"],
        "agg5_ARI": cl["agg5_ari"], "best_k_silhouette": cl["best_k"],
    }

    showcase = ["CSH-001", "CLO-002", "BAG-001", "HOM-001", "JWL-003", "CLO-019"]
    lines = [
        f"## {key} (`{MODELS[key][0]}`)",
        "",
        f"Embedding dim {emb.shape[1]}. Off-diagonal cosine: min {off_diag.min():.3f}, "
        f"mean {off_diag.mean():.3f}, max {off_diag.max():.3f}.",
        "",
        "### Mean cosine similarity between categories",
        "",
        cat_sim.round(3).to_markdown(),
        "",
        f"Intra-category mean **{intra:.3f}** vs inter-category mean **{inter:.3f}** (gap {intra - inter:.3f}).",
        f"Nearest-neighbor purity@5 (share of top-5 neighbors in the same category): **{nn.neighbor_purity.mean():.2%}**.",
        "",
        "Lowest-purity products (their neighbors mostly come from other categories):",
        "",
        nn.nsmallest(8, "neighbor_purity")[["sku", "name", "category", "neighbor_purity"]].to_markdown(index=False),
        "",
        "### Nearest neighbors for a few products",
        "",
    ]
    for sku in showcase:
        r = nn[nn.sku == sku].iloc[0]
        lines.append(f"**{r['name']}** ({r.category})")
        lines += [f"- {n} ({c}) - {s}" for n, c, s in r.neighbors]
        lines.append("")
    lines += [
        "### Most similar cross-category pairs",
        "",
        cross.to_markdown(index=False),
        "",
        "### Clustering",
        "",
        f"KMeans k=5 vs true categories: ARI **{cl['kmeans5_ari']:.3f}**, NMI **{cl['kmeans5_nmi']:.3f}**. "
        f"Agglomerative (cosine, average) k=5: ARI {cl['agg5_ari']:.3f}, NMI {cl['agg5_nmi']:.3f}.",
        "",
        "Category x KMeans cluster:",
        "",
        cl["kmeans5_crosstab"].to_markdown(),
        "",
        "Silhouette (cosine) by k: " + ", ".join(f"k={k}: {v:.3f}" for k, v in cl["silhouette_sweep"].items()),
        "",
        f"#### Clusters at the best-silhouette k={cl['best_k']}",
        "",
    ]
    for c, names in cl["kmeans_best_members"].items():
        lines.append(f"- **Cluster {c}** ({len(names)}): " + "; ".join(names))
    lines += ["", f"![umap](umap_{key}.png)", "", f"![heatmap](heatmap_{key}.png)", ""]
    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", choices=["description", "name+description"], default="description")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame([{"sku": p.sku, "name": p.name, "category": p.category, "subcategory": p.subcategory,
                        "price": p.price, "description": p.description} for p in PRODUCTS])
    texts = df.description.tolist() if args.text == "description" else (df.name + ". " + df.description).tolist()

    write_docx(df, OUT / "quince_descriptions.docx")
    df.to_csv(OUT / "products_meta.csv", index=False)

    sections, summaries = [], []
    for key in MODELS:
        print(f"Embedding with {key}...")
        section, summary = run_model(key, df, texts)
        sections.append(section)
        summaries.append(summary)

    report = [
        "# Quince description embeddings",
        "",
        f"{len(df)} products, 5 categories, embedded text = `{args.text}`.",
        "",
        "## Model comparison",
        "",
        pd.DataFrame(summaries).round(3).to_markdown(index=False),
        "",
        *sections,
    ]
    (OUT / "report.md").write_text("\n".join(report))
    if (OUT / "set_model_results.json").exists():  # keep the set-model section across reruns
        from quince_set_model import update_project_report
        update_project_report()
    print(pd.DataFrame(summaries).round(3).to_string(index=False))
    print(f"\nWrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
