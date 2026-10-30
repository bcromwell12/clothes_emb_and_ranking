# Product embeddings and session re-ranking

An experiment in product embeddings, taxonomy-aware fine-tuning, and real-time session re-ranking on a small mock e-commerce catalog, with an interactive explorer to poke at all of it.

> **Mock data.** The 106 products, prices, margins, shoppers and sessions are made up for a case study. They are not pulled from Quince, and this project is not affiliated with Quince.

## What's here

1. **A mock catalog**: 106 products in five departments (cashmere sweaters, clothing, bags, furniture & bed, and 26 jewelry pieces), with backend-style descriptions, prices, made-up margins and a three-level taxonomy (department → type → material, audience, occasion and price band).
2. **Description embeddings**: raw BERT (mean-pooled) vs. MPNet, compared with cosine statistics, KMeans and agglomerative clustering, and UMAP plots. MPNet separates the departments much better (top-5 neighbor purity 0.80 vs. 0.66, KMeans ARI 0.59 vs. 0.44).
3. **A masked set model**: the taxonomy as "menus" (`[Jewelry] [Rings] [Gold] → every matching product`). Frozen MPNet vectors go through a small adapter. A transformer *without positional encoding* learns two tasks: hide a product and predict it, and hide a header slot and name it. Tested on menus held out of training entirely:
   - It fills in unseen menus far better than raw MPNet: mAP 0.89 vs. 0.49 from the header alone.
   - It names the category of a three-product list correctly 98% of the time, vs. 62% for raw MPNet.
4. **Session re-ranking**: a shopper's clicks and searches re-rank a list. The same test (11 shopping intents × 10 sessions × 5 shoppers) compares five systems. The measure is next-click hit@10: whether the shopper's next click lands in the re-ranked top 10. Without any re-rank it's 13%.

   | System | hit@10 | Search recall@10 |
   |---|---:|---:|
   | BGE + cross-encoder | **66%** | **89%** |
   | BGE | 64% | 80% |
   | MPNet + attribute tags | 62% | 74% |
   | MPNet | 61% | 76% |
   | Set model | 59% | 74% |

   The set model is best on mixed "complete the look" sessions (wedding outfit, travel, necklaces, furniture). It's weak on single-type sessions such as sweaters.
5. **An explorer**: a single-page app showing the embedding map, a diversity/margin ranker, a three-step session re-rank, side-by-side comparison of re-rankers, a new-product sandbox, and coloring by taxonomy facet.

Full numbers: [`outputs/report.md`](outputs/report.md), [`outputs/set_model_report.md`](outputs/set_model_report.md), [`outputs/rerank_improvements.log`](outputs/rerank_improvements.log).

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv): `uv venv --python 3.12 && uv pip install -r requirements.txt`.

The first run downloads the encoders from Hugging Face, about 1 GB in total:
- `bert-base-uncased`
- `all-mpnet-base-v2`
- `bge-small-en-v1.5`
- `ms-marco-MiniLM-L-6-v2`

## Run the explorer

The computed outputs are checked in, so the explorer works straight away:

```bash
.venv/bin/python quince_explorer_server.py      # http://localhost:8765
```

The server takes about 20 seconds to load the models. Opening `explorer/index.html` directly also works, but without the server you only get the static parts: the map, ranker and diagnostics. Typed queries, the session re-rank and the sandbox need the server.

**Using the session re-rank** (the section below the map):
1. **Session.** Add clicked items and search terms, or load an example session.
2. **First pass.** Build the list the shopper sees first: search results for the terms, or a mock shopper's "Picked for you" list.
3. **Re-rank.** Pick a system and a session weight, and optionally apply the diversity/margin knobs afterwards. Then re-rank or compare both systems side by side.

Each re-ranked row shows how much of its score came from the list and how much from the session. The map draws the session anchors and what they pulled in.

## Rebuild everything

Run these in order:

```bash
.venv/bin/python quince_embeddings.py           # embeddings, Word file of descriptions, report, plots
.venv/bin/python quince_set_model.py            # train + evaluate the masked set model (~5 min on a laptop CPU)
.venv/bin/python quince_explorer_data.py        # UMAP maps and explorer/explorer_data.js
.venv/bin/python quince_rerank_improvements.py  # the five-system re-rank comparison
```

Try the set model from the command line:

```bash
.venv/bin/python quince_set_model.py --complete Jewelry Rings Gold          # fill in a menu
.venv/bin/python quince_set_model.py --name-list JWL-008 JWL-013 JWL-021    # name a list's categories
.venv/bin/python quince_set_model.py --query "cozy gift for mom"
```

## Files

| File | What it does |
|---|---|
| `quince_catalog.py` | The mock catalog (`Product` dataclass, deterministic mock fields and margins) |
| `quince_taxonomy.py` | Department → type → facets, by keyword rules over the product text |
| `quince_embeddings.py` | BERT / MPNet embeddings, similarity, clustering, plots, report |
| `quince_set_model.py` | Masked set model, skip-gram baseline, held-out evaluation |
| `quince_session_rerank.py` | The toy session re-rank: offline shortlist + click and search anchors |
| `quince_item_tags.py` | Hand-written season / formality / occasion tags per product |
| `quince_rerank_improvements.py` | Five re-rank systems compared on many simulated sessions |
| `quince_rerank_service.py` | First pass + re-rank as functions the explorer server calls |
| `quince_explorer_data.py` | Exports maps, neighbors and clusters for the explorer |
| `quince_explorer_server.py` | Serves the explorer and its API (embed, first pass, re-rank, new product) |
| `explorer/` | The explorer page and its data |
| `outputs/` | Embeddings, fitted UMAPs, set-model weights, reports, plots, the Word file |

## Caveats

- Everything is synthetic. The taxonomy is derived from the same text the encoders read, so some results are partly circular; the reports say where.
- The re-rank test's "next click" labels come from the taxonomy, and its shoppers and sessions are simulated.
- With "search results" as the first pass, the re-rank mostly reorders. The search is already in the first pass, so the re-rank has the most effect on a shopper's "Picked for you" list.
