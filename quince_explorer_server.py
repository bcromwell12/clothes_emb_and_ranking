"""
Local server for the embedding explorer: serves explorer/ and embeds free-text queries.

Endpoints:
    GET  /                 the explorer page (explorer/index.html)
    GET  /api/health       {"ok": true, "models": [...]}
    POST /api/embed        {"text": "..."} -> per model: map position, top-10 products, cosine to all products
    GET  /api/rerank/options  re-rank systems and mock shoppers (quince_rerank_service.py)
    POST /api/rerank       {"system", "lam", "clicks": [sku], "views": [sku], "queries": [text], "order": [sku] | "shopper"}
                           -> before/after rows with each row's score split into list and session parts
    POST /api/first-pass   {"source": "search" | "shopper", "queries": [text], "clicks": [sku], "shopper"}
                           -> the list before re-ranking, as an ordered list of SKUs
    POST /api/new-product  {"name", "description", "query"} -> where a made-up product lands in each space,
                           its nearest products, its rank for the query, and the set model's names for it

Queries are embedded with exactly the same encoders and pooling as the products
(quince_embeddings.embed) and placed on the map with the fitted UMAP models saved
by quince_explorer_data.py. In the tuned "mpnet-set" space the query is the set model's
prediction for a menu headed by the query text.

Usage:
    .venv/bin/python quince_explorer_server.py            # http://localhost:8765
    .venv/bin/python quince_explorer_server.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

from quince_embeddings import MODELS, embed
from quince_explorer_data import encoder_keys
import quince_rerank_service as rerank_service

ROOT = Path(__file__).parent
STATIC = ROOT / "explorer"
MAX_QUERY_CHARS = 1000

KEYS = encoder_keys()
PRODUCT_EMB = {key: np.load(ROOT / "outputs" / f"embeddings_{key}.npy") for key in KEYS}
UMAPS = {key: joblib.load(ROOT / "outputs" / f"umap_{key}.joblib") for key in KEYS}
LOCK = threading.Lock()  # one embedding at a time; torch and numba are happier that way


def query_vector(text: str) -> np.ndarray:
    """Tuned space: the set model completes [MASK] [QUERY: text] (see quince_set_model.py)."""
    from quince_set_model import query_vector as complete_query
    return complete_query(text)


def new_product(text: str, query: str) -> dict:
    """Place a made-up product (embedded like the catalog descriptions) in every space."""
    from quince_set_model import adapt, name_vector
    raw_mpnet = embed("mpnet", [text])[0]
    models = {}
    for key in KEYS:
        if key == "mpnet-set":
            v = adapt(raw_mpnet)[0]
            q = query_vector(query) if query else None
        else:
            v = raw_mpnet if key == "mpnet" else embed(key, [text])[0]
            q = embed(key, [query])[0] if query else None
        sims = PRODUCT_EMB[key] @ v
        u = UMAPS[key]
        xy = (u["umap"].transform(v[None, :])[0] - u["lo"]) / (u["hi"] - u["lo"])
        top = np.argsort(-sims)[:5]
        entry = {"xy": np.clip(xy, -0.03, 1.03).astype(float).round(4).tolist(),
                 "neighbors": top.tolist(), "sims": sims[top].astype(float).round(3).tolist()}
        if q is not None:
            mine = float(v @ q)
            entry["query"] = {"sim": round(mine, 3), "rank": 1 + int((PRODUCT_EMB[key] @ q > mine).sum()),
                              "of": len(sims) + 1}
        models[key] = entry
    return {"models": models, "names": name_vector(raw_mpnet) if "mpnet-set" in KEYS else {}}


def embed_query(text: str) -> dict:
    out = {}
    with LOCK:
        for key in KEYS:
            q = query_vector(text) if key == "mpnet-set" else embed(key, [text])[0]
            sims = PRODUCT_EMB[key] @ q
            u = UMAPS[key]
            xy = (u["umap"].transform(q[None, :])[0] - u["lo"]) / (u["hi"] - u["lo"])
            top = np.argsort(-sims)[:10]
            out[key] = {
                "xy": np.clip(xy, -0.03, 1.03).astype(float).round(4).tolist(),
                "sim": sims.astype(float).round(3).tolist(),
                "neighbors": top.tolist(),
            }
    return out


def str_list(body: dict, key: str) -> list[str]:
    """A JSON list of strings, or a clear error (a bare string would otherwise split into letters)."""
    v = body.get(key, [])
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"{key} must be a list of strings.")
    return v


class Handler(SimpleHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            return self._json(200, {"ok": True, "models": KEYS})
        if self.path == "/api/rerank/options":
            return self._json(200, rerank_service.options())
        super().do_GET()

    def do_POST(self) -> None:
        if self.path not in ("/api/embed", "/api/rerank", "/api/first-pass", "/api/new-product"):
            return self._json(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Send a JSON body."})
        if self.path == "/api/rerank":
            return self._rerank(body)
        if self.path == "/api/new-product":
            return self._new_product(body)
        if self.path == "/api/first-pass":
            return self._first_pass(body)
        text = str(body.get("text", "")).strip()
        if not text:
            return self._json(400, {"error": "Type a query to embed."})
        if len(text) > MAX_QUERY_CHARS:
            return self._json(400, {"error": f"Keep queries under {MAX_QUERY_CHARS} characters."})
        self._json(200, {"text": text, "models": embed_query(text)})

    def _first_pass(self, body: dict) -> None:
        try:
            queries = [t[:MAX_QUERY_CHARS] for t in str_list(body, "queries")][:10]
            with LOCK:
                out = rerank_service.first_pass(str(body.get("source", "")), queries, str_list(body, "clicks")[:10],
                                                body.get("shopper"), views=str_list(body, "views")[:10])
        except (ValueError, TypeError) as err:
            return self._json(400, {"error": str(err)})
        self._json(200, out)

    def _new_product(self, body: dict) -> None:
        name = str(body.get("name") or "").strip()[:200]
        text = str(body.get("description") or "").strip()[:MAX_QUERY_CHARS] or name
        if not text:
            return self._json(400, {"error": "Give the new product a name or a description."})
        query = str(body.get("query") or "").strip()[:MAX_QUERY_CHARS]
        with LOCK:
            out = new_product(text, query)
        self._json(200, {"name": name or text[:60], "description": text, "query": query, **out})

    def _rerank(self, body: dict) -> None:
        try:
            try:
                lam = min(1.0, max(0.0, float(body.get("lam", rerank_service.DEFAULT_LAM))))
            except (TypeError, ValueError):
                raise ValueError("lam must be a number between 0 and 1.")
            queries = [t[:MAX_QUERY_CHARS] for t in str_list(body, "queries")][:10]
            order = str_list(body, "order")
            with LOCK:
                out = rerank_service.rerank(str(body.get("system", "")), str_list(body, "clicks")[:10],
                                            queries, lam, order=order or None, shopper=body.get("shopper"),
                                            views=str_list(body, "views")[:10])
        except (ValueError, TypeError) as err:
            return self._json(400, {"error": str(err)})
        self._json(200, out)

    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in self.path:
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print("Loading encoders and warming up UMAP...", flush=True)
    embed_query("warm up")
    rerank_service.warm()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(STATIC)))
    print(f"Explorer running at http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
