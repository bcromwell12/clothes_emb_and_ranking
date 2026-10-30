"""
Masked set model over text embeddings. The goal is the embedding space; the prediction
tasks are only how it gets trained.

A menu is a header plus every product it lists, e.g.

    [Jewelry] [Earrings] [Gold]   ->   Huggie Hoops, Pearl Studs, Diamond Studs, ...

Every token is a frozen MPNet embedding (header value, product, or free-text query) passed
through a small shared adapter. Role embeddings say what each token is (department, type,
material, ..., item, query, and one mask role per thing that can be predicted), the way
BERT's segment embeddings work. No positional encoding: order on the list does not matter.

Two objectives, equally weighted:
    item     show the header and 0-7 of the listed products, predict a missing product
    header   show 1-7 listed products and some of the header, predict a hidden header slot
             (with the whole header hidden this is "here is a list, name its category")

Both use a full softmax over the candidates (no negative sampling; the catalog is small).
Every correct answer gets equal weight: each one is scored against the wrong answers only,
so the model cannot satisfy the loss by ranking a few list members high.

Every product is trained on. What is held out is a fifth of the multi-slot menus
(e.g. Jewelry + Rings + Gold), never shown in training, though each of their header values
appears elsewhere. Filling those in tests whether the space composes concepts it learned
separately.

Writes:
    outputs/set_model.pt                 weights for the multi-objective transformer
    outputs/embeddings_mpnet-set.npy     its adapted product embeddings (for the explorer)
    outputs/set_model_report.md          results tables and free-text examples
    outputs/set_model_results.json
    outputs/report.md                    adds or refreshes a summary section in the project report

Usage:
    .venv/bin/python quince_set_model.py                                  # train + evaluate
    .venv/bin/python quince_set_model.py --complete Jewelry Earrings --items JWL-001
    .venv/bin/python quince_set_model.py --name-list JWL-001 JWL-008 JWL-015
    .venv/bin/python quince_set_model.py --query "cozy gift for mom"
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quince_catalog import PRODUCTS, PRODUCTS_BY_SKU
from quince_embeddings import embed
from quince_taxonomy import TAXONOMY

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
BASE = "mpnet"
KEY = "mpnet-set"
WEIGHTS = OUT / "set_model.pt"
N = len(PRODUCTS)

SLOTS = ("department", "type", "material", "audience", "occasion", "price")
ROLES = ["item", "query", *SLOTS, "mask_item", *(f"mask_{s}" for s in SLOTS)]
R = {r: i for i, r in enumerate(ROLES)}
TEMPLATES = [
    ("department",), ("department", "type"), ("department", "type", "material"), ("department", "material"),
    ("department", "occasion"), ("department", "audience"), ("department", "type", "audience"),
    ("department", "type", "occasion"), ("department", "price"), ("material",), ("occasion",),
    ("material", "occasion"), ("occasion", "price"), ("audience", "material"),
]
MAX_CTX = 7
TAU = 0.05
QUERY_P = 0.3  # share of item-task examples whose header is written as one free-text phrase
STEPS = 2000
FREE_TEXT = ["cozy gift for mom", "something to wear to a wedding", "warm sweater for him to wear to the office",
             "summer vacation essentials", "gold jewelry under $100", "furnish a bedroom"]


# ---------------------------------------------------------------- menus and text bank

def product_slots(t) -> dict[str, tuple[str, ...]]:
    return {"department": (t.department,), "type": (t.type,), "material": t.materials,
            "audience": (t.audience,), "occasion": t.occasions, "price": (t.price_band,)}


SLOT_SETS = [{s: set(v) for s, v in product_slots(t).items()} for t in TAXONOMY]


def build_menus(min_items: int = 3) -> list[tuple[tuple, list[int]]]:
    """Every header combination from TEMPLATES -> the products it lists."""
    menus: dict[tuple, set[int]] = defaultdict(set)
    for i, t in enumerate(TAXONOMY):
        s = product_slots(t)
        for tpl in TEMPLATES:
            for combo in itertools.product(*(s[k] for k in tpl)):
                menus[tuple(zip(tpl, combo))].add(i)
    return [(h, sorted(ix)) for h, ix in sorted(menus.items()) if len(ix) >= min_items]


def holdout_menus(menus, frac: float = 0.2, seed: int = 0) -> list[tuple]:
    """Hold out multi-slot menus whose header values all still appear in some training menu."""
    rng = random.Random(seed)
    multi = [h for h, _ in menus if len(h) >= 2]
    rng.shuffle(multi)
    held: set[tuple] = set()
    for h in multi:
        if len(held) >= frac * len(menus):
            break
        rest = [g for g, _ in menus if g != h and g not in held]
        if all(any(sv in g for g in rest) for sv in h):
            held.add(h)
    return sorted(held)


def phrase(header: tuple) -> str:
    return " ".join(v for _, v in header)


class TextBank:
    """Rows 0..N-1 are products, then header values, then header phrases (free-text queries)."""

    def __init__(self, menus):
        self.items = np.load(OUT / f"embeddings_{BASE}.npy").astype(np.float32)
        slot_values = sorted({sv for h, _ in menus for sv in h})
        values = [v for _, v in slot_values]
        assert len(set(values)) == len(values), "a header value is shared across slots"
        phrases = sorted({phrase(h) for h, _ in menus})
        texts = values + phrases
        vecs = embed(BASE, texts).astype(np.float32)
        self.index = {t: N + j for j, t in enumerate(texts)}
        self.slot_of = {v: s for s, v in slot_values}
        self.slot_rows = {s: [self.index[v] for s2, v in slot_values if s2 == s] for s in SLOTS}
        self.vectors = torch.tensor(np.vstack([self.items, vecs]))

    def header_tokens(self, header, as_query: bool = False) -> list[tuple[int, int]]:
        if as_query:
            return [(self.index[phrase(header)], R["query"])]
        return [(self.index[v], R[slot]) for slot, v in header]


# ---------------------------------------------------------------- model

class Adapter(nn.Module):
    """Residual MLP on the unit sphere. Starts as the identity, so training starts from MPNet."""

    def __init__(self, dim: int = 768, hidden: int = 768, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.net(x), dim=-1)


class SetModel(nn.Module):
    def __init__(self, arch: str = "transformer", dim: int = 768, d: int = 256, layers: int = 2,
                 heads: int = 4, p: float = 0.1):
        super().__init__()
        self.arch = arch
        self.adapter = Adapter(dim)
        self.inp = nn.Linear(dim, d)
        self.role = nn.Embedding(len(ROLES), d)
        self.mask_vec = nn.Parameter(torch.zeros(d))
        if arch == "transformer":
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:  # deep sets: mean-pool phi(token), then rho
            self.phi = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            self.rho = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.out = nn.Linear(d, dim)

    def forward(self, adapted, idx, roles, pad):
        """adapted: adapter(bank). idx/roles/pad: B x L; slot 0 is the mask, its role says what to predict."""
        h = self.inp(adapted[idx]) + self.role(roles)
        is_mask = torch.zeros_like(pad)
        is_mask[:, 0] = True
        h = torch.where(is_mask[..., None], self.mask_vec + self.role(roles), h)
        if self.arch == "transformer":
            z = self.enc(h, src_key_padding_mask=pad)[:, 0]
        else:
            keep = (~pad & ~is_mask).float()[..., None]
            z = self.rho(h[:, 0] + (self.phi(h) * keep).sum(1) / keep.sum(1).clamp(min=1))
        return F.normalize(self.out(z), dim=-1)


def collate(batch_tokens: list[list[tuple[int, int]]], mask_roles: list[int]):
    L = 1 + max(len(t) for t in batch_tokens)
    idx = torch.zeros(len(batch_tokens), L, dtype=torch.long)
    roles = torch.zeros(len(batch_tokens), L, dtype=torch.long)
    pad = torch.ones(len(batch_tokens), L, dtype=torch.bool)
    for b, (toks, mr) in enumerate(zip(batch_tokens, mask_roles)):
        roles[b, 0], pad[b, 0] = mr, False
        for j, (i, r) in enumerate(toks, start=1):
            idx[b, j], roles[b, j], pad[b, j] = i, r, False
    return idx, roles, pad


def shared_values(slot: str, items) -> set[str]:
    """Values of a slot that every listed product has: all correct names for the list."""
    return set.intersection(*(SLOT_SETS[i][slot] for i in items))


# ---------------------------------------------------------------- training

def sample_item_task(rng, header, pool, bank):
    k = rng.randint(0, min(MAX_CTX, len(pool) - 1))
    ctx = rng.sample(pool, k)
    toks = bank.header_tokens(header, as_query=rng.random() < QUERY_P) + [(i, R["item"]) for i in ctx]
    return toks, R["mask_item"], range(N), set(pool) - set(ctx), set(ctx)


def sample_header_task(rng, header, pool, bank):
    j = rng.randrange(len(header))
    slot = header[j][0]
    kept = [sv for i, sv in enumerate(header) if i != j and rng.random() < 0.5]
    ctx = rng.sample(pool, rng.randint(1, min(MAX_CTX, len(pool))))
    toks = bank.header_tokens(kept) + [(i, R["item"]) for i in ctx]
    pos = {bank.index[v] for v in shared_values(slot, ctx) if v in bank.slot_of}
    return toks, R[f"mask_{slot}"], bank.slot_rows[slot], pos, set()


def train_set_model(arch, train_menus, bank, objectives=("item", "header"), steps=STEPS, batch=128,
                    seed=0, log=print):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SetModel(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    M = len(bank.vectors)
    t0 = time.time()
    for step in range(steps):
        toks, mroles = [], []
        cand, pos = torch.zeros(batch, M, dtype=torch.bool), torch.zeros(batch, M, dtype=torch.bool)
        task = torch.zeros(batch, dtype=torch.long)
        for b in range(batch):
            header, pool = rng.choice(train_menus)
            obj = rng.choice(objectives)
            sampler = sample_item_task if obj == "item" else sample_header_task
            t, mr, c, p, exclude = sampler(rng, header, pool, bank)
            toks.append(t)
            mroles.append(mr)
            cand[b, list(c)] = True
            cand[b, list(exclude)] = False
            pos[b, list(p)] = True
            task[b] = objectives.index(obj)
        adapted = model.adapter(bank.vectors)
        logits = model(adapted, *collate(toks, mroles)) @ adapted.T / TAU
        # equal weight per correct answer: each one against the wrong answers only
        lse_neg = torch.logsumexp(logits.masked_fill(~(cand & ~pos), -1e4), 1, keepdim=True)
        per_pos = F.softplus(lse_neg - logits) * pos
        per_example = per_pos.sum(1) / pos.sum(1).clamp(min=1)
        loss = sum(per_example[task == t].mean() for t in range(len(objectives)) if (task == t).any())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"  [{arch} {'+'.join(objectives)}] step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)")
    return model.eval()


def train_skipgram(train_menus, dim=64, sentences=40000, epochs=4, neg=5, seed=0):
    """Order-free SGNS baseline: every pair of tokens in a sampled menu is a (center, context) pair."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    vocab = {f"item:{i}": i for i in range(N)}
    for h, _ in train_menus:
        for slot, v in h:
            vocab.setdefault(f"{slot}:{v}", len(vocab))
    pairs = []
    for _ in range(sentences):
        h, pool = rng.choice(train_menus)
        sent = [vocab[f"{s}:{v}"] for s, v in h] + \
               [vocab[f"item:{i}"] for i in rng.sample(pool, min(len(pool), rng.randint(2, 8)))]
        pairs += [(a, b) for a in sent for b in sent if a != b]
    pairs = torch.tensor(pairs)
    freq = torch.bincount(pairs[:, 1], minlength=len(vocab)).float() ** 0.75
    w_in, w_out = nn.Embedding(len(vocab), dim), nn.Embedding(len(vocab), dim)
    nn.init.uniform_(w_in.weight, -0.5 / dim, 0.5 / dim)
    nn.init.zeros_(w_out.weight)
    opt = torch.optim.Adam(list(w_in.parameters()) + list(w_out.parameters()), lr=0.01)
    for _ in range(epochs):
        for chunk in pairs[torch.randperm(len(pairs))].split(8192):
            c, o = w_in(chunk[:, 0]), w_out(chunk[:, 1])
            negs = w_out(torch.multinomial(freq, len(chunk) * neg, replacement=True).view(len(chunk), neg))
            loss = -(F.logsigmoid((c * o).sum(-1)) + F.logsigmoid(-(negs @ c[..., None]).squeeze(-1)).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return vocab, F.normalize(w_in.weight.detach(), dim=-1)


# ---------------------------------------------------------------- scoring
# Each "space" scores two things: completion (header + items -> products) and naming
# (items -> header values of one slot). Centroid spaces average their input vectors.

class CentroidSpace:
    def __init__(self, vectors, bank):
        self.V, self.bank = F.normalize(vectors, dim=-1), bank

    def _rows(self, header, ctx, as_query):
        return [i for i, _ in self.bank.header_tokens(header, as_query)] + list(ctx)

    def complete(self, header, ctx, as_query=False):
        return self.V[:N] @ F.normalize(self.V[self._rows(header, ctx, as_query)].mean(0), dim=-1)

    def name(self, slot, ctx):
        rows = self.bank.slot_rows[slot]
        return rows, self.V[rows] @ F.normalize(self.V[list(ctx)].mean(0), dim=-1)


class SkipGramSpace:
    supports_query = False

    def __init__(self, vocab, W, bank):
        self.vocab, self.W, self.bank = vocab, W, bank

    def complete(self, header, ctx, as_query=False):
        rows = [self.vocab[f"{s}:{v}"] for s, v in header] + list(ctx)
        return self.W[:N] @ F.normalize(self.W[rows].mean(0), dim=-1)

    def name(self, slot, ctx):
        rows = self.bank.slot_rows[slot]
        sg = [self.vocab.get(f"{slot}:{self._value(r)}") for r in rows]
        scores = torch.full((len(rows),), -1e9)
        known = [j for j, s in enumerate(sg) if s is not None]
        scores[known] = self.W[[sg[j] for j in known]] @ F.normalize(self.W[list(ctx)].mean(0), dim=-1)
        return rows, scores

    def _value(self, row):
        return next(t for t, r in self.bank.index.items() if r == row)


class ModelSpace:
    def __init__(self, model, bank):
        self.m, self.bank = model, bank
        with torch.no_grad():
            self.A = model.adapter(bank.vectors)

    def _predict(self, toks, mask_role):
        with torch.no_grad():
            return self.m(self.A, *collate([toks], [mask_role]))[0]

    def complete(self, header, ctx, as_query=False):
        toks = self.bank.header_tokens(header, as_query) + [(i, R["item"]) for i in ctx]
        return self.A[:N] @ self._predict(toks, R["mask_item"])

    def name(self, slot, ctx):
        rows = self.bank.slot_rows[slot]
        return rows, self.A[rows] @ self._predict([(i, R["item"]) for i in ctx], R[f"mask_{slot}"])


# ---------------------------------------------------------------- evaluation

def average_precision(order, pos):
    hits, total = 0, 0.0
    for r, i in enumerate(order, start=1):
        if i in pos:
            hits += 1
            total += hits / r
    return total / max(1, len(pos))


def eval_completion(menus, spaces, seed=0):
    """Fill in each menu from its header alone, from header + 2 items, and from a free-text phrase."""
    rng = random.Random(seed)
    cases = [(h, ix, ctx) for h, ix in menus for ctx in ([], rng.sample(ix, 2))]
    out = {}
    for name, sp in spaces.items():
        res = defaultdict(list)
        for as_query in (False, True):
            if as_query and not getattr(sp, "supports_query", True):
                continue
            for h, ix, ctx in cases:
                if as_query and ctx:
                    continue
                tag = "query" if as_query else ("header" if not ctx else "header+2")
                s = sp.complete(h, ctx, as_query).clone()
                s[ctx] = -1e9
                order = torch.argsort(s, descending=True).tolist()[: N - len(ctx)]
                pos = set(ix) - set(ctx)
                res[f"{tag} P@5"].append(len(set(order[:5]) & pos) / 5)
                res[f"{tag} R@10"].append(len(set(order[:10]) & pos) / len(pos))
                res[f"{tag} mAP"].append(average_precision(order, pos))
        out[name] = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
    return out


def eval_naming(menus, spaces, seed=0, k=3):
    """Show k products from a menu, no header; name each slot of its header. Top-1 must be a value
    every shown product has."""
    rng = random.Random(seed)
    cases = [(slot, rng.sample(ix, k)) for h, ix in menus for slot, _ in h]
    out = {}
    for name, sp in spaces.items():
        by_slot = defaultdict(list)
        for slot, ctx in cases:
            rows, s = sp.name(slot, ctx)
            top = rows[int(torch.argmax(s))]
            ok = sp.bank.slot_of[next(t for t, r in sp.bank.index.items() if r == top)] == slot and \
                next(t for t, r in sp.bank.index.items() if r == top) in shared_values(slot, ctx)
            by_slot[slot].append(ok)
            by_slot["all"].append(ok)
        out[name] = {s: round(float(np.mean(v)), 3) for s, v in by_slot.items()}
    return out


def space_quality(spaces):
    """Leave-one-out 5-NN label agreement over all products, per facet. In-sample: the
    taxonomy was used in training, so this measures how much of it the geometry absorbed."""
    out = {}
    for name, X in spaces.items():
        X = F.normalize(X, dim=-1)
        sim = X @ X.T
        sim.fill_diagonal_(-2)
        nn5 = torch.topk(sim, 5, dim=1).indices.tolist()
        row = {}
        for slot in ("department", "type", "material", "audience", "occasion", "price"):
            row[slot] = round(float(np.mean([np.mean([bool(SLOT_SETS[i][slot] & SLOT_SETS[j][slot]) for j in nn5[i]])
                                             for i in range(N)])), 3)
        off = sim[~torch.eye(N, dtype=torch.bool)]
        row["cos mean"] = round(float(off.mean()), 3)
        out[name] = row
    return out


def free_text_examples(raw_E, model, k=5):
    rows = []
    with torch.no_grad():
        A = model.adapter(raw_E)
        q = torch.tensor(embed(BASE, FREE_TEXT), dtype=torch.float32)
        tuned = predict_query(model, q) @ A.T
        for t, r, a, s in zip(FREE_TEXT, q @ raw_E.T, model.adapter(q) @ A.T, tuned):
            rows.append({"query": t, **{name: [PRODUCTS[i].name for i in torch.topk(v, k).indices]
                                        for name, v in (("MPNet (raw)", r), ("Adapter only", a), ("Set transformer", s))}})
    return rows


def predict_query(model, q_raw):
    """Fill [MASK_ITEM] [QUERY: text]: where a menu headed by the query would put its products."""
    aq = model.adapter(q_raw)
    idx, roles, pad = collate([[(j, R["query"])] for j in range(len(q_raw))], [R["mask_item"]] * len(q_raw))
    return model(aq, idx, roles, pad)


# ---------------------------------------------------------------- saved model API (explorer server, CLI)

@lru_cache(maxsize=1)
def load() -> tuple[SetModel, TextBank]:
    ckpt = torch.load(WEIGHTS, weights_only=False)
    model = SetModel(ckpt["arch"])
    model.load_state_dict(ckpt["state"])
    return model.eval(), TextBank(ckpt["menus"])


def query_vector(text: str) -> np.ndarray:
    """A free-text query mapped into the tuned product space (what the explorer plots and ranks)."""
    model, _ = load()
    with torch.no_grad():
        return predict_query(model, torch.tensor(embed(BASE, [text]), dtype=torch.float32))[0].numpy()


def adapt(raw: np.ndarray) -> np.ndarray:
    """Raw MPNet vectors -> the tuned space (unit vectors)."""
    model, _ = load()
    with torch.no_grad():
        return model.adapter(torch.tensor(np.atleast_2d(raw), dtype=torch.float32)).numpy()


def session_scores(item_ids: list[int], query: str) -> np.ndarray:
    """Set completion for a shopping session: [MASK] [QUERY: search] [ITEM: clicked] ...
    Returns the cosine of every product to the predicted missing item."""
    model, bank = load()
    with torch.no_grad():
        A = model.adapter(bank.vectors)
        toks = [(int(i), R["item"]) for i in item_ids]
        if query:
            A = torch.cat([A, model.adapter(torch.tensor(embed(BASE, [query]), dtype=torch.float32))])
            toks.insert(0, (len(A) - 1, R["query"]))
        pred = model(A, *collate([toks], [R["mask_item"]]))[0]
        return (A[:N] @ pred).numpy()


def name_vector(raw: np.ndarray, k: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Name each header slot for one (possibly new) product given its raw MPNet vector."""
    model, bank = load()
    labels = {r: t for t, r in bank.index.items()}
    out = {}
    with torch.no_grad():
        A = torch.cat([model.adapter(bank.vectors), model.adapter(torch.tensor(raw[None], dtype=torch.float32))])
        j = len(A) - 1
        for slot in SLOTS:
            rows = bank.slot_rows[slot]
            pred = model(A, *collate([[(j, R["item"])]], [R[f"mask_{slot}"]]))[0]
            top = torch.topk(A[rows] @ pred, min(k, len(rows)))
            out[slot] = [(labels[rows[t]], round(float(v), 3)) for v, t in zip(top.values, top.indices)]
    return out


def _ctx(skus):
    return [PRODUCTS.index(PRODUCTS_BY_SKU[s]) for s in skus]


def complete(header_values: list[str], skus: list[str], k: int = 10):
    model, bank = load()
    unknown = [v for v in header_values if v not in bank.slot_of]
    if unknown:
        raise SystemExit(f"Unknown header value(s) {unknown}. Use names printed by quince_taxonomy.py.")
    sp, ctx = ModelSpace(model, bank), _ctx(skus)
    s = sp.complete(tuple((bank.slot_of[v], v) for v in header_values), ctx)
    s[ctx] = -1e9
    return [(f"{PRODUCTS[i].sku}  {PRODUCTS[i].name}", float(s[i])) for i in torch.topk(s, k).indices]


def name_list(skus: list[str]):
    model, bank = load()
    sp, ctx = ModelSpace(model, bank), _ctx(skus)
    out = []
    for slot in SLOTS:
        rows, s = sp.name(slot, ctx)
        best = torch.topk(s, min(3, len(rows)))
        labels = {r: t for t, r in bank.index.items()}
        out.append((slot, [(labels[rows[j]], float(v)) for v, j in zip(best.values, best.indices)]))
    return out


# ---------------------------------------------------------------- report

def table(title, rows, cols=None, fmt="{:.3f}"):
    cols = cols or list(next(iter(rows.values())))
    lines = ([f"### {title}", ""] if title else []) + ["| | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(fmt.format(r[c]) if c in r else "–" for c in cols) + " |")
    return lines + [""]


def write_report(res, n_train, held):
    L = ["# Masked set model: results", "",
         f"{n_train + len(held)} menus; {len(held)} multi-slot menus held out of training (never seen), "
         f"{n_train} trained on. Every product is in training. Held-out menus test composition: each of their "
         "header values was learned on other menus.", "",
         "Held out: " + "; ".join(" · ".join(v for _, v in h) for h in held), "",
         "## 1. Filling in held-out menus", "",
         "*P@5*: share of the top 5 on the menu. *R@10*: share of the menu in the top 10. *mAP*: mean average precision.", ""]
    comp = res["completion_held"]
    for tag, title in (("header", "From the header alone"), ("header+2", "From the header plus 2 listed products"),
                       ("query", "From the header written as a free-text phrase")):
        rows = {k: {c.split(" ", 1)[1]: v for c, v in r.items() if c.startswith(tag + " ")} for k, r in comp.items()}
        L += table(title, {k: v for k, v in rows.items() if v})
    L += ["Same metric on menus seen in training, header alone (the memorization ceiling):", ""]
    L += table("", {k: {c.split(" ", 1)[1]: v for c, v in r.items() if c.startswith("header ")}
                    for k, r in res["completion_train"].items()})
    L += ["## 2. Naming a list", "", "Show 3 products from a held-out menu, no header, and name each slot of its "
          "header. Correct if the top answer is a value all 3 products have.", ""]
    L += table("", res["naming"])
    L += ["## 3. The product space itself", "", "Leave-one-out 5-NN: share of each product's 5 nearest neighbors "
          "sharing its value. In-sample (the taxonomy shaped training), so read it as how much structure the "
          "geometry absorbed; price is not in the text.", ""]
    L += table("", res["space"])
    L += ["## 4. Free-text queries (top 5)", ""]
    for ex in res["free_text"]:
        L += [f"**{ex['query']}**", ""] + [f"- {k}: " + "; ".join(ex[k]) for k in
                                           ("MPNet (raw)", "Adapter only", "Set transformer")] + [""]
    (OUT / "set_model_report.md").write_text("\n".join(L))


REPORT_START, REPORT_END = "<!-- set-model:start -->", "<!-- set-model:end -->"


def project_report_section(res) -> str:
    """Summary of the set-model experiment for the project report (outputs/report.md)."""
    comp, name, space = res["completion_held"], res["naming"], res["space"]
    held = res.get("held", [])
    L = [REPORT_START, "## Masked set model: learning the taxonomy into the space", "",
         "Each product list (\"menu\") is a header plus every product it lists, e.g. `Jewelry · Rings · Gold`. "
         "All tokens are frozen MPNet vectors passed through a small shared adapter; role embeddings stand in "
         "for positions, so list order does not matter. Two equally weighted objectives: hide a product and "
         "predict it, and hide a header slot and name it from the listed products. Full softmax (InfoNCE), "
         "each correct answer weighted equally.", "",
         f"Every product is trained on. {len(held)} multi-slot menus are held out entirely; each of their header "
         "values was learned on other menus, so filling them in tests whether learned concepts compose.", "",
         "### Filling in held-out menus from the header alone", "",
         "| | mAP | R@10 | P@5 |", "|---|---:|---:|---:|"]
    L += [f"| {k} | {r['header mAP']:.3f} | {r['header R@10']:.3f} | {r['header P@5']:.3f} |" for k, r in comp.items()]
    L += ["", "### Naming a list (3 products, no header)", "",
          "Top answer must be a value all 3 products share.", "", "| | all slots | department | type | material | price |",
          "|---|---:|---:|---:|---:|---:|"]
    L += [f"| {k} | {r['all']:.3f} | {r['department']:.3f} | {r['type']:.3f} | {r['material']:.3f} | {r['price']:.3f} |"
          for k, r in name.items()]
    L += ["", "### The product space itself (5-NN label agreement, in-sample)", "",
          "| | department | type | occasion | price |", "|---|---:|---:|---:|---:|"]
    L += [f"| {k} | {r['department']:.3f} | {r['type']:.3f} | {r['occasion']:.3f} | {r['price']:.3f} |"
          for k, r in space.items()]
    L += ["", "Price is not in the product text, so any price structure in a space came from the menus.", "",
          "Taxonomy: `quince_taxonomy.py`. Model and evaluation: `quince_set_model.py`. "
          "Full tables and free-text query examples: [set_model_report.md](set_model_report.md).", REPORT_END, ""]
    return "\n".join(L)


def update_project_report(res=None) -> None:
    """Insert or replace the set-model section in outputs/report.md."""
    res = res or json.loads((OUT / "set_model_results.json").read_text())
    path = OUT / "report.md"
    text = path.read_text() if path.exists() else ""
    if REPORT_START in text:
        text = text[: text.index(REPORT_START)] + text[text.index(REPORT_END) + len(REPORT_END):].lstrip("\n")
    path.write_text(text.rstrip("\n") + "\n\n" + project_report_section(res))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--complete", nargs="*", metavar="HEADER", help="header values, e.g. Jewelry Earrings Gold")
    parser.add_argument("--items", nargs="*", default=[], metavar="SKU", help="visible products for --complete")
    parser.add_argument("--name-list", nargs="+", metavar="SKU", help="name the header for a list of products")
    parser.add_argument("--query", help="free-text query to complete")
    args = parser.parse_args()

    if args.complete is not None:
        for label, s in complete(args.complete, args.items):
            print(f"{s:6.3f}  {label}")
        return
    if args.name_list:
        for slot, top in name_list(args.name_list):
            print(f"{slot:11s} " + "   ".join(f"{v} ({s:.2f})" for v, s in top))
        return
    if args.query:
        model, bank = load()
        s = model.adapter(bank.vectors[:N]).detach() @ torch.tensor(query_vector(args.query))
        for i in torch.topk(s, 10).indices:
            print(f"{float(s[i]):6.3f}  {PRODUCTS[i].sku}  {PRODUCTS[i].name}")
        return

    menus = build_menus()
    held = holdout_menus(menus)
    held_set = set(held)
    train = [(h, ix) for h, ix in menus if h not in held_set]
    test = [(h, ix) for h, ix in menus if h in held_set]
    print(f"{len(menus)} menus: {len(train)} train, {len(test)} held out")
    bank = TextBank(menus)

    models = {
        "Set transformer, item only": train_set_model("transformer", train, bank, ("item",), args.steps),
        "Set transformer, item + header": train_set_model("transformer", train, bank, ("item", "header"), args.steps),
        "Deep Sets, item + header": train_set_model("deepsets", train, bank, ("item", "header"), args.steps),
    }
    print("  [skip-gram] training")
    vocab, W = train_skipgram(train)
    best = models["Set transformer, item + header"]
    spaces = {
        "MPNet (raw)": CentroidSpace(bank.vectors, bank),
        "Skip-gram tokens": SkipGramSpace(vocab, W, bank),
        "Adapter only (item only)": CentroidSpace(models["Set transformer, item only"].adapter(bank.vectors).detach(), bank),
        "Adapter only (item + header)": CentroidSpace(best.adapter(bank.vectors).detach(), bank),
        **{k: ModelSpace(m, bank) for k, m in models.items()},
    }
    E = bank.vectors[:N]
    with torch.no_grad():
        res = {
            "completion_held": eval_completion(test, spaces),
            "completion_train": eval_completion(train, {k: spaces[k] for k in
                                                        ("MPNet (raw)", "Set transformer, item + header")}),
            "naming": eval_naming(test, {k: v for k, v in spaces.items() if k != "Set transformer, item only"}),
            "space": space_quality({"MPNet (raw)": E, "Skip-gram item vectors": W[:N],
                                    "Adapter (item only)": models["Set transformer, item only"].adapter(E),
                                    "Adapter (item + header)": best.adapter(E),
                                    "Adapter (Deep Sets, item + header)": models["Deep Sets, item + header"].adapter(E)}),
            "free_text": free_text_examples(E, best),
        }

    torch.save({"arch": "transformer", "state": best.state_dict(), "menus": menus, "held": held}, WEIGHTS)
    with torch.no_grad():
        np.save(OUT / f"embeddings_{KEY}.npy", best.adapter(E).numpy())
    (OUT / "set_model_results.json").write_text(json.dumps({**res, "held": [phrase(h) for h in held]}, indent=2))
    write_report(res, len(train), held)
    update_project_report({**res, "held": [phrase(h) for h in held]})
    print((OUT / "set_model_report.md").read_text())


if __name__ == "__main__":
    main()
