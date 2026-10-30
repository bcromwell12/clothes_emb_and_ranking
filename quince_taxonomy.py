"""
A three-level merchandising taxonomy for the mock Quince catalog, plus cross-cutting facets.

    department  ->  type  ->  facets (material, audience, occasion, price band)

Department is the existing category. Type cleans up the catalog's subcategories
(gendered or singleton subcategories fold into a shared type; gender moves to the
audience facet). Facets cut across the tree: cashmere shows up in sweaters and
accessories, linen in clothing and bedding, leather in bags, shoes and furniture.

Materials, audience and most occasions are read from the product text by keyword
rules, so a text encoder can in principle recover them. The price band and the
"gift" occasion depend on price, which is not in the description, so they test
what the encoder cannot see.

Usage:
    .venv/bin/python quince_taxonomy.py          # print the tree and facet counts
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from quince_catalog import CATEGORIES, PRODUCTS, Product

TYPE_MAP = {
    "Women's Sweaters": "Sweaters", "Men's Sweaters": "Sweaters", "Lounge": "Loungewear",
    "Accessories": "Cold-Weather Accessories",
    "Tops": "Tops", "Men's Tops": "Tops", "Bottoms": "Bottoms", "Men's Bottoms": "Bottoms",
    "Dresses": "Dresses & Skirts", "Outerwear": "Outerwear", "Shoes": "Shoes",
    "Sleepwear": "Loungewear", "Activewear": "Activewear & Swim", "Swim": "Activewear & Swim",
    "Totes": "Totes", "Crossbody": "Crossbody & Belt Bags", "Shoulder Bags": "Shoulder Bags",
    "Clutches": "Shoulder Bags", "Backpacks": "Work & Travel", "Work": "Work & Travel",
    "Travel": "Work & Travel", "Small Leather Goods": "Small Leather Goods",
    "Bedding": "Bedding", "Pillows": "Bedding", "Throws & Blankets": "Bedding", "Bath": "Bath",
    "Furniture": "Furniture", "Rugs": "Decor", "Lighting": "Decor",
    "Earrings": "Earrings", "Necklaces": "Necklaces", "Men's Jewelry": "Necklaces",
    "Bracelets": "Bracelets", "Rings": "Rings",
}

# first match wins per product, but a product can carry several materials
MATERIAL_RULES = [
    ("Cashmere", ["cashmere"]),
    ("Silk", ["silk"]),
    ("Linen", ["linen"]),
    ("Leather", ["leather", "lambskin", "suede", "nappa"]),
    ("Organic Cotton", ["organic cotton", "organic pima"]),
    ("Cotton", ["cotton", "pima"]),
    ("Wool", ["wool", "merino"]),
    ("Down", ["white down", "fill down", "responsible down"]),
    ("Recycled Nylon", ["recycled nylon", "regenerated nylon"]),
    ("Gold", ["14k", "18k gold", "gold vermeil"]),
    ("Sterling Silver", ["sterling silver", "925"]),
    ("Diamond", ["diamond"]),
    ("Pearl", ["pearl"]),
    ("Jade", ["jade"]),
    ("Ruby", ["ruby"]),
    ("Onyx", ["onyx"]),
    ("Turquoise", ["turquoise"]),
    ("Stainless Steel", ["stainless steel"]),
    ("Wood", ["oak", "acacia", "hardwood", "wood frame"]),
    ("Natural Fiber", ["jute", "raffia", "straw", "bamboo", "tencel", "lyocell"]),
    ("Stone & Ceramic", ["marble", "ceramic"]),
    ("Performance Blend", ["rayon", "ponte"]),
]

OCCASION_RULES = [
    ("Work", ["office", "office-ready", "workwear", "laptop", "briefcase", "sport coat"]),
    ("Travel", ["travel", "carry-on", "packable", "weekender", "toiletry", "packing"]),
    ("Lounge & Sleep", ["lounge", "pajama", "jogger", "sleep", "hoodie", "relax", "cozy"]),
    ("Night Out", ["evening", "nights out", "night out", "clutch", "cocktail", "dressed up", "dress up"]),
    ("Summer", ["summer", "beach", "breezy", "swim", "cooling", "hot sleepers"]),
    ("Cold Weather", ["winter", "warmth", "warm", "puffer", "coat", "beanie", "gloves"]),
    ("Everyday", ["everyday", "staple", "all-day", "errands", "daily"]),
    ("Wedding & Milestone", ["wedding", "heirloom", "engraving", "personalized", "eternity"]),
]

AUDIENCE_RULES = [("Women", ["women's", "dress", "skirt", "blouse", "legging",
                                                 "swimsuit", "clutch", "hobo", "bucket bag", "high-rise"]), ("Men", ["men's"])]


@dataclass(frozen=True)
class Taxon:
    sku: str
    department: str
    type: str
    materials: tuple[str, ...]
    audience: str
    occasions: tuple[str, ...]
    price_band: str

    def facets(self) -> dict[str, tuple[str, ...]]:
        return {"material": self.materials, "audience": (self.audience,),
                "occasion": self.occasions, "price": (self.price_band,)}


def _matches(text: str, rules) -> list[str]:
    return [label for label, keys in rules
            if any(re.search(rf"(?<![\w']){re.escape(k)}(?!\w)", text) for k in keys)]


def _price_band(p: Product) -> str:
    if p.price < 50:
        return "Under $50"
    if p.price < 150:
        return "$50 to $150"
    if p.price < 500:
        return "$150 to $500"
    return "$500 and Up"


def classify(p: Product) -> Taxon:
    text = f"{p.name} {p.subcategory} {p.materials} {p.description}".lower()
    materials = _matches(f"{p.name} {p.materials}".lower(), MATERIAL_RULES)
    if "Organic Cotton" in materials and "Cotton" in materials:
        materials.remove("Cotton")
    if "Gold" in materials and "Sterling Silver" in materials and "vermeil" in text:
        materials.remove("Sterling Silver")  # vermeil reads as gold
    if p.category == "furniture_bed":
        audience = "Home"
    else:
        aud = _matches(text, AUDIENCE_RULES)
        audience = aud[0] if aud else ("Women" if p.category == "jewelry" else "Unisex")
    occasions = _matches(text, OCCASION_RULES)
    # price-driven occasion: not readable from the description
    if p.category in ("jewelry", "cashmere_sweaters", "bags") and p.price <= 100:
        occasions.append("Gift")
    return Taxon(p.sku, CATEGORIES[p.category]["label"], TYPE_MAP[p.subcategory],
                 tuple(materials) or ("Other",), audience, tuple(dict.fromkeys(occasions)) or ("Everyday",),
                 _price_band(p))


TAXONOMY: list[Taxon] = [classify(p) for p in PRODUCTS]
TAXONOMY_BY_SKU = {t.sku: t for t in TAXONOMY}
FACET_KINDS = ("material", "audience", "occasion", "price")


def tree() -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in TAXONOMY:
        out[t.department][t.type].append(t.sku)
    return out


def facet_index() -> dict[str, dict[str, list[int]]]:
    """facet kind -> value -> product indices."""
    idx: dict[str, dict[str, list[int]]] = {k: defaultdict(list) for k in FACET_KINDS}
    for i, t in enumerate(TAXONOMY):
        for kind, values in t.facets().items():
            for v in values:
                idx[kind][v].append(i)
    return idx


if __name__ == "__main__":
    for dept, types in tree().items():
        print(dept)
        for typ, skus in types.items():
            print(f"  {typ:26s} {len(skus):2d}  {' '.join(skus)}")
    for kind, values in facet_index().items():
        print(f"\n{kind}: " + ", ".join(f"{v} ({len(ix)})" for v, ix in sorted(values.items(), key=lambda kv: -len(kv[1]))))
