"""
Mock Quince product catalog: 106 products across 5 categories (20 each, jewelry 26).

Each product is modeled like a backend/e-commerce record (SKU, slug, pricing,
inventory, variants, materials, care, SEO, etc.). Prices are illustrative and
modeled on Quince's general price points, not pulled from their live site.

Usage:
    from quince_catalog import PRODUCTS, get_by_category, to_json
    python quince_catalog.py          # prints a summary
    python quince_catalog.py --json   # dumps all products as JSON
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Product:
    id: int
    sku: str
    slug: str
    name: str
    category: str
    subcategory: str
    description: str
    price: float
    compare_at_price: float          # "traditional retail" price Quince compares against
    unit_cost: float                 # landed cost per unit (made up)
    currency: str
    materials: str
    colors: list[str]
    sizes: list[str]
    care_instructions: str
    inventory: dict[str, int]        # stock per size
    tags: list[str]
    rating: float
    review_count: int
    status: str                      # "active" | "draft" | "archived"
    is_featured: bool
    created_at: str
    seo_title: str
    seo_description: str
    image_urls: list[str] = field(default_factory=list)

    @property
    def total_inventory(self) -> int:
        return sum(self.inventory.values())

    @property
    def discount_pct(self) -> int:
        return round((1 - self.price / self.compare_at_price) * 100)

    @property
    def margin_pct(self) -> float:
        """Gross margin as a fraction of price."""
        return round(1 - self.unit_cost / self.price, 3)

    @property
    def margin_dollars(self) -> float:
        """Gross margin per unit sold."""
        return round(self.price - self.unit_cost, 2)

    @property
    def in_stock(self) -> bool:
        return self.total_inventory > 0


# ---------------------------------------------------------------------------
# Category config
# ---------------------------------------------------------------------------

CATEGORIES = {
    "cashmere_sweaters": {"label": "Cashmere Sweaters", "sku_prefix": "CSH"},
    "clothing":          {"label": "General Clothing",  "sku_prefix": "CLO"},
    "bags":              {"label": "Bags",              "sku_prefix": "BAG"},
    "furniture_bed":     {"label": "Furniture & Bed",   "sku_prefix": "HOM"},
    "jewelry":           {"label": "Jewelry",           "sku_prefix": "JWL"},
}

APPAREL_SIZES = ["XS", "S", "M", "L", "XL"]
MENS_SIZES = ["S", "M", "L", "XL", "XXL"]
PANT_SIZES = ["0", "2", "4", "6", "8", "10", "12", "14"]
SHOE_SIZES = ["6", "7", "8", "9", "10", "11"]
ONE_SIZE = ["One Size"]
BED_SIZES = ["Twin", "Full", "Queen", "King", "Cal King"]
DUVET_SIZES = ["Full/Queen", "King/Cal King"]
RUG_SIZES = ["5x8", "8x10", "9x12"]
RING_SIZES = ["5", "6", "7", "8", "9"]
CHAIN_LENGTHS = ['16"', '18"', '20"']

# Made-up gross margin ranges (fraction of price) by category, nudged per item below.
# Low-markup DTC: thinner than traditional retail, with home furniture thinnest
# (freight-heavy) and lab-grown jewelry fattest.
MARGIN_RANGES = {
    "cashmere_sweaters": (0.42, 0.58),
    "clothing":          (0.40, 0.62),
    "bags":              (0.48, 0.66),
    "furniture_bed":     (0.38, 0.58),
    "jewelry":           (0.35, 0.60),
}


def _mock_margin(category: str, subcategory: str, name: str, rng: random.Random) -> float:
    lo, hi = MARGIN_RANGES[category]
    m = rng.uniform(lo, hi)
    if subcategory == "Furniture":
        m -= 0.12               # big, bulky, expensive to ship
    if subcategory in ("Accessories", "Small Leather Goods"):
        m += 0.06               # small, cheap to make relative to price
    if "Lab Grown" in name:
        m += 0.15               # lab stones carry a fat margin
    elif "14K Gold" in name:
        m -= 0.08               # gold content is most of the cost
    return min(max(m, 0.18), 0.78)


CARE = {
    "cashmere_sweaters": "Hand wash cold with cashmere shampoo or dry clean. Lay flat to dry. Store folded.",
    "clothing": "Machine wash cold on gentle, inside out. Tumble dry low or hang dry. See garment label.",
    "bags": "Wipe clean with a soft, dry cloth. Condition leather periodically. Store in included dust bag.",
    "furniture_bed": "See product label. Textiles: machine wash cold, tumble dry low. Furniture: dust with a dry cloth.",
    "jewelry": "Avoid contact with water, perfume, and lotions. Polish with a soft cloth. Store in included pouch.",
}


# ---------------------------------------------------------------------------
# Raw product rows
# (name, subcategory, price, compare_at, materials, colors, sizes, description)
# ---------------------------------------------------------------------------

RAW_PRODUCTS: dict[str, list[tuple]] = {
    "cashmere_sweaters": [
        ("Mongolian Cashmere Crewneck Sweater", "Women's Sweaters", 50, 140,
         "100% Grade-A Mongolian cashmere, 12gg", ["Oatmeal", "Black", "Heather Grey", "Navy", "Ivory", "Camel"], APPAREL_SIZES,
         "Our bestselling crewneck in Grade-A Mongolian cashmere. Ultra-soft, lightweight, and warm with a relaxed fit that layers effortlessly. Ribbed cuffs, hem, and neckline."),
        ("Mongolian Cashmere V-Neck Sweater", "Women's Sweaters", 50, 140,
         "100% Grade-A Mongolian cashmere, 12gg", ["Black", "Camel", "Light Grey", "Blush", "Ivory"], APPAREL_SIZES,
         "A wardrobe staple with a flattering V-neckline. Knit from long-fiber Mongolian cashmere for softness that only improves with wear."),
        ("Mongolian Cashmere Turtleneck Sweater", "Women's Sweaters", 60, 160,
         "100% Grade-A Mongolian cashmere, 12gg", ["Ivory", "Black", "Oatmeal", "Chocolate", "Forest Green"], APPAREL_SIZES,
         "A fold-over turtleneck with a slim, polished fit. Warm enough to wear alone and thin enough to layer under a blazer or coat."),
        ("Mongolian Cashmere Cardigan Sweater", "Women's Sweaters", 70, 175,
         "100% Grade-A Mongolian cashmere, 12gg", ["Heather Grey", "Black", "Camel", "Navy"], APPAREL_SIZES,
         "A classic button-front cardigan with horn-effect buttons and a relaxed silhouette. The easiest layer for office, travel, and weekends."),
        ("Mongolian Cashmere Oversized Boyfriend Sweater", "Women's Sweaters", 80, 200,
         "100% Grade-A Mongolian cashmere, 7gg", ["Oatmeal", "Charcoal", "Ivory", "Camel"], APPAREL_SIZES,
         "A slouchy, oversized crewneck in a chunkier 7-gauge knit. Dropped shoulders and a longer body for cozy, lived-in style."),
        ("Mongolian Cashmere Cropped Polo Sweater", "Women's Sweaters", 65, 165,
         "100% Grade-A Mongolian cashmere, 12gg", ["Navy", "Ivory", "Black", "Cherry"], APPAREL_SIZES,
         "A cropped polo with a soft collar and three-button placket. Pairs perfectly with high-rise trousers and skirts."),
        ("Mongolian Cashmere Fisherman Cable Knit Sweater", "Women's Sweaters", 100, 250,
         "100% Grade-A Mongolian cashmere, 5gg", ["Ivory", "Oatmeal", "Heather Grey"], APPAREL_SIZES,
         "A heritage-inspired cable knit made luxurious in heavyweight Mongolian cashmere. Substantial, warm, and built to last for seasons."),
        ("Mongolian Cashmere Mockneck Sweater", "Women's Sweaters", 60, 150,
         "100% Grade-A Mongolian cashmere, 12gg", ["Black", "Camel", "Light Grey", "Burgundy"], APPAREL_SIZES,
         "A modern mockneck with a slightly cropped, boxy fit. Soft, breathable, and polished enough for any occasion."),
        ("Mongolian Cashmere Sweater Vest", "Women's Sweaters", 45, 120,
         "100% Grade-A Mongolian cashmere, 12gg", ["Oatmeal", "Navy", "Black"], APPAREL_SIZES,
         "A V-neck sweater vest that adds warmth without bulk. Layer over button-downs or wear alone with a tailored trouser."),
        ("Mongolian Cashmere Hoodie", "Women's Sweaters", 90, 225,
         "100% Grade-A Mongolian cashmere, 12gg", ["Heather Grey", "Black", "Oatmeal"], APPAREL_SIZES,
         "The elevated hoodie. Drawstring hood, kangaroo pocket, and ribbed trims in pure Mongolian cashmere for luxe lounging."),
        ("Mongolian Cashmere Wrap", "Accessories", 70, 180,
         "100% Grade-A Mongolian cashmere, woven", ["Camel", "Grey", "Black", "Ivory", "Plaid"], ONE_SIZE,
         "An oversized woven wrap that doubles as a travel blanket. Finished with fringed edges and impossibly soft hand-feel."),
        ("Mongolian Cashmere Ribbed Beanie", "Accessories", 30, 85,
         "100% Grade-A Mongolian cashmere", ["Oatmeal", "Black", "Grey", "Camel", "Red"], ONE_SIZE,
         "A ribbed beanie with a fold-over cuff. Warm, lightweight, and itch-free for all-day winter wear."),
        ("Men's Mongolian Cashmere Crewneck Sweater", "Men's Sweaters", 60, 150,
         "100% Grade-A Mongolian cashmere, 12gg", ["Navy", "Charcoal", "Camel", "Black", "Olive"], MENS_SIZES,
         "A timeless crewneck with a tailored fit through the body and sleeves. Layers cleanly over a collared shirt or tee."),
        ("Men's Mongolian Cashmere V-Neck Sweater", "Men's Sweaters", 60, 150,
         "100% Grade-A Mongolian cashmere, 12gg", ["Charcoal", "Navy", "Black", "Oatmeal"], MENS_SIZES,
         "A refined V-neck for workwear and beyond. Lightweight, breathable warmth in fully-fashioned cashmere."),
        ("Men's Mongolian Cashmere Quarter Zip Sweater", "Men's Sweaters", 80, 200,
         "100% Grade-A Mongolian cashmere, 12gg", ["Navy", "Heather Grey", "Black", "Brown"], MENS_SIZES,
         "A quarter-zip pullover with a metal zipper and stand collar. A sharper alternative to the fleece pullover."),
        ("Men's Mongolian Cashmere Shawl Collar Cardigan", "Men's Sweaters", 110, 280,
         "100% Grade-A Mongolian cashmere, 7gg", ["Oatmeal", "Charcoal", "Navy"], MENS_SIZES,
         "A heavyweight shawl-collar cardigan with patch pockets and horn buttons. The ultimate weekend layer."),
        ("Men's Mongolian Cashmere Hoodie", "Men's Sweaters", 100, 250,
         "100% Grade-A Mongolian cashmere, 12gg", ["Heather Grey", "Navy", "Black"], MENS_SIZES,
         "A relaxed pullover hoodie in pure cashmere. Soft drawstrings and a kangaroo pocket for elevated off-duty style."),
        ("Men's Mongolian Cashmere Turtleneck Sweater", "Men's Sweaters", 70, 175,
         "100% Grade-A Mongolian cashmere, 12gg", ["Black", "Charcoal", "Camel"], MENS_SIZES,
         "A slim turtleneck that works under a sport coat or on its own. Warm, smooth, and wrinkle-resistant."),
        ("Mongolian Cashmere Jogger Pants", "Lounge", 80, 200,
         "100% Grade-A Mongolian cashmere, 12gg", ["Heather Grey", "Black", "Oatmeal"], APPAREL_SIZES,
         "Cashmere joggers with an elastic drawstring waist, side pockets, and ribbed cuffs. Pair with the matching hoodie."),
        ("Mongolian Cashmere Touchscreen Gloves", "Accessories", 35, 90,
         "100% Grade-A Mongolian cashmere, conductive fingertips", ["Black", "Grey", "Camel"], ["S/M", "M/L"],
         "Ribbed cashmere gloves with touchscreen-compatible fingertips, so you stay warm without taking them off."),
    ],

    "clothing": [
        ("Organic Cotton Pima Crewneck T-Shirt", "Tops", 15, 40,
         "100% organic Pima cotton jersey", ["White", "Black", "Heather Grey", "Navy", "Sage"], APPAREL_SIZES,
         "An everyday tee in GOTS-certified organic Pima cotton. Long-staple fibers make it extra soft, durable, and resistant to pilling."),
        ("Washable Stretch Silk Button-Down Shirt", "Tops", 60, 150,
         "93% mulberry silk, 7% spandex", ["Ivory", "Black", "Champagne", "Navy", "Sage"], APPAREL_SIZES,
         "A 22-momme silk blouse with a touch of stretch. Machine washable, so luxury doesn't come with a dry cleaning bill."),
        ("100% European Linen Wide Leg Pants", "Bottoms", 40, 110,
         "100% European flax linen", ["Natural", "White", "Black", "Olive"], APPAREL_SIZES,
         "Breezy wide-leg pants with an elastic back waistband and side pockets. Garment-washed linen softens with every wear."),
        ("Men's 100% European Linen Long Sleeve Shirt", "Men's Tops", 40, 100,
         "100% European flax linen", ["White", "Light Blue", "Sand", "Navy", "Olive"], MENS_SIZES,
         "A relaxed button-up in pre-washed European linen. Breathable and naturally temperature-regulating for warm days."),
        ("Comfort Stretch High Rise Straight Jean", "Bottoms", 50, 130,
         "92% cotton, 6% polyester, 2% elastane", ["Light Wash", "Medium Wash", "Dark Rinse", "Black"], PANT_SIZES,
         "A high-rise straight jean with just enough stretch for all-day comfort. Holds its shape without bagging out."),
        ("Organic Cotton French Terry Sweatshirt", "Tops", 30, 80,
         "100% organic cotton French terry", ["Heather Grey", "Navy", "Oatmeal", "Black"], APPAREL_SIZES,
         "A midweight crewneck sweatshirt in loopback French terry. Brushed for softness with ribbed cuffs and hem."),
        ("Washable Silk Bias Cut Slip Skirt", "Bottoms", 50, 130,
         "100% mulberry silk charmeuse", ["Champagne", "Black", "Emerald", "Navy"], APPAREL_SIZES,
         "A fluid bias-cut midi skirt in lustrous silk charmeuse. Elastic waist for easy wear; dress up or down."),
        ("Ultra-Stretch Ponte Straight Leg Pants", "Bottoms", 40, 110,
         "68% rayon, 27% nylon, 5% spandex", ["Black", "Navy", "Charcoal", "Camel"], PANT_SIZES,
         "Pull-on pants that look tailored and feel like leggings. Wrinkle-resistant ponte with a smooth, flat-front waist."),
        ("Tencel Jersey Midi Dress", "Dresses", 40, 100,
         "95% Tencel lyocell, 5% spandex", ["Black", "Navy", "Olive", "Terracotta"], APPAREL_SIZES,
         "A drapey jersey dress with a scoop neck and midi length. Sustainably sourced Tencel keeps you cool and comfortable."),
        ("Italian Wool Double-Breasted Coat", "Outerwear", 200, 500,
         "80% Italian wool, 20% polyamide", ["Camel", "Black", "Charcoal"], APPAREL_SIZES,
         "A tailored double-breasted coat cut from Italian-milled wool. Fully lined with welt pockets and a notch lapel."),
        ("100% Leather Moto Jacket", "Outerwear", 250, 600,
         "100% lambskin leather, polyester lining", ["Black", "Cognac"], APPAREL_SIZES,
         "A buttery lambskin moto jacket with asymmetric zip, snap lapels, and zip pockets. An heirloom piece at a fraction of the price."),
        ("Responsible Down Puffer Jacket", "Outerwear", 120, 300,
         "Recycled nylon shell, RDS-certified 700-fill down", ["Black", "Navy", "Olive", "Ivory"], APPAREL_SIZES,
         "A lightweight, packable puffer filled with responsibly sourced 700-fill down. Water-resistant shell for winter days."),
        ("Australian Merino Wool Crewneck Sweater", "Tops", 40, 100,
         "100% Australian merino wool, 14gg", ["Navy", "Black", "Grey", "Burgundy"], APPAREL_SIZES,
         "A fine-gauge merino crewneck that's naturally breathable and odor-resistant. Travels well and resists wrinkles."),
        ("Men's Performance Stretch Chino Pants", "Men's Bottoms", 45, 110,
         "97% cotton, 3% elastane", ["Khaki", "Navy", "Olive", "Stone", "Black"], ["30", "32", "34", "36", "38"],
         "A slim-straight chino with four-way stretch and a soft brushed finish. Office-ready with weekend-level comfort."),
        ("Men's Organic Cotton Oxford Shirt", "Men's Tops", 35, 90,
         "100% organic cotton Oxford cloth", ["White", "Light Blue", "Pink", "Blue Stripe"], MENS_SIZES,
         "A button-down collar Oxford shirt made from GOTS-certified organic cotton. Gets better with every wash."),
        ("Washable Silk Pajama Set", "Sleepwear", 100, 250,
         "100% mulberry silk, 22 momme", ["Champagne", "Navy", "Blush", "Black"], APPAREL_SIZES,
         "A classic notch-collar silk pajama set with piped trim. Temperature-regulating and machine washable."),
        ("Flowknit High Rise Performance Legging", "Activewear", 35, 90,
         "75% recycled nylon, 25% spandex", ["Black", "Navy", "Olive", "Plum"], APPAREL_SIZES,
         "Buttery-soft, squat-proof leggings with a high-rise waistband and hidden pocket. Moisture-wicking for studio or street."),
        ("Recycled Nylon Square Neck One-Piece Swimsuit", "Swim", 35, 90,
         "78% recycled nylon, 22% spandex", ["Black", "Navy", "Red", "Sage"], APPAREL_SIZES,
         "A supportive square-neck one-piece with a fully lined body. Made from regenerated nylon and chlorine-resistant."),
        ("Italian Leather Chelsea Boot", "Shoes", 130, 320,
         "100% Italian leather upper, rubber sole", ["Black", "Brown", "Suede Taupe"], SHOE_SIZES,
         "A sleek Chelsea boot crafted from Italian leather with elastic side panels and a lug sole for traction."),
        ("Men's Pima Cotton Pique Polo", "Men's Tops", 30, 80,
         "100% Pima cotton pique", ["White", "Navy", "Black", "Sage", "Light Blue"], MENS_SIZES,
         "A modern-fit polo in breathable Pima cotton pique. Holds its collar shape wash after wash."),
    ],

    "bags": [
        ("Italian Leather Classic Tote", "Totes", 150, 400,
         "100% Italian pebbled leather", ["Black", "Cognac", "Taupe", "Navy"], ONE_SIZE,
         "A structured everyday tote in full-grain Italian leather. Fits a 15\" laptop with an interior zip pocket and magnetic closure."),
        ("Italian Leather Crossbody Bag", "Crossbody", 80, 220,
         "100% Italian leather", ["Black", "Cognac", "Ivory", "Burgundy"], ONE_SIZE,
         "A compact crossbody with an adjustable strap and gold-tone hardware. Room for phone, wallet, keys, and essentials."),
        ("Italian Leather Mini Crossbody Bag", "Crossbody", 70, 180,
         "100% Italian leather", ["Black", "Tan", "Sage", "Blush"], ONE_SIZE,
         "A petite crossbody for hands-free nights out. Zip-top closure with a card slot inside."),
        ("Italian Leather Bucket Bag", "Shoulder Bags", 120, 300,
         "100% Italian suede and leather", ["Black", "Cognac", "Chocolate"], ONE_SIZE,
         "A slouchy bucket bag with a drawstring closure and detachable shoulder strap. Unlined for a lightweight feel."),
        ("Italian Leather Hobo Bag", "Shoulder Bags", 130, 350,
         "100% Italian nappa leather", ["Black", "Taupe", "Camel"], ONE_SIZE,
         "A soft, slouchy hobo in buttery nappa leather. Comfortably sits on the shoulder with a roomy main compartment."),
        ("Italian Leather Structured Shoulder Bag", "Shoulder Bags", 110, 280,
         "100% Italian smooth leather", ["Black", "Cognac", "Ivory"], ONE_SIZE,
         "A polished shoulder bag with a clean silhouette and turn-lock closure. Transitions seamlessly from desk to dinner."),
        ("Italian Leather Everyday Backpack", "Backpacks", 150, 375,
         "100% Italian pebbled leather", ["Black", "Cognac"], ONE_SIZE,
         "A minimalist leather backpack with a padded laptop sleeve, adjustable straps, and a top grab handle."),
        ("Italian Leather Weekender Duffle Bag", "Travel", 200, 500,
         "100% Italian full-grain leather", ["Cognac", "Black", "Dark Brown"], ONE_SIZE,
         "A carry-on-friendly weekender with a separate shoe compartment, detachable strap, and brass hardware."),
        ("Italian Leather Card Case", "Small Leather Goods", 25, 70,
         "100% Italian leather", ["Black", "Cognac", "Navy", "Red"], ONE_SIZE,
         "A slim card case with four card slots and a center pocket. Fits easily in a front pocket or small bag."),
        ("Italian Leather Zip Around Wallet", "Small Leather Goods", 50, 130,
         "100% Italian pebbled leather", ["Black", "Cognac", "Taupe"], ONE_SIZE,
         "A full-size wallet with 12 card slots, two bill compartments, and a zip coin pocket."),
        ("Italian Leather Belt Bag", "Crossbody", 60, 160,
         "100% Italian leather", ["Black", "Cognac"], ONE_SIZE,
         "A versatile belt bag that wears at the waist or across the body. Adjustable strap and front zip pocket."),
        ("Italian Leather Woven Tote", "Totes", 180, 450,
         "100% Italian leather, hand-woven", ["Black", "Cognac", "Natural"], ONE_SIZE,
         "A hand-woven leather tote made by artisans in Italy. Soft, sculptural, and roomy enough for daily carry."),
        ("Italian Leather Laptop Briefcase", "Work", 170, 425,
         "100% Italian full-grain leather", ["Black", "Cognac"], ONE_SIZE,
         "A slim briefcase with a padded 16\" laptop compartment, organizer pockets, and detachable shoulder strap."),
        ("Italian Leather Evening Clutch", "Clutches", 60, 150,
         "100% Italian leather", ["Black", "Gold Metallic", "Ivory"], ONE_SIZE,
         "A sleek envelope clutch with an optional chain strap. Sized for phone, cards, and lipstick."),
        ("Recycled Nylon Packable Travel Tote", "Travel", 40, 100,
         "100% recycled nylon, leather trim", ["Black", "Navy", "Olive"], ONE_SIZE,
         "A lightweight, water-resistant tote that folds into its own pocket. Leather handles add a refined touch."),
        ("Organic Cotton Canvas Tote", "Totes", 30, 75,
         "100% organic cotton canvas, leather handles", ["Natural", "Navy Stripe", "Black"], ONE_SIZE,
         "A heavyweight canvas tote for groceries, beach days, and errands. Interior pocket and reinforced base."),
        ("Italian Leather Saddle Bag", "Crossbody", 120, 300,
         "100% Italian vegetable-tanned leather", ["Cognac", "Black", "Chocolate"], ONE_SIZE,
         "A curved saddle bag in vegetable-tanned leather that develops a rich patina over time."),
        ("Handwoven Straw Market Tote", "Totes", 50, 120,
         "Natural raffia straw, leather handles", ["Natural", "Natural/Black"], ONE_SIZE,
         "A handwoven market tote for summer. Durable raffia with soft leather handles and a cotton drawstring liner."),
        ("Italian Leather Dopp Kit", "Travel", 50, 120,
         "100% Italian leather, water-resistant lining", ["Black", "Cognac"], ONE_SIZE,
         "A structured toiletry bag with a wipeable lining and wide zip opening for easy packing."),
        ("Italian Leather Phone Crossbody", "Crossbody", 50, 130,
         "100% Italian leather", ["Black", "Taupe", "Sage", "Cognac"], ONE_SIZE,
         "A minimalist phone bag with an exterior card slot and adjustable crossbody strap. Fits most large phones."),
    ],

    "furniture_bed": [
        ("100% European Linen Duvet Cover Set", "Bedding", 120, 300,
         "100% European flax linen, stone-washed", ["White", "Natural", "Sage", "Terracotta", "Slate"], DUVET_SIZES,
         "Stone-washed linen duvet cover with two matching shams. Breathable, temperature-regulating, and softer with every wash."),
        ("Organic Percale Sheet Set", "Bedding", 90, 230,
         "100% GOTS-certified organic cotton percale, 400TC", ["White", "Ivory", "Light Grey", "Navy"], BED_SIZES,
         "Crisp, cool percale sheets in long-staple organic cotton. Set includes flat sheet, fitted sheet, and two pillowcases."),
        ("Bamboo Lyocell Sheet Set", "Bedding", 110, 280,
         "100% bamboo lyocell", ["White", "Sand", "Grey", "Blue"], BED_SIZES,
         "Silky, cooling sheets ideal for hot sleepers. Moisture-wicking bamboo lyocell with deep-pocket fitted sheet."),
        ("Organic Sateen Sheet Set", "Bedding", 100, 250,
         "100% GOTS-certified organic cotton sateen, 400TC", ["White", "Champagne", "Charcoal", "Sage"], BED_SIZES,
         "Smooth, lustrous sateen with a subtle sheen. Woven from long-staple organic cotton for a hotel-quality feel."),
        ("Down Alternative Comforter", "Bedding", 120, 300,
         "Cotton shell, hypoallergenic microfiber fill", ["White"], DUVET_SIZES,
         "A lofty all-season comforter with box-stitch construction to prevent shifting. Hypoallergenic and machine washable."),
        ("European White Down Comforter", "Bedding", 200, 500,
         "Cotton sateen shell, 650-fill RDS-certified European white down", ["White"], DUVET_SIZES,
         "Cloud-like warmth from responsibly sourced European down. Baffle-box construction keeps fill evenly distributed."),
        ("100% European Linen Quilt", "Bedding", 150, 375,
         "100% European linen face, cotton fill", ["White", "Natural", "Clay"], DUVET_SIZES,
         "A lightweight quilt with channel stitching. Perfect alone in summer or layered over a duvet in winter."),
        ("Turkish Cotton Bath Towel Set", "Bath", 50, 120,
         "100% Turkish cotton, 650 GSM", ["White", "Grey", "Navy", "Sand"], ["Set of 2", "Set of 4"],
         "Plush, absorbent towels woven in Turkey from long-staple cotton. Gets fluffier with every wash."),
        ("Organic Cotton Waffle Knit Blanket", "Throws & Blankets", 80, 200,
         "100% GOTS-certified organic cotton", ["White", "Oatmeal", "Sage", "Charcoal"], ["Throw", "Full/Queen", "King"],
         "A breathable waffle-weave blanket for year-round layering. Textured, lightweight, and naturally soft."),
        ("European White Down Pillow", "Pillows", 60, 150,
         "Cotton shell, 750-fill European white down", ["White"], ["Standard", "King"],
         "A medium-firm down pillow that cradles your head and neck. Available in standard and king sizes."),
        ("Solid Oak Bed Frame", "Furniture", 900, 2200,
         "Solid white oak, natural oil finish", ["Natural Oak", "Walnut Stain"], ["Queen", "King"],
         "A minimalist platform bed crafted from solid white oak. No box spring needed; assembles in under 30 minutes."),
        ("Solid Wood Two-Drawer Nightstand", "Furniture", 300, 700,
         "Solid acacia wood, soft-close drawers", ["Natural", "Walnut"], ONE_SIZE,
         "A compact nightstand with two soft-close drawers and a clean mid-century profile."),
        ("Performance Linen Upholstered Platform Bed", "Furniture", 1100, 2600,
         "Kiln-dried wood frame, performance linen-blend upholstery", ["Natural", "Grey", "Ivory"], ["Queen", "King"],
         "A tailored upholstered bed with a padded headboard in stain-resistant performance linen."),
        ("Performance Linen Three-Seat Sofa", "Furniture", 1500, 3600,
         "Kiln-dried hardwood frame, performance linen, feather-wrapped cushions", ["Ivory", "Oatmeal", "Slate"], ONE_SIZE,
         "A deep-seated sofa with feather-wrapped cushions and stain-resistant performance fabric. Built to sink into."),
        ("Italian Leather Lounge Chair", "Furniture", 900, 2200,
         "Full-grain Italian leather, solid oak frame", ["Cognac", "Black"], ONE_SIZE,
         "A low-slung lounge chair in full-grain Italian leather that patinas beautifully over time."),
        ("Marble Round Side Table", "Furniture", 250, 600,
         "Solid Carrara marble", ["White Carrara", "Black Marquina"], ONE_SIZE,
         "A sculptural pedestal side table carved from solid marble. Every piece has unique natural veining."),
        ("Solid Oak Dining Table", "Furniture", 1200, 2800,
         "Solid white oak, matte lacquer finish", ["Natural Oak"], ['72"', '84"'],
         "A timeless dining table in solid white oak that seats six to eight. Finished to resist water and stains."),
        ("Handwoven Jute Area Rug", "Rugs", 150, 400,
         "100% natural jute", ["Natural"], RUG_SIZES,
         "A chunky handwoven jute rug that adds texture and warmth. Durable enough for high-traffic areas."),
        ("Hand-Knotted Wool Area Rug", "Rugs", 400, 1000,
         "100% New Zealand wool, hand-knotted", ["Ivory", "Grey", "Rust"], RUG_SIZES,
         "A hand-knotted wool rug with a subtle tonal pattern. Naturally stain-resistant and incredibly soft underfoot."),
        ("Handmade Ceramic Table Lamp", "Lighting", 100, 250,
         "Handmade ceramic base, linen shade", ["Matte White", "Sand", "Charcoal"], ONE_SIZE,
         "A handmade ceramic lamp with a reactive glaze and natural linen shade. Each base is one of a kind."),
    ],

    "jewelry": [
        ("14K Gold Huggie Hoop Earrings", "Earrings", 50, 150,
         "Solid 14K yellow gold", ["Yellow Gold", "White Gold"], ONE_SIZE,
         "Everyday huggie hoops in solid 14K gold. Hinged closure for easy on and off; comfortable enough to sleep in."),
        ("14K Gold Paperclip Chain Necklace", "Necklaces", 150, 400,
         "Solid 14K yellow gold", ["Yellow Gold"], CHAIN_LENGTHS,
         "A modern paperclip-link chain in solid 14K gold. Wear alone or layer with pendants."),
        ("Lab Grown Diamond Stud Earrings", "Earrings", 200, 800,
         "14K gold, lab grown diamonds (0.5 ctw), VS clarity, F-G color", ["Yellow Gold", "White Gold"], ["0.5 ctw", "1.0 ctw", "2.0 ctw"],
         "Brilliant-cut lab grown diamond studs in a four-prong basket setting. Chemically identical to mined diamonds, IGI-certified."),
        ("Lab Grown Diamond Tennis Bracelet", "Bracelets", 1000, 3500,
         "14K gold, lab grown diamonds (3 ctw)", ["Yellow Gold", "White Gold"], ['6.5"', '7"'],
         "A classic tennis bracelet set with round brilliant lab grown diamonds and a secure box clasp."),
        ("Freshwater Pearl Stud Earrings", "Earrings", 30, 90,
         "Freshwater cultured pearls, 14K gold posts", ["White"], ["6mm", "8mm"],
         "Lustrous freshwater pearl studs on solid 14K gold posts. A timeless finishing touch."),
        ("Freshwater Pearl Strand Necklace", "Necklaces", 100, 300,
         "Freshwater cultured pearls, 14K gold clasp", ["White"], CHAIN_LENGTHS,
         "A hand-knotted strand of round freshwater pearls with a 14K gold clasp. An heirloom-worthy classic."),
        ("14K Gold Oval Signet Ring", "Rings", 200, 500,
         "Solid 14K yellow gold", ["Yellow Gold"], RING_SIZES,
         "A polished oval signet ring with a substantial weight. Ready for engraving or to wear as-is."),
        ("Sterling Silver Chunky Hoop Earrings", "Earrings", 40, 110,
         "925 sterling silver", ["Silver"], ["Small", "Medium", "Large"],
         "Bold, lightweight tube hoops in recycled 925 sterling silver. Hollow construction for all-day comfort."),
        ("Gold Vermeil Herringbone Necklace", "Necklaces", 60, 160,
         "18K gold vermeil over sterling silver", ["Gold"], CHAIN_LENGTHS,
         "A sleek, liquid-like herringbone chain in thick 18K gold vermeil. Lies flat and catches the light beautifully."),
        ("Lab Grown Diamond Solitaire Pendant", "Necklaces", 300, 900,
         "14K gold, lab grown diamond (0.5 ct)", ["Yellow Gold", "White Gold"], CHAIN_LENGTHS,
         "A single round lab grown diamond suspended on a delicate cable chain. Simple, elegant, everyday sparkle."),
        ("Lab Grown Diamond Eternity Band", "Rings", 600, 1800,
         "14K gold, lab grown diamonds (1.5 ctw)", ["Yellow Gold", "White Gold"], RING_SIZES,
         "A full eternity band with shared-prong round lab grown diamonds. Ideal as a wedding band or stacking ring."),
        ("14K Gold Initial Pendant Necklace", "Necklaces", 100, 250,
         "Solid 14K yellow gold", ["Yellow Gold"], CHAIN_LENGTHS,
         "A personalized initial pendant on a fine cable chain. Choose any letter A-Z at checkout."),
        ("Sterling Silver Sculptural Cuff Bracelet", "Bracelets", 50, 130,
         "925 sterling silver", ["Silver", "Gold Vermeil"], ONE_SIZE,
         "An organically shaped open cuff with a high-polish finish. Slightly adjustable for a comfortable fit."),
        ("14K Gold Bead Bracelet", "Bracelets", 90, 230,
         "Solid 14K yellow gold", ["Yellow Gold"], ['6.5"', '7"'],
         "Delicate polished gold beads on a fine chain. Perfect for stacking or wearing solo."),
        ("Freshwater Pearl Drop Earrings", "Earrings", 40, 110,
         "Freshwater cultured pearls, 14K gold vermeil", ["White"], ONE_SIZE,
         "Baroque pearl drops on gold vermeil huggies. Each pearl is uniquely shaped for an organic, modern look."),
        ("Lab Grown Emerald Cocktail Ring", "Rings", 400, 1100,
         "14K gold, lab grown emerald (2 ct), lab grown diamond halo", ["Yellow Gold"], RING_SIZES,
         "An emerald-cut lab grown emerald framed in a halo of lab grown diamonds. Rich color without the mined markup."),
        ("14K Gold Rope Chain Necklace", "Necklaces", 250, 650,
         "Solid 14K yellow gold", ["Yellow Gold"], CHAIN_LENGTHS,
         "A twisted rope chain in solid 14K gold that catches light from every angle. Lobster clasp closure."),
        ("Gold Vermeil Stacking Ring Set", "Rings", 50, 130,
         "18K gold vermeil over sterling silver", ["Gold"], RING_SIZES,
         "A set of three stackable rings: one smooth, one twisted, one beaded. Wear together or mix and match."),
        ("Lab Grown Diamond Bezel Station Necklace", "Necklaces", 350, 950,
         "14K gold, lab grown diamonds (0.5 ctw)", ["Yellow Gold", "White Gold"], CHAIN_LENGTHS,
         "Five bezel-set lab grown diamonds spaced along a fine chain for an effortless, scattered sparkle."),
        ("Men's Sterling Silver Curb Chain Necklace", "Men's Jewelry", 80, 200,
         "925 sterling silver", ["Silver"], ['20"', '22"', '24"'],
         "A substantial 5mm curb chain in polished sterling silver. Secure lobster clasp; tarnish-resistant finish."),
        # Added later so jewelry isn't almost all gold: silver with stones, and stainless steel.
        ("Sterling Silver Jade Pendant Necklace", "Necklaces", 90, 220,
         "925 sterling silver, natural nephrite jade", ["Silver/Green"], CHAIN_LENGTHS,
         "A smooth oval of natural nephrite jade in a sterling silver bezel on a fine cable chain. Each stone's green varies slightly."),
        ("Lab Grown Ruby Sterling Silver Ring", "Rings", 120, 300,
         "925 sterling silver, lab grown ruby (1 ct)", ["Silver/Red"], RING_SIZES,
         "An oval lab grown ruby in a four-prong sterling silver setting. Deep red color, chemically identical to mined ruby."),
        ("Sterling Silver Onyx Stud Earrings", "Earrings", 45, 110,
         "925 sterling silver, black onyx", ["Silver/Black"], ONE_SIZE,
         "Round black onyx cabochons in polished sterling silver settings. A graphic, everyday alternative to diamond studs."),
        ("Sterling Silver Turquoise Cuff Bracelet", "Bracelets", 110, 280,
         "925 sterling silver, stabilized turquoise", ["Silver/Turquoise"], ONE_SIZE,
         "A hammered sterling silver cuff set with a single stabilized turquoise stone. Open back for an adjustable fit."),
        ("Men's Stainless Steel ID Bracelet", "Men's Jewelry", 45, 110,
         "316L stainless steel", ["Steel", "Black"], ['7.5"', '8"', '8.5"'],
         "A brushed stainless steel ID bracelet with a flat engravable plate. Waterproof, scratch-resistant, and won't tarnish."),
        ("Stainless Steel Everyday Hoop Earrings", "Earrings", 30, 75,
         "316L stainless steel, hypoallergenic", ["Silver", "Gold Tone"], ONE_SIZE,
         "Lightweight tube hoops in hypoallergenic stainless steel. Shower-, sweat-, and swim-proof for true everyday wear."),
    ],
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _build_catalog(seed: int = 42) -> list[Product]:
    """Turn the raw rows into full Product records with deterministic mock data."""
    rng = random.Random(seed)
    margin_rng = random.Random(seed + 1)  # separate stream so the other mock fields stay unchanged
    products: list[Product] = []
    next_id = 1

    for category, rows in RAW_PRODUCTS.items():
        prefix = CATEGORIES[category]["sku_prefix"]
        for i, (name, subcat, price, compare_at, materials, colors, sizes, desc) in enumerate(rows, start=1):
            slug = _slugify(name)
            products.append(Product(
                id=next_id,
                sku=f"{prefix}-{i:03d}",
                slug=slug,
                name=name,
                category=category,
                subcategory=subcat,
                description=desc,
                price=float(price),
                compare_at_price=float(compare_at),
                unit_cost=round(price * (1 - _mock_margin(category, subcat, name, margin_rng)), 2),
                currency="USD",
                materials=materials,
                colors=list(colors),
                sizes=list(sizes),
                care_instructions=CARE[category],
                inventory={s: rng.choice([0, rng.randint(5, 400)]) if rng.random() < 0.1 else rng.randint(5, 400)
                           for s in sizes},
                tags=sorted({category, _slugify(subcat), *_slugify(materials).split("-")[:3]}),
                rating=round(rng.uniform(4.3, 4.9), 1),
                review_count=rng.randint(40, 6000),
                status="active",
                is_featured=rng.random() < 0.2,
                created_at=(date(2026, 1, 1) + timedelta(days=rng.randint(0, 290))).isoformat(),
                seo_title=f"{name} | Quince",
                seo_description=desc[:155],
                image_urls=[f"https://images.example.com/{slug}/{c.lower().replace(' ', '-').replace('/', '-')}.jpg"
                            for c in colors],
            ))
            next_id += 1
    return products


PRODUCTS: list[Product] = _build_catalog()
PRODUCTS_BY_SKU: dict[str, Product] = {p.sku: p for p in PRODUCTS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_by_category(category: str) -> list[Product]:
    return [p for p in PRODUCTS if p.category == category]


def get_by_sku(sku: str) -> Optional[Product]:
    return PRODUCTS_BY_SKU.get(sku)


def search(query: str) -> list[Product]:
    q = query.lower()
    return [p for p in PRODUCTS if q in p.name.lower() or q in p.description.lower()]


def to_dicts() -> list[dict]:
    return [asdict(p) for p in PRODUCTS]


def to_json(indent: int = 2) -> str:
    return json.dumps(to_dicts(), indent=indent)


if __name__ == "__main__":
    if "--json" in sys.argv:
        print(to_json())
    else:
        print(f"Total products: {len(PRODUCTS)}\n")
        for key, meta in CATEGORIES.items():
            items = get_by_category(key)
            prices = [p.price for p in items]
            print(f"{meta['label']:<20} {len(items):>3} items   ${min(prices):,.0f} - ${max(prices):,.0f}")
        sample = PRODUCTS[0]
        print(f"\nSample record ({sample.sku}):")
        print(json.dumps(asdict(sample), indent=2))
