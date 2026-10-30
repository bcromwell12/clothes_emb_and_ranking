# Masked set model: results

193 menus; 39 multi-slot menus held out of training (never seen), 154 trained on. Every product is in training. Held-out menus test composition: each of their header values was learned on other menus.

Held out: Home · Linen; Home · Organic Cotton; Unisex · Leather; Unisex · Organic Cotton; Women · Diamond; Women · Pearl; Bags · Unisex; Bags · Work; Bags · $150 to $500; Bags · Crossbody & Belt Bags · Gift; Bags · Shoulder Bags · Women; Bags · Totes; Bags · Work & Travel · Travel; Cashmere Sweaters · Lounge & Sleep; Cashmere Sweaters · Work; Cashmere Sweaters · Under $50; Cashmere Sweaters · Cold-Weather Accessories · Gift; Cashmere Sweaters · Sweaters · Cold Weather; Furniture & Bed · Wood; Furniture & Bed · Cold Weather; Furniture & Bed · $50 to $150; Furniture & Bed · Furniture · Wood; Furniture & Bed · Furniture · Everyday; General Clothing · Organic Cotton; General Clothing · Under $50; General Clothing · Outerwear · Unisex; General Clothing · Tops · Men; Jewelry · Diamond; Jewelry · Everyday; Jewelry · Earrings · Everyday; Jewelry · Necklaces; Jewelry · Necklaces · Gold; Cashmere · Gift; Cashmere · Lounge & Sleep; Leather · Everyday; Sterling Silver · Everyday; Wood · Everyday; Cold Weather · Under $50; Everyday · $150 to $500

## 1. Filling in held-out menus

*P@5*: share of the top 5 on the menu. *R@10*: share of the menu in the top 10. *mAP*: mean average precision.

### From the header alone

| | P@5 | R@10 | mAP |
|---|---:|---:|---:|
| MPNet (raw) | 0.415 | 0.587 | 0.494 |
| Skip-gram tokens | 0.513 | 0.778 | 0.614 |
| Adapter only (item only) | 0.559 | 0.730 | 0.624 |
| Adapter only (item + header) | 0.697 | 0.855 | 0.836 |
| Set transformer, item only | 0.744 | 0.884 | 0.896 |
| Set transformer, item + header | 0.728 | 0.888 | 0.891 |
| Deep Sets, item + header | 0.626 | 0.802 | 0.800 |

### From the header plus 2 listed products

| | P@5 | R@10 | mAP |
|---|---:|---:|---:|
| MPNet (raw) | 0.333 | 0.614 | 0.478 |
| Skip-gram tokens | 0.426 | 0.801 | 0.659 |
| Adapter only (item only) | 0.513 | 0.918 | 0.900 |
| Adapter only (item + header) | 0.544 | 0.921 | 0.916 |
| Set transformer, item only | 0.497 | 0.902 | 0.907 |
| Set transformer, item + header | 0.503 | 0.912 | 0.885 |
| Deep Sets, item + header | 0.451 | 0.888 | 0.832 |

### From the header written as a free-text phrase

| | P@5 | R@10 | mAP |
|---|---:|---:|---:|
| MPNet (raw) | 0.405 | 0.574 | 0.481 |
| Adapter only (item only) | 0.523 | 0.683 | 0.619 |
| Adapter only (item + header) | 0.662 | 0.840 | 0.793 |
| Set transformer, item only | 0.733 | 0.876 | 0.878 |
| Set transformer, item + header | 0.713 | 0.861 | 0.867 |
| Deep Sets, item + header | 0.641 | 0.773 | 0.805 |

Same metric on menus seen in training, header alone (the memorization ceiling):

| | P@5 | R@10 | mAP |
|---|---:|---:|---:|
| MPNet (raw) | 0.412 | 0.478 | 0.440 |
| Set transformer, item + header | 0.877 | 0.912 | 1.000 |

## 2. Naming a list

Show 3 products from a held-out menu, no header, and name each slot of its header. Correct if the top answer is a value all 3 products have.

| | audience | all | material | department | occasion | price | type |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNet (raw) | 0.300 | 0.618 | 0.812 | 0.808 | 0.389 | 0.500 | 0.615 |
| Skip-gram tokens | 0.900 | 0.831 | 1.000 | 1.000 | 0.389 | 0.500 | 1.000 |
| Adapter only (item only) | 0.500 | 0.775 | 0.688 | 1.000 | 0.667 | 0.667 | 0.846 |
| Adapter only (item + header) | 1.000 | 0.921 | 1.000 | 1.000 | 0.722 | 0.667 | 1.000 |
| Set transformer, item + header | 1.000 | 0.978 | 1.000 | 1.000 | 1.000 | 0.667 | 1.000 |
| Deep Sets, item + header | 1.000 | 0.955 | 1.000 | 1.000 | 0.944 | 0.500 | 1.000 |

## 3. The product space itself

Leave-one-out 5-NN: share of each product's 5 nearest neighbors sharing its value. In-sample (the taxonomy shaped training), so read it as how much structure the geometry absorbed; price is not in the text.

| | department | type | material | audience | occasion | price | cos mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNet (raw) | 0.798 | 0.432 | 0.585 | 0.621 | 0.530 | 0.445 | 0.365 |
| Skip-gram item vectors | 0.989 | 0.504 | 0.704 | 0.749 | 0.792 | 0.721 | 0.267 |
| Adapter (item only) | 0.962 | 0.600 | 0.715 | 0.796 | 0.802 | 0.715 | -0.006 |
| Adapter (item + header) | 0.970 | 0.632 | 0.719 | 0.806 | 0.766 | 0.702 | -0.003 |
| Adapter (Deep Sets, item + header) | 0.964 | 0.615 | 0.708 | 0.789 | 0.781 | 0.715 | -0.001 |

## 4. Free-text queries (top 5)

**cozy gift for mom**

- MPNet (raw): 100% European Linen Quilt; Washable Stretch Silk Button-Down Shirt; Mongolian Cashmere Ribbed Beanie; 100% European Linen Duvet Cover Set; Down Alternative Comforter
- Adapter only: Mongolian Cashmere Jogger Pants; 14K Gold Huggie Hoop Earrings; Men's Mongolian Cashmere Hoodie; Washable Silk Pajama Set; 14K Gold Bead Bracelet
- Set transformer: Men's Mongolian Cashmere Hoodie; Men's Mongolian Cashmere V-Neck Sweater; Men's Mongolian Cashmere Turtleneck Sweater; Mongolian Cashmere Jogger Pants; Mongolian Cashmere Oversized Boyfriend Sweater

**something to wear to a wedding**

- MPNet (raw): Washable Stretch Silk Button-Down Shirt; Mongolian Cashmere Cropped Polo Sweater; Mongolian Cashmere Turtleneck Sweater; Men's Mongolian Cashmere Shawl Collar Cardigan; Washable Silk Pajama Set
- Adapter only: Lab Grown Diamond Eternity Band; 14K Gold Oval Signet Ring; Freshwater Pearl Stud Earrings; Freshwater Pearl Drop Earrings; Freshwater Pearl Strand Necklace
- Set transformer: Freshwater Pearl Strand Necklace; 100% Leather Moto Jacket; 14K Gold Initial Pendant Necklace; Lab Grown Diamond Eternity Band; 14K Gold Oval Signet Ring

**warm sweater for him to wear to the office**

- MPNet (raw): Mongolian Cashmere Sweater Vest; Men's Mongolian Cashmere Hoodie; Mongolian Cashmere Ribbed Beanie; Men's Mongolian Cashmere Shawl Collar Cardigan; Mongolian Cashmere Oversized Boyfriend Sweater
- Adapter only: Men's Mongolian Cashmere Turtleneck Sweater; Men's Mongolian Cashmere V-Neck Sweater; Mongolian Cashmere Sweater Vest; Mongolian Cashmere Turtleneck Sweater; Men's Mongolian Cashmere Crewneck Sweater
- Set transformer: Men's Mongolian Cashmere V-Neck Sweater; Men's Mongolian Cashmere Turtleneck Sweater; Mongolian Cashmere Ribbed Beanie; Organic Cotton Waffle Knit Blanket; Men's Mongolian Cashmere Quarter Zip Sweater

**summer vacation essentials**

- MPNet (raw): Organic Cotton Canvas Tote; Italian Leather Everyday Backpack; Italian Leather Laptop Briefcase; Italian Leather Weekender Duffle Bag; Italian Leather Bucket Bag
- Adapter only: Bamboo Lyocell Sheet Set; Recycled Nylon Packable Travel Tote; Handwoven Straw Market Tote; Organic Cotton Canvas Tote; Recycled Nylon Square Neck One-Piece Swimsuit
- Set transformer: Recycled Nylon Packable Travel Tote; Responsible Down Puffer Jacket; Italian Leather Dopp Kit; Recycled Nylon Square Neck One-Piece Swimsuit; Bamboo Lyocell Sheet Set

**gold jewelry under $100**

- MPNet (raw): 14K Gold Bead Bracelet; Freshwater Pearl Strand Necklace; 14K Gold Paperclip Chain Necklace; Lab Grown Diamond Tennis Bracelet; Men's Stainless Steel ID Bracelet
- Adapter only: Freshwater Pearl Drop Earrings; Freshwater Pearl Stud Earrings; Lab Grown Diamond Stud Earrings; Freshwater Pearl Strand Necklace; Lab Grown Diamond Eternity Band
- Set transformer: Freshwater Pearl Drop Earrings; Freshwater Pearl Stud Earrings; Freshwater Pearl Strand Necklace; 14K Gold Initial Pendant Necklace; 14K Gold Huggie Hoop Earrings

**furnish a bedroom**

- MPNet (raw): Performance Linen Upholstered Platform Bed; Solid Wood Two-Drawer Nightstand; Solid Oak Bed Frame; Solid Oak Dining Table; 100% European Linen Duvet Cover Set
- Adapter only: Bamboo Lyocell Sheet Set; Freshwater Pearl Stud Earrings; Organic Percale Sheet Set; Freshwater Pearl Drop Earrings; Organic Sateen Sheet Set
- Set transformer: 100% European Linen Wide Leg Pants; Handwoven Straw Market Tote; Stainless Steel Everyday Hoop Earrings; Bamboo Lyocell Sheet Set; Organic Cotton Canvas Tote
