"""Pub 3 : sous-titres bleu marine + étiquettes recouverts en français, appareil renommé Vigilair."""
import json, subprocess, re, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 360, 640, 2, 24
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts") + "/"
NAVY = (8, 1, 52)
_fc = {}
def fontw(px, w=700):
    key = (max(6, int(px)), w)
    if key not in _fc: _fc[key] = ImageFont.truetype(FD + f"Poppins-{w}.ttf", key[0])
    return _fc[key]
def font(px): return fontw(px, 700)
STATE = {}

SPECS = [
 (138.9, 142.6, "plain", 102, 128, 60, 300, "DÉTECTEUR DE BRICOLAGE"),
 (140.3, 142.3, "red",   452, 486, 120, 360, "ALARME À 70 PPM SEULEMENT"),
 (142.7, 143.95, "red",  100, 134, 0, 360, "SOUS 70 PPM"),
 (165.2, 167.8, "green", 482, 520, 150, 360, "DÈS 0 PPM"),
 (155.5, 157.2, "plain", 100, 130, 60, 300, "VIGILAIR"),
 (155.5, 157.2, "green", 195, 228, 0, 200, "4 EN 1"),
 (155.5, 157.2, "plain", 248, 272, 20, 180, "DÉTECTEUR CO"),
 (160.2, 165.25, "plain", 104, 132, 20, 200, "VRAIS CHIFFRES"),
 (160.2, 165.25, "red",  152, 180, 0, 210, "MONOXYDE DE CARBONE"),
 (160.2, 165.25, "green", 199, 227, 0, 210, "GAZ NATUREL"),
 (160.2, 165.25, "green", 246, 274, 0, 210, "PROPANE"),
 (160.2, 165.25, "green", 296, 324, 0, 210, "GAZ INFLAMMABLES"),
 (177.5, 178.9, "plain", 102, 134, 80, 290, "VIGILAIR"),
 (177.5, 178.9, "blue",  238, 268, 0, 200, "LIEN SOUS"),
 (177.5, 178.9, "blue",  278, 308, 0, 200, "LA VIDÉO"),
 (188.0, 190.7, "plain", 102, 134, 80, 290, "VIGILAIR"),
 (188.0, 190.7, "blue",  238, 268, 0, 200, "LIEN SOUS"),
 (188.0, 190.7, "blue",  278, 308, 0, 200, "LA VIDÉO"),
 (185.35, 188.05, "plain", 110, 136, 110, 250, "PLUS DE"),
 (185.35, 188.05, "plain", 140, 184, 80, 280, "12 000"),
 (185.35, 188.05, "navy", 216, 250, 0, 360, "FAMILLES FRANÇAISES"),
]
NO_SUB = []
def cmask(name, f):
    r, g, b = [f[..., c].astype(int) for c in range(3)]
    if name == "navy":   return (b > 22) & (b < 120) & (r < 60) & (g < 55) & (b - r > 15)
    if name == "red":    return (r > 170) & (g < 90) & (b < 100)
    if name in ("green", "circle"): return (g > 100) & (r < 100) & (b < 120) & (g - r > 45)
    if name == "orange": return (r > 200) & (g > 110) & (g < 200) & (b < 90)
    if name == "blue":   return (b > 160) & (r < 100) & (g > 50) & (g < 160)
    if name in ("white", "whitebg", "plain"):  return (r > 220) & (g > 220) & (b > 220)

def spec_box(sp, f):
    t0, t1, col, y0, y1, x0, x1, txt = sp
    sub = f[y0:y1, x0:x1]
    m = cmask(col, sub)
    if col == "navy":   # lignes contenant un vrai bandeau (≥30 px)
        rows = m.sum(1) >= 30
        if rows.sum() < 6: return None
        ys = np.where(rows)[0]
        cols = np.where(m[ys].sum(0) >= max(3, len(ys) * 0.3))[0]
        if len(cols) and cmask("white", sub[ys.min():ys.max()+1, cols.min():cols.max()+1]).sum() < 15: return None
        wm = cmask("white", sub[ys.min():ys.max()+1])
        wc = np.where(wm.sum(0) > 0)[0]
        if len(wc):
            hh = ys.max() - ys.min() + 1
            lo, hi = wc.min() - 1.5 * hh, wc.max() + 1.5 * hh
            mw = (m[ys] | cmask("white", sub[ys])).sum(0) >= max(3, len(ys) * 0.3)
            c2 = np.where(mw)[0]
            c2 = c2[(c2 >= lo) & (c2 <= hi)]
            if len(c2):
                runs = np.split(c2, np.where(np.diff(c2) > 3)[0] + 1)
                best = max(runs, key=lambda r: len(np.intersect1d(r, wc)))
                cols = np.arange(min(best.min(), wc.min()), max(best.max(), wc.max()) + 1)
    else:
        if m.sum() < 60: return None
        ys = np.where(m.sum(1) >= 6)[0]
        if len(ys) < 5: return None
        cols = np.where(m[ys].sum(0) >= 2)[0]
    if len(cols) == 0: return None
    if col in ("white", "whitebg"):
        return [x0 + cols.min() - 8, y0 + ys.min() - 6, x0 + cols.max() + 8, y0 + ys.max() + 6]
    if col == "circle":
        return [x0 + 2, y0 + ys.min(), x1 - 2, y0 + ys.max()]
    return [x0 + cols.min() - 1, y0 + ys.min() - 1, x0 + cols.max() + 1, y0 + ys.max() + 1]

def median_color(col, f, box):
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = f[max(0, y0):y1, max(0, x0):x1]
    m = cmask("green" if col == "circle" else col, sub)
    if col == "whitebg":
        ring = np.concatenate([f[max(0,y0-4), max(0,x0):x1], f[min(H-1,y1+4), max(0,x0):x1]])
        return tuple(int(v) for v in np.median(ring, axis=0))
    if col == "white" or m.sum() < 10: return NAVY if col in ("white", "navy") else None
    return tuple(int(v) for v in np.median(sub[m], axis=0))


# ---------------- sous-titres FR ----------------
seq = json.load(open("words_new.json"))
chunks = []; cur = []
for n_, w in enumerate(seq):
    if cur and ((w["w"][:1].isupper() and w["w"] not in ("Vigilair", "Vigilair.", "CO")) or w.get("blk") != cur[-1].get("blk")):
        chunks.append(cur); cur = []
    cur.append(w)
    txt = " ".join(x["w"] for x in cur)
    nxt_w = seq[n_ + 1]["w"] if n_ + 1 < len(seq) else ""
    glued = w["w"].isdigit() and nxt_w[:1].isdigit()
    if not glued and (len(cur) >= 4 or len(txt) >= 20 or re.search(r"[,.;:?!…»]$", w["w"])):
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
merged = []
for c in chunks:
    if merged and len(" ".join(x["w"] for x in c)) <= 3 and c[0].get("blk") == merged[-1][-1].get("blk"):
        merged[-1] = merged[-1] + c
    else: merged.append(c)
CH = []
for i, c in enumerate(merged):
    s = c[0]["s"]; e = c[-1]["e"]
    nxt = merged[i + 1][0]["s"] if i + 1 < len(merged) else 1e9
    CH.append((s, nxt if nxt - e < 2.5 else e + 0.4, " ".join(x["w"] for x in c)))
def chunk_at(t):
    for s, e, txt in CH:
        if s <= t < e: return txt
    return None
def prev_chunk(t):
    pc = [c for c in CH if c[0] <= t]
    return pc[-1][2] if pc else None

NB = json.load(open("wboxes.json"))
DEV_SHOTS = []
def extend(b, f):
    x0, y0, x1, y1 = b
    band = f[int(y0)+4:int(y1)-4]
    wh = cmask("white", band); nv = cmask("navy", band) | wh
    col_ok = (nv.mean(0) > 0.6) | (wh.sum(0) > 0)
    while x1 < W - 1 and col_ok[int(x1):min(W, int(x1) + 12)].any() and nv[:, int(x1):min(W, int(x1) + 12)].mean() > 0.35: x1 += 3
    while x0 > 1 and col_ok[max(0, int(x0) - 12):int(x0)].any() and nv[:, max(0, int(x0) - 12):int(x0)].mean() > 0.35: x0 -= 3
    return [x0, y0, x1, y1]

def rrect(d, box, color, rad):
    d.rounded_rectangle([int(v) for v in box], radius=int(rad), fill=color)

def draw_label(d, box, color, txt, maxh=None, minw=0, big=False):
    """box en coords source ; dessine en coords x S."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    fs = min(h, 30) * 0.56 * S
    if big: fs = (h - 12) * 1.45 * S
    f = font(fs)
    tw = d.textlength(txt, font=f)
    padx = min(h, 30) * 0.42 * S
    need = tw + 2 * padx
    maxw = (W - 8) * S
    if need > maxw:
        f = font(fs * maxw / need); tw = d.textlength(txt, font=f); need = maxw
    cx = (x0 + x1) / 2 * S
    bw = max((x1 - x0) * S + 4, need, minw)
    bx0 = max(2 * S, cx - bw / 2); bx1 = min((W - 2) * S, bx0 + bw); bx0 = bx1 - bw if bx1 - bw >= 0 else 0
    by0 = y0 * S - 2; by1 = y1 * S + 2
    rrect(d, (bx0, by0, bx1, by1), color, min(9 * S, (by1 - by0) * 0.28))
    tb = d.textbbox((0, 0), txt, font=f, anchor="ls")
    th = -tb[1]
    cxt = (bx0 + bx1) / 2
    base = (by0 + by1) / 2 + th * 0.5 - 1
    d.text((cxt, base), txt, font=f, fill=(255, 255, 255), anchor="ms")

import cv2
# ---------------- textes de l'appareil ----------------
f166 = np.array(Image.open("f166.png").convert("L")).astype(np.float32)
f190 = np.array(Image.open("f190.png").convert("L")).astype(np.float32)
f152 = np.array(Image.open("f152.png").convert("L")).astype(np.float32)
f162 = np.array(Image.open("f162.png").convert("L")).astype(np.float32)
TPL = {
    "row": ([f166[429:450, 70:300], f152[325:341, 130:235]], [("Marche", 0.12), ("Alarme", 0.47), ("Défaut", 0.84)]),
    "gas": ([f166[553:577, 100:264], f152[388:404, 140:220], f190[369:385, 142:210]], [("Détecteur gaz & CO", 0.5)]),
    "logo": ([f190[403:418, 157:206], f152[425:443, 155:208], f162[408:421, 150:210]], [("Vigilair", 0.5)]),
}
SCALES = {"row": np.geomspace(0.35, 2.6, 34), "gas": np.geomspace(0.25, 2.2, 34), "logo": np.geomspace(0.5, 3.2, 34)}
DEV_WIN = (0, 0)
DEV_SHOTS_OLD = [(171.43, 173.17), (175.13, 179.23), (181.67, 185.3), (213.7, 220.4), (226.3, 236.1), (248.7, 253.1), (280.3, 284.9)]

def device_fix(f):
    g = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
    draws = []; mask = np.zeros((H, W), np.uint8)
    for k, (tps, words_) in TPL.items():
        best = None
        for tp, sc in [(tp, sc) for tp in tps for sc in SCALES[k]]:
            t = cv2.resize(tp, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            if t.shape[0] < 4 or t.shape[1] < 14 or t.shape[0] >= H or t.shape[1] >= W: continue
            res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if best is None or mv > best[0]: best = (mv, ml[0], ml[1], t.shape[1], t.shape[0])
        thr = {"row": 0.45, "gas": 0.5, "logo": 0.6}[k]
        if best is None or best[0] < thr: continue
        mv, x, y, w, h = best
        if k == "gas": STATE["gas"] = (x, y, w, h)
        if w < 22: continue  # trop petit pour être lisible
        sub = f[y:y + h, x:x + w].astype(int)
        lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
        m = ((lum > np.median(lum) + 18) & (sat < 50)).astype(np.uint8)
        if m.sum() < 4: continue
        col = tuple(int(v) for v in np.median(sub[m.astype(bool)], axis=0))
        mask[y:y + h, x:x + w] |= cv2.dilate(m, np.ones((3, 3), np.uint8))
        hh = h * (0.62 if k != "row" else 0.55)
        for wd, rx in words_:
            draws.append((x + w * rx, y + h / 2, wd, hh, col))
    # "Gas & CO-Melder" : cherché au-dessus du logo s'il n'a pas été reconnu
    lg = [dw for dw in draws if dw[2] == "Vigilair"]
    if lg and not any(dw[2] == "Détecteur gaz & CO" for dw in draws):
        lx, ly, _, lh, _ = lg[0]
        lw = lh / 0.62 * 3.3   # largeur approx. du logo
        X0, X1 = int(lx - 0.95 * lw), int(lx + 0.95 * lw)
        Y0, Y1 = int(ly - 1.15 * lw), int(ly - 0.35 * lw)
        X0, Y0 = max(0, X0), max(0, Y0); X1, Y1 = min(W, X1), min(H, Y1)
        if X1 - X0 > 10 and Y1 - Y0 > 6:
            sub = f[Y0:Y1, X0:X1].astype(int)
            lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
            m = ((lum > np.median(lum) + 14) & (sat < 55)).astype(np.uint8)
            ys, xs = np.where(m)
            if len(xs) > 8:
                rows = np.bincount(ys, minlength=m.shape[0])
                cy = int(np.argmax(np.convolve(rows, np.ones(5), "same")))
                band = (ys > cy - 6) & (ys < cy + 6)
                if band.sum() > 6:
                    bx0, bx1 = xs[band].min(), xs[band].max()
                    by0, by1 = max(0, cy - 6), min(m.shape[0], cy + 6)
                    mm = np.zeros((H, W), np.uint8)
                    mm[Y0 + by0:Y0 + by1, X0 + bx0:X0 + bx1 + 1] = m[by0:by1, bx0:bx1 + 1]
                    mask |= cv2.dilate(mm, np.ones((3, 3), np.uint8))
                    col = tuple(int(v) for v in np.median(sub[m.astype(bool)], axis=0))
                    draws.append((X0 + (bx0 + bx1) / 2, Y0 + cy, "Détecteur gaz & CO", max(4, (bx1 - bx0) * 0.11), col))
    # logo coupé en bas d'image sur les très gros plans : déduit de la position de "Gas & CO"
    gz = STATE.pop("gas", None)
    if gz and gz[2] > 110 and not any(dw[2] == "Vigilair" for dw in draws):
        x, y, w, h = gz
        ly = int(y + h / 2 + 0.44 * w); lw = int(0.62 * w)
        x0, x1 = int(x + w / 2 - lw / 2), int(x + w / 2 + lw / 2)
        y0, y1 = max(0, ly - int(0.06 * w)), min(H, ly + int(0.06 * w))
        if y0 < H - 3:
            sub = f[y0:y1, x0:x1].astype(int)
            lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
            m = ((lum > np.median(lum) + 18) & (sat < 50)).astype(np.uint8)
            if m.sum() > 4:
                col = tuple(int(v) for v in np.median(sub[m.astype(bool)], axis=0))
                mask[y0:y1, x0:x1] |= cv2.dilate(m, np.ones((3, 3), np.uint8))
                draws.append((x + w / 2, ly, "Vigilair", 0.1 * w, col))
    if mask.any():
        f = cv2.inpaint(np.ascontiguousarray(f), mask, 3, cv2.INPAINT_TELEA)
    return f, draws

def _gray_mask(f, x0, y0, x1, y1):
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 4 or y1 - y0 < 3: return None, None
    sub = f[y0:y1, x0:x1].astype(int)
    lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
    m = ((lum > np.median(lum) + 9) & (sat < 60)).astype(np.uint8)
    if m.sum() < 4: return None, None
    strong = (lum > np.median(lum) + 16) & (sat < 55)
    col = tuple(int(v) for v in np.median(sub[strong if strong.any() else m.astype(bool)], axis=0))
    full = np.zeros((H, W), np.uint8)
    if FULLRECT[0]: full[y0:y1, x0:x1] = 1
    else: full[y0:y1, x0:x1] = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return full, col
FULLRECT = [False]

def led_fix(f, t):
    g = (f[..., 1].astype(int) > 170) & (f[..., 0].astype(int) < 170) & (f[..., 1].astype(int) - f[..., 0].astype(int) > 50)
    n, lab, st, cen = cv2.connectedComponentsWithStats(g.astype(np.uint8))
    bl = [(cen[j][0], cen[j][1], st[j][4]) for j in range(1, n) if st[j][4] >= 4]
    lastled = STATE.get("led")
    if not bl:
        if lastled and t - lastled[0] < 0.8: bl = lastled[1]
        else: return f, []
    single = {0: 1.2, 1: 1.15, 2: 1.4}
    zone = 0 if t < 161 else (1 if 174 < t < 175.3 else (2 if 177.7 < t < 179.0 else None))
    if zone is not None:   # très gros plan : voyant du haut seulement
        L1 = max(bl, key=lambda b: b[2]); L2 = None; s = single[zone]
    else:
        bl2 = sorted(bl, key=lambda b: b[1])
        L1, L2 = bl2[0], bl2[-1]
        if L2[1] - L1[1] < 40:
            if lastled and t - lastled[0] < 0.8:
                bl2 = sorted(lastled[1], key=lambda b: b[1]); L1, L2 = bl2[0], bl2[-1]
            if L2[1] - L1[1] < 40: return f, []
        STATE["led"] = (t, [L1, L2])
        s = (L2[1] - L1[1]) / 140.0
    draws = []; mask = np.zeros((H, W), np.uint8)
    yc = L1[1] - 23 * s
    m, col = _gray_mask(f, L1[0] - 32 * s, yc - 13 * s, L1[0] + 198 * s, yc + 11 * s)
    if m is not None:
        mask |= m
        for wd, dx in (("Marche", 0), ("Alarme", 75), ("Défaut", 161)):
            draws.append((L1[0] + dx * s, yc, wd, 11 * s, col))
    if L2 is None and zone in (1, 2):
        gx, gy = L1[0] + 81 * s, L1[1] + 103 * s
        if gy - 12 * s < H:
            m, col = _gray_mask(f, gx - 84 * s, gy - 12 * s, gx + 84 * s, gy + 12 * s)
            if m is not None:
                mask |= m; draws.append((gx, gy, "Détecteur gaz & CO", 13 * s, col))
    if L2 is not None:
        gx, gy = L2[0] + 26 * s, L2[1] - 37 * s
        m, col = _gray_mask(f, gx - 84 * s, gy - 12 * s, gx + 84 * s, gy + 12 * s)
        if m is not None:
            mask |= m; draws.append((gx, gy, "Détecteur gaz & CO", 13 * s, col))
        lx, ly = L2[0] + 29 * s, L2[1] + 35 * s
        m, col = _gray_mask(f, lx - 58 * s, ly - 12 * s, lx + 58 * s, ly + 12 * s)
        if m is not None:
            mask |= m; draws.append((lx, ly, "Vigilair", 15 * s, col))
    if mask.any():
        f = cv2.inpaint(np.ascontiguousarray(f), mask, 3, cv2.INPAINT_TELEA)
    return f, draws

# ---------- suivi lissé des inscriptions de l'appareil ----------
WMUL = {"Vigilair": 8.0, "Détecteur gaz & CO": 13.0, "Marche": 7.5, "Alarme": 7.5, "Défaut": 7.5}

def _raw_device_unused(f, t):
    STATE.pop("gas", None)
    if 159.2 <= t <= 160.75 or 160.9 <= t <= 169.6 or 174.3 <= t <= 175.15 or 177.85 <= t <= 178.95:
        _, dr = led_fix(f, t)
    elif any(a <= t <= b for a, b in DEV_SHOTS):
        _, dr = device_fix(f)
    else:
        dr = []
    return dr

KEYS = [(149.25, 150.96, 150.1, [('gas', 179.5, 288.5, 30.5), ('logo', 179.5, 303.6, 20.9)]), (150.96, 151.29, 151.15, [('gas', 179.5, 288.5, 30.5), ('logo', 179.5, 303.6, 20.9)]), (151.29, 151.58, 151.45, [('gas', 179.5, 288.5, 30.5), ('logo', 179.5, 303.6, 20.9)]), (151.58, 151.88, 151.75, [('gas', 179.5, 288.5, 30.5), ('logo', 179.5, 303.6, 20.9)]), (153.71, 155.54, 154.6, [('gas', 216.5, 315.5, 30.5), ('logo', 216.5, 330.6, 20.9)]), (155.54, 157.17, 156.3, [('row', 278, 267.5, 72), ('gas', 276, 315, 52), ('logo', 277, 344, 36)]), (157.17, 160.25, 158.7, [('gas', 216.5, 315.5, 30.5), ('logo', 216.5, 330.6, 20.9)]), (160.25, 165.21, 162.7, [('row', 295.0, 279.0, 75.6), ('gas', 295.0, 328.3, 58.8), ('logo', 295.0, 357.4, 40.3)]), (165.21, 167.75, 166.5, [('row', 185, 445, 230), ('gas', 185, 576, 170)]), (169.29, 170.38, 169.8, [('gas', 216.0, 313.6, 29.4), ('logo', 216.0, 328.2, 20.2)]), (170.38, 170.71, 170.55, [('gas', 217.5, 315.5, 30.5), ('logo', 217.5, 330.6, 20.9)]), (170.71, 171.21, 170.95, [('row', 184, 454, 230), ('gas', 184, 587, 170)]), (172.71, 173.96, 173.3, [('row', 185, 536, 240)]), (177.54, 178.88, 178.2, [('row', 253, 310, 90), ('gas', 252, 370, 68), ('logo', 252, 407, 45)]), (188.04, 190.58, 189.3, [('row', 253, 310, 90), ('gas', 252, 370, 68), ('logo', 252, 407, 45)])]
LINEAR = []
HREL = {"row": 0.085, "gas": 0.13, "logo": 0.24}

CUTF = json.load(open("cutframes.json")) if os.path.exists("cutframes.json") else []
def snap(t):
    j = int(round(t * FPS))
    near = [c for c in CUTF if abs(c - j) <= 6]
    return min(near, key=lambda c: abs(c - j)) if near else j
def frange(t0, t1):
    return range(snap(t0), snap(t1))

def _polyfit(js, vals, deg):
    js = np.asarray(js, float); vals = np.asarray(vals, float)
    deg = min(deg, max(0, len(js) - 1))
    w = np.ones(len(js))
    for _ in range(3):
        cs = [np.polyfit(js, vals[:, c], deg, w=w) for c in range(vals.shape[1])]
        fit = np.stack([np.polyval(cc, js) for cc in cs], 1)
        err = np.abs(fit - vals).max(1)
        sc = np.median(err) * 3 + 1.0
        w = np.where(err > sc, 0.05, 1.0)
    return cs

def _track(frames, k0, cx, cy, w, h):
    g0 = frames[k0]
    x0, y0 = int(cx - w / 2 - 2), int(cy - h / 2 - 2); x1, y1 = int(cx + w / 2 + 2), int(cy + h / 2 + 2)
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
    tpl = g0[y0:y1, x0:x1]
    res = {k0: (cx, cy, 1.0)}
    for direction in (1, -1):
        pcx, pcy, ps = cx, cy, 1.0
        k = k0 + direction
        while 0 <= k < len(frames):
            g = frames[k]; best = None
            for sc in ps * np.array([0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02]):
                t = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                r = int(0.12 * w * sc + 10)
                X0, Y0 = int(max(0, pcx - t.shape[1] / 2 - r)), int(max(0, pcy - t.shape[0] / 2 - r))
                X1, Y1 = int(min(W, pcx + t.shape[1] / 2 + r)), int(min(H, pcy + t.shape[0] / 2 + r))
                win = g[Y0:Y1, X0:X1]
                if win.shape[0] <= t.shape[0] or win.shape[1] <= t.shape[1]: continue
                m = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(m)
                dx = dy = 0.0
                mx_, my_ = ml
                if 0 < mx_ < m.shape[1] - 1:
                    a_, b_, c_ = m[my_, mx_ - 1], m[my_, mx_], m[my_, mx_ + 1]; den = a_ - 2 * b_ + c_
                    if den < 0: dx = 0.5 * (a_ - c_) / den
                if 0 < my_ < m.shape[0] - 1:
                    a_, b_, c_ = m[my_ - 1, mx_], m[my_, mx_], m[my_ + 1, mx_]; den = a_ - 2 * b_ + c_
                    if den < 0: dy = 0.5 * (a_ - c_) / den
                if best is None or mv > best[0]: best = (mv, X0 + mx_ + dx + t.shape[1] / 2, Y0 + my_ + dy + t.shape[0] / 2, sc)
            if best is not None and best[0] > 0.25:
                pcx, pcy, ps = best[1], best[2], best[3]
            res[k] = (pcx, pcy, ps)
            k += direction
    return res

def _smooth(arr, a=7, b=4):
    arr = np.array(arr, dtype=float)
    med = np.array([np.median(arr[max(0, k - a):k + a + 1], axis=0) for k in range(len(arr))])
    return np.array([med[max(0, k - b):k + b + 1].mean(axis=0) for k in range(len(arr))])

def _refine(gray_list, k0, xs, ys, scs, cx0, cy0, w0, h0):
    """Recale chaque image sur le vrai texte d'origine (modèle pris sur l'image clé)."""
    g0 = gray_list[k0]
    tpl0 = g0[int(max(0, cy0 - h0 / 2)):int(cy0 + h0 / 2), int(max(0, cx0 - w0 / 2)):int(cx0 + w0 / 2)]
    if tpl0.size == 0 or tpl0.std() < 2: return xs, ys
    rx = np.zeros(len(xs)); ry = np.zeros(len(xs)); ok = np.zeros(len(xs), bool)
    for q in range(len(xs)):
        sc = scs[q]
        t = cv2.resize(tpl0, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        mg = 6 + 0.15 * t.shape[1]
        X0, Y0 = int(max(0, xs[q] - t.shape[1] / 2 - mg)), int(max(0, ys[q] - t.shape[0] / 2 - mg))
        X1, Y1 = int(min(W, xs[q] + t.shape[1] / 2 + mg)), int(min(H, ys[q] + t.shape[0] / 2 + mg))
        win = gray_list[q][Y0:Y1, X0:X1]
        if win.shape[0] <= t.shape[0] or win.shape[1] <= t.shape[1]: continue
        m = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(m)
        if mv < 0.55: continue
        rx[q] = X0 + ml[0] + t.shape[1] / 2 - xs[q]; ry[q] = Y0 + ml[1] + t.shape[0] / 2 - ys[q]; ok[q] = True
    if ok.sum() < 3: return xs, ys
    qi = np.arange(len(xs))
    rx = np.interp(qi, qi[ok], rx[ok]); ry = np.interp(qi, qi[ok], ry[ok])
    sm = _smooth(np.stack([rx, ry], 1), a=3, b=2)
    return xs + sm[:, 0], ys + sm[:, 1]

def build_devtrack():
    path = "devtrack.json"
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    allf = {}; i = 0
    want = set()
    for t0, t1, *_ in KEYS + LINEAR:
        want.update(frange(t0, t1))
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        if i in want: allf[i] = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        i += 1
    out = {}
    for t0, t1, tk, els in KEYS:
        idx = [j for j in frange(t0, t1) if j in allf]
        if not idx: continue
        gray = [cv2.cvtColor(allf[j], cv2.COLOR_RGB2GRAY).astype(np.float32) for j in idx]
        k0 = min(range(len(idx)), key=lambda q: abs(idx[q] - tk * FPS))
        # un seul gros modèle (écran + façade de l'appareil), beaucoup plus fiable que les petits textes
        bx0 = min(cx - w / 2 for _, cx, cy, w in els) - 12; bx1 = max(cx + w / 2 for _, cx, cy, w in els) + 12
        wmax = max(w for _, cx, cy, w in els)
        by0 = min(cy for _, cx, cy, w in els) - 0.9 * wmax; by1 = max(cy for _, cx, cy, w in els) + 0.15 * wmax
        bx0, by0 = max(0, bx0), max(0, by0); bx1, by1 = min(W, bx1), min(H, by1)
        BCX, BCY = (bx0 + bx1) / 2, (by0 + by1) / 2
        tr = _track(gray, k0, BCX, BCY, bx1 - bx0 - 4, by1 - by0 - 4)
        seq = np.array([tr[q] for q in range(len(idx))])
        dur = len(idx) / FPS
        sm = _smooth(seq, a=3, b=2)
        sm += np.array([BCX, BCY, 1.0]) - sm[k0]
        f0 = allf[idx[k0]]
        for name, cx, cy, w in els:
            h = max(5, HREL[name] * w * 1.4)
            sub = f0[int(max(0, cy - h / 2)):int(cy + h / 2), int(max(0, cx - w / 2)):int(cx + w / 2)].astype(int)
            col = [120, 120, 120]
            if sub.size:
                L = sub.mean(2); S_ = sub.max(2) - sub.min(2); mm = (L > np.median(L) + 12) & (S_ < 60)
                if mm.sum() > 3: col = [int(v) for v in np.median(sub[mm], axis=0)]
            XS = sm[:, 0] + (cx - BCX) * sm[:, 2]; YS = sm[:, 1] + (cy - BCY) * sm[:, 2]
            XS, YS = _refine(gray, k0, XS, YS, sm[:, 2], cx, cy, w + 6, max(6, HREL[name] * w * 1.6))
            for q, j in enumerate(idx):
                sc = sm[q, 2]
                x = XS[q]; y = YS[q]; ww = w * sc
                if name == "row":
                    for wd, rx in (("POWER", -0.38), ("ALARM", 0.0), ("FAULT", 0.38)):
                        out.setdefault(j, []).append([x + rx * ww, y, wd, HREL[name] * ww * 1.05, col, ww / 3 + 4])
                elif name == "gas":
                    out.setdefault(j, []).append([x, y, "Gas & CO Detector", HREL[name] * ww * 0.85, col, ww + 6])
                    if len([e for e in els if e[0] == "logo"]) == 0:
                        out.setdefault(j, []).append([x, y + 0.44 * ww, "", 0.13 * ww, col, 0.62 * ww])
                else:
                    out.setdefault(j, []).append([x, y, "", HREL[name] * ww * 0.9, col, ww + 6])
    for t0, t1, ta, tb, els in LINEAR:
        jl = [j for j in frange(t0, t1) if j in allf]
        gl = [cv2.cvtColor(allf[j], cv2.COLOR_RGB2GRAY).astype(np.float32) for j in jl]
        ka = min(range(len(jl)), key=lambda q: abs(jl[q] - ta * FPS))
        REF = {}
        for name, xa, ya, wa, xb, yb, wb in els:
            us = np.array([min(1, max(0, (j / FPS - ta) / (tb - ta))) for j in jl])
            XS = xa + us * (xb - xa); YS = ya + us * (yb - ya); SC = (wa + us * (wb - wa)) / wa
            REF[name] = _refine(gl, ka, XS, YS, SC, XS[ka], YS[ka], wa + 6, max(6, HREL[name] * wa * 1.6))
        for qq, j in enumerate(jl):
            u = min(1, max(0, (j / FPS - ta) / (tb - ta)))
            f0 = allf.get(j)
            for name, xa, ya, wa, xb, yb, wb in els:
                x = REF[name][0][qq]; y = REF[name][1][qq]; ww = wa + u * (wb - wa)
                col = [120, 120, 120]
                if f0 is not None:
                    h = HREL[name] * ww * 1.4
                    sub = f0[int(y - h / 2):int(y + h / 2), int(x - ww / 2):int(x + ww / 2)].astype(int)
                    if sub.size:
                        L = sub.mean(2); S_ = sub.max(2) - sub.min(2); mm = (L > np.median(L) + 12) & (S_ < 60)
                        if mm.sum() > 3: col = [int(v) for v in np.median(sub[mm], axis=0)]
                if name == "row":
                    for wd, rx in (("POWER", -0.38), ("ALARM", 0.0), ("FAULT", 0.38)):
                        out.setdefault(j, []).append([x + rx * ww, y, wd, HREL[name] * ww * 1.05, col, ww / 3 + 4])
                elif name == "gas":
                    out.setdefault(j, []).append([x, y, "Gas & CO Detector", HREL[name] * ww * 0.85, col, ww + 6])
                else:
                    out.setdefault(j, []).append([x, y, "", HREL[name] * ww * 0.9, col, ww + 6])
    json.dump(out, open(path, "w"))
    return out

DEVTRACK = None
def stable_device(f, i):
    global DEVTRACK
    if DEVTRACK is None: DEVTRACK = build_devtrack()
    els = DEVTRACK.get(i, [])
    if not els: return f, []
    mask = np.zeros((H, W), np.uint8); draws = []
    for cx, cy, wd, hh, col, wb in els:
        w = wb * 1.0; h = 1.8 * hh
        x0, x1 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2)); y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
        if x1 - x0 < 3 or y1 - y0 < 3: continue
        mask[y0:y1, x0:x1] = 1
        draws.append((cx, cy, wd, hh, tuple(col)))
    if mask.any():
        f = cv2.inpaint(np.ascontiguousarray(f), mask, 3, cv2.INPAINT_TELEA)
    return f, draws


def claimed_at(t):
    return [(sp[3] - 4, sp[4] + 4) for sp in SPECS if sp[0] <= t <= sp[1]]

def sub_boxes(f, i):
    t = i / FPS
    if any(a <= t <= b for a, b in NO_SUB): return []
    cl = claimed_at(t)
    bs = [b for b in NB[i] if not any(b[1] < c1 and b[1] + b[3] > c0 for c0, c1 in cl)]
    out = []
    for b in bs:
        x, y, w, h = german_box(f, b)
        out.append([x, y, x + w, y + h])
    return out

def german_box(f, tb):
    x, y, w, h = [int(v) for v in tb]
    wm = (f[..., 0] > 225) & (f[..., 1] > 225) & (f[..., 2] > 225)
    y0, y1 = max(0, y), min(H, y + h); x0, x1 = max(0, x), min(W, x + w)
    while x0 > 0 and wm[y0:y1, x0 - 1].mean() > 0.6: x0 -= 1
    while x1 < W and wm[y0:y1, x1].mean() > 0.6: x1 += 1
    while y0 > 0 and wm[y0 - 1, x0:x1].mean() > 0.6: y0 -= 1
    while y1 < H and wm[y1, x0:x1].mean() > 0.6: y1 += 1
    return [x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4]

def build_runs():
    if os.path.exists("runs.json"): return json.load(open("runs.json"))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    gb = []; i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        bs = sub_boxes(f, i)
        gb.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None)
        i += 1
    runs = []; cur = None
    for i, b in enumerate(gb):
        if b is None: continue
        cy = (b[1] + b[3]) / 2
        if cur and i - cur["e"] <= 6 and abs(cy - cur["cy"]) <= 14:
            cur["e"] = i; cur["boxes"].append(b)
        else:
            if cur: runs.append(cur)
            cur = {"s": i, "e": i, "cy": cy, "boxes": [b]}
    if cur: runs.append(cur)
    out = []
    for r in runs:
        bs = np.array(r["boxes"], dtype=float)
        out.append([max(0, r["s"] - 8), r["e"] + 2, [float(bs[:, 0].min()), float(np.median(bs[:, 1])), float(bs[:, 2].max()), float(np.median(bs[:, 3]))]])
    json.dump(out, open("runs.json", "w"))
    return out

RUNS = None; RUN_OF = {}; RUN_W = {}
def run_at(i):
    global RUNS
    if RUNS is None:
        RUNS = build_runs()
        for k, (s0, e0, b) in enumerate(RUNS):
            real = s0 + 8
            cuts_before = [c for c in CUTF if s0 < c <= real]
            if cuts_before: s0 = max(cuts_before)   # ne pas déborder sur le plan précédent
            for j in range(s0, e0 + 1): RUN_OF.setdefault(j, k)
    k = RUN_OF.get(i)
    return (k, RUNS[k]) if k is not None else (None, None)

def sub_draw(d, k, run, txt):
    s0, e0, b = run
    x0, y0, x1, y1 = b
    f = fontw(15 * S, 700)
    if k not in RUN_W:
        texts = set(filter(None, [chunk_at(j / FPS) or prev_chunk(j / FPS) for j in range(s0, e0 + 1)]))
        RUN_W[k] = max([d.textlength(tt, font=fontw(16.5 * S, 500)) for tt in texts] or [0])
    bw = max((x1 - x0) * S, RUN_W[k] + 22 * S)
    bw = min(bw, (W - 6) * S)
    cx = (x0 + x1) / 2 * S
    bx0 = min(max(3 * S, cx - bw / 2), W * S - 3 * S - bw); bx0 = min(bx0, x0 * S)
    bx1 = max(bx0 + bw, x1 * S)
    by0 = y0 * S - 2; by1 = y1 * S + 2
    by0 = min(by0, (y0 + y1) / 2 * S - 14 * S); by1 = max(by1, (y0 + y1) / 2 * S + 14 * S)
    rrect(d, (bx0, by0, bx1, by1), (255, 255, 255), 6 * S)
    if txt:
        f = fontw(16.5 * S, 500)
        tw = d.textlength(txt, font=f)
        if tw > (bx1 - bx0) - 16 * S: f = fontw(16.5 * S * ((bx1 - bx0) - 16 * S) / tw, 500)
        d.text(((bx0 + bx1) / 2, (by0 + by1) / 2), txt, font=f, fill=(12, 12, 12), anchor="mm")

SPEC_UNION = {}
ELEM = [  # (nom, mots, y prévu depuis le HAUT de l'écran / largeur écran, fenêtre y, largeur/écran, hauteur texte/écran, demi-fenêtre x)
    ("row",  [("Marche", -0.40), ("Alarme", 0.0), ("Défaut", 0.42)], 2.0, (1.75, 2.25), 1.2, 0.045, 0.75),
    ("gas",  [("Détecteur gaz & CO", 0.0)], 2.8, (2.5, 3.1), 0.88, 0.06, 0.40),
    ("logo", [("Vigilair", 0.0)], 3.3, (3.05, 3.65), 0.55, 0.075, 0.35),
]
def find_lcds(f):
    a = f.astype(int); lum = a.mean(2); sat = a.max(2) - a.min(2)
    m = cv2.morphologyEx(((lum > 85) & (sat < 40)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for j in range(1, n):
        x, y, w, h, ar = st[j]
        if ar > 900 and 1.3 < h / w < 2.1 and ar / (w * h) > 0.55:
            by0, by1 = min(H, y + h + int(0.1 * w)), min(H, y + h + int(1.0 * w))
            bx0, bx1 = max(0, x - int(0.1 * w)), min(W, x + w + int(0.1 * w))
            if by1 - by0 < 4 or np.median(lum[by0:by1, bx0:bx1]) < 60:   # écran posé sur un boîtier sombre
                out.append((x, y, w, h))
    return sorted(out, key=lambda b: b[0])

def _text_lines(f, lx, ly, lw, lh):
    a = f.astype(int); lum = a.mean(2); sat = a.max(2) - a.min(2)
    X0, X1 = int(max(0, lx - 0.3 * lw)), int(min(W, lx + lw + 0.3 * lw))
    Y0, Y1 = int(min(H, ly + lh + 0.03 * lw)), int(min(H, ly + lh + 2.6 * lw))
    if Y1 - Y0 < 6 or X1 - X0 < 10: return [], (X0, Y0)
    L = lum[Y0:Y1, X0:X1]; Sx = sat[Y0:Y1, X0:X1]
    body = np.median(L)
    m = ((L > body + 22) & (Sx < 50)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, max(3, int(lw * 0.06))), np.uint8))
    rows = m.sum(1)
    lines = []; y = 0
    while y < len(rows):
        if rows[y] >= 2:
            y0 = y
            while y < len(rows) and rows[y] >= 1: y += 1
            sub = m[y0:y]
            cols = np.where(sub.sum(0) > 0)[0]
            if len(cols):
                lines.append((y0 + Y0, y - 1 + Y0, cols.min() + X0, cols.max() + X0))
        y += 1
    return lines, (X0, Y0)

def _raw_device(f, t):
    if not any(a <= t <= b for a, b in DEV_SHOTS): return []
    draws = []
    for di, (lx, ly, lw, lh) in enumerate(find_lcds(f)):
        lines, _ = _text_lines(f, lx, ly, lw, lh)
        good = [l for l in lines if 2 <= l[1] - l[0] + 1 <= 0.2 * lw and (l[3] - l[2]) >= 0.3 * lw]
        if not good: continue
        a_ = f.astype(int); lum = a_.mean(2); sat = a_.max(2) - a_.min(2)
        def colof(l):
            sub = f[l[0]:l[1] + 1, l[2]:l[3] + 1].astype(int); L = sub.mean(2); S_ = sub.max(2) - sub.min(2)
            mm = (L > np.median(L) + 15) & (S_ < 55)
            return tuple(int(v) for v in np.median(sub[mm], axis=0)) if mm.sum() > 2 else (120, 120, 120)
        row = next((l for l in good if (l[3] - l[2]) > 0.95 * lw), None)
        rest = [l for l in good if l is not row and (row is None or l[0] > row[1])]
        gas = next((l for l in rest if 0.62 * lw < (l[3] - l[2]) <= 1.05 * lw), None)
        logo = None
        cand = [l for l in rest if l is not gas and 0.32 * lw < (l[3] - l[2]) <= 0.75 * lw and (gas is None or l[0] > gas[1])]
        if cand: logo = cand[-1]
        for name, l in (("row", row), ("gas", gas), ("logo", logo)):
            if l is None: continue
            cx = (l[2] + l[3]) / 2; cy = (l[0] + l[1]) / 2; wl = l[3] - l[2] + 1; hl = l[1] - l[0] + 1
            col = colof(l)
            if name == "row":
                for wd, rx in (("Marche", 0.13), ("Alarme", 0.5), ("Défaut", 0.87)):
                    draws.append((l[2] + rx * wl, cy, wd, hl * 1.15) + col + (f"{wd}#{di}", wl / 3 + 6))
            elif name == "gas":
                draws.append((cx, cy, "Détecteur gaz & CO", hl * 1.15) + col + (f"gas#{di}", wl + 8))
            else:
                draws.append((cx, cy, "Vigilair", hl * 1.25) + col + (f"logo#{di}", wl + 8))
    return draws

# ---------- appareil : correction collée aux pixels (alignement ECC affine) ----------
def _shots():
    out = []
    for t0, t1, tk, els in KEYS:
        out.append((t0, t1, tk, [(n, cx, cy, w) for n, cx, cy, w in els]))
    for t0, t1, ta, tb, els in LINEAR:
        out.append((t0, t1, ta, [(n, xa, ya, wa) for n, xa, ya, wa, xb, yb, wb in els]))
    return out

def _read_frames(want):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    fr = {}; i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        if i in want: fr[i] = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        i += 1
    return fr

def _elem_rects(els):
    R = []
    for name, cx, cy, w in els:
        if name == "row":
            hh = HREL[name] * w * 1.05; R.append((name, cx, cy, w + 24, 2.4 * hh, hh))
        elif name == "gas":
            hh = HREL[name] * w * 0.85; R.append((name, cx, cy, w + 8, 1.9 * hh, hh))
        else:
            hh = HREL[name] * w * 0.9; R.append((name, cx, cy, w + 8, max(1.7 * hh, 8), hh))
    return R

def _ncc(K, F, M, rects):
    wk = cv2.warpAffine(K, M, (W, H)); out = []
    for cx, cy, w, h in rects:
        X, Y = M @ np.array([cx, cy, 1.0], np.float32)
        x0, x1, y0, y1 = int(X - w / 2), int(X + w / 2), int(Y - h / 2), int(Y + h / 2)
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < 6 or y1 - y0 < 4: continue
        A = wk[y0:y1, x0:x1]; B = F[y0:y1, x0:x1]
        A = A - A.mean(); B = B - B.mean()
        out.append(float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-6)))
    return float(np.mean(out)) if out else -1

AFF = None; DELTA = {}
def build_affine():
    global AFF
    if os.path.exists("affine.npz"):
        z = np.load("affine.npz", allow_pickle=True); AFF = z["aff"].item()
        for k, v in z["delta"].item().items(): DELTA[k] = v
        return
    shots = _shots(); want = set()
    for t0, t1, tk, els in shots: want.update(frange(t0, t1))
    fr = _read_frames(want)
    AFF = {}
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    for si, (t0, t1, tk, els) in enumerate(shots):
        idx = [j for j in frange(t0, t1) if j in fr]
        if not idx: continue
        k0 = min(range(len(idx)), key=lambda q: abs(idx[q] - tk * FPS))
        K = fr[idx[k0]]
        G = {j: cv2.GaussianBlur(cv2.cvtColor(fr[j], cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0) for j in idx}
        wmax = max(w for _, _, _, w in els)
        x0 = int(max(0, min(cx - w / 2 for _, cx, cy, w in els) - 15)); x1 = int(min(W, max(cx + w / 2 for _, cx, cy, w in els) + 15))
        y0 = int(max(0, min(cy for _, cx, cy, w in els) - 1.0 * wmax)); y1 = int(min(H, max(cy for _, cx, cy, w in els) + 0.25 * wmax))
        mask = np.zeros((H, W), np.uint8); mask[y0:y1, x0:x1] = 1
        G8 = {j: cv2.cvtColor(fr[j], cv2.COLOR_RGB2GRAY) for j in idx}
        Gf = {j: cv2.GaussianBlur(G8[j].astype(np.float32), (3, 3), 0) for j in idx}
        Kf = Gf[idx[k0]]; roi = [(cx, cy, w, max(8, HREL[n] * w * 1.8)) for n, cx, cy, w in els]
        mk8 = (mask * 255).astype(np.uint8)
        pts0 = cv2.goodFeaturesToTrack(G8[idx[k0]], 400, 0.01, 4, mask=mk8)
        A = {idx[k0]: np.eye(2, 3, dtype=np.float32)}
        for rng in (range(k0 + 1, len(idx)), range(k0 - 1, -1, -1)):
            pp = pts0.copy(); pj = idx[k0]; good = np.ones(len(pp), bool); prevM = np.eye(2, 3, dtype=np.float32)
            for q in rng:
                j = idx[q]
                nx, st, err = cv2.calcOpticalFlowPyrLK(G8[pj], G8[j], pp, None, winSize=(21, 21), maxLevel=3)
                good &= st.ravel() == 1; pp = nx; pj = j
                M = None
                if good.sum() >= 6:
                    M, inl = cv2.estimateAffinePartial2D(pts0[good], pp[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                if M is None: M = prevM
                M = M.astype(np.float32)
                # candidat 2 : affinage ECC initialisé par le flux optique ; on garde le meilleur
                best = (_ncc(Kf, Gf[j], M, roi), M)
                try:
                    cc, M2 = cv2.findTransformECC(Kf, Gf[j], M.copy(), cv2.MOTION_AFFINE, crit, mask, 5)
                    sc2 = _ncc(Kf, Gf[j], M2, roi)
                    if sc2 > best[0]: best = (sc2, M2)
                except cv2.error:
                    pass
                A[j] = best[1]; prevM = A[j]
        # lissage léger des paramètres pour supprimer le bruit de l'estimation
        js = sorted(A); P = np.array([A[j].reshape(-1) for j in js])
        Ps = _smooth(P, a=1, b=1)
        for q, j in enumerate(js): AFF[j] = (si, Ps[q].reshape(2, 3).astype(np.float32))
        # correction calculée une seule fois sur l'image clé (en résolution x2)
        K2 = np.array(Image.fromarray(K).resize((W * S, H * S), Image.LANCZOS))
        mk = np.zeros((H * S, W * S), np.uint8)
        R = _elem_rects(els)
        for name, cx, cy, ww, hh_, th in R:
            mk[int((cy - hh_ / 2) * S):int((cy + hh_ / 2) * S), int((cx - ww / 2) * S):int((cx + ww / 2) * S)] = 1
        Kc = cv2.inpaint(K2, mk, 5, cv2.INPAINT_TELEA)
        pim = Image.fromarray(Kc); dr = ImageDraw.Draw(pim); COLS = {}
        for name, cx, cy, ww, hh_, th in R:
            sub = K[int(cy - hh_ / 2):int(cy + hh_ / 2), int(cx - ww / 2):int(cx + ww / 2)].astype(int)
            col = (120, 120, 120)
            if sub.size:
                L = sub.mean(2); S_ = sub.max(2) - sub.min(2); mm = (L > np.median(L) + 12) & (S_ < 60)
                if mm.sum() > 3: col = tuple(int(v) for v in np.median(sub[mm], axis=0))
            COLS[name] = col
            wtxt = ww - 8
            if name == "row":
                for wd, rx in (("POWER", -0.38), ("ALARM", 0.0), ("FAULT", 0.38)):
                    dr.text(((cx + rx * wtxt) * S, cy * S), wd, font=fontw(th * S, 500), fill=col, anchor="mm")
            elif name == "gas":
                dr.text((cx * S, cy * S), "Gas & CO Detector", font=fontw(th * S, 500), fill=col, anchor="mm")
        Kc = np.array(pim)
        tm = (np.abs(Kc.astype(int) - cv2.inpaint(K2, mk, 5, cv2.INPAINT_TELEA).astype(int)).sum(2) > 6).astype(np.uint8)
        mk = mk | cv2.dilate(tm, np.ones((7, 7), np.uint8))
        Kc2 = cv2.inpaint(K2, mk, 5, cv2.INPAINT_TELEA)
        pim = Image.fromarray(Kc2); dr = ImageDraw.Draw(pim)
        for name, cx, cy, ww, hh_, th in R:
            col = COLS.get(name, (120, 120, 120)); wtxt = ww - 8
            if name == "row":
                for wd, rx in (("POWER", -0.38), ("ALARM", 0.0), ("FAULT", 0.38)):
                    dr.text(((cx + rx * wtxt) * S, cy * S), wd, font=fontw(th * S, 500), fill=col, anchor="mm")
            elif name == "gas":
                dr.text((cx * S, cy * S), "Gas & CO Detector", font=fontw(th * S, 500), fill=col, anchor="mm")
        Kc = np.array(pim)
        feather = cv2.GaussianBlur(cv2.dilate(mk, np.ones((5, 5), np.uint8)).astype(np.float32), (5, 5), 0)
        ys_, xs_ = np.where(feather > 0.01)
        by0, by1, bx0, bx1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
        DELTA[si] = (Kc[by0:by1, bx0:bx1].astype(np.uint8), feather[by0:by1, bx0:bx1].astype(np.float16), int(bx0), int(by0))
    np.savez_compressed("affine.npz", aff=np.array(AFF, dtype=object), delta=np.array(DELTA, dtype=object))

def apply_device(im, i):
    if AFF is None: build_affine()
    a = AFF.get(i)
    if a is None: return im
    si, M = a
    P, F, ox, oy = DELTA[si]
    M2 = M.copy(); M2[:, 2] *= S
    # le patch est en coordonnées (ox, oy) de l'image clé x2
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float32)
    M3 = (np.vstack([M2, [0, 0, 1]]) @ T)[:2]
    pw = cv2.warpAffine(P, M3, (W * S, H * S), flags=cv2.INTER_LINEAR, borderValue=0).astype(np.float32)
    mw = cv2.warpAffine(F.astype(np.float32), M3, (W * S, H * S), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    arr = np.asarray(im).astype(np.float32)
    arr = arr * (1 - mw) + pw * mw
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

SPEC_INFO = {}
def precompute_specs():
    if os.path.exists("specunion.json"):
        SPEC_INFO.update({int(k): v for k, v in json.load(open("specunion.json")).items()}); return
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        t = i / FPS
        act = [(k, sp) for k, sp in enumerate(SPECS) if sp[0] <= t <= sp[1] and sp[2] != "plain"]
        if act:
            f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            for k, sp in act:
                b = spec_box(sp, f)
                if b is None: continue
                b = [float(v) for v in b]
                inf = SPEC_INFO.get(k)
                area = (b[2] - b[0]) * (b[3] - b[1])
                if inf is None:
                    inf = {"box": b, "first": i, "last": i, "area": 0, "col": None}
                else:
                    u = inf["box"]; inf["box"] = [min(b[0], u[0]), min(b[1], u[1]), max(b[2], u[2]), max(b[3], u[3])]
                    inf["last"] = i
                if area > inf["area"]:
                    c = median_color(sp[2], f, b); inf["area"] = area; inf["col"] = list(c) if c else None
                SPEC_INFO[k] = inf
        i += 1
    json.dump(SPEC_INFO, open("specunion.json", "w"))

def process(f, i):
    t = i / FPS
    draws = []
    draws = []
    plains = []
    for sp in SPECS:
        if sp[2] == "plain" and sp[0] <= t <= sp[1]:
            y0, y1, x0p, x1p = sp[3], sp[4], sp[5], sp[6]
            wm = np.zeros((y1 - y0, W), np.uint8); wm[:, x0p:x1p] = cmask("plain", f[y0:y1, x0p:x1p])
            if wm.sum() > 20:
                mk = np.zeros((H, W), np.uint8); mk[y0:y1] = cv2.dilate(wm, np.ones((5, 5), np.uint8))
                f = cv2.inpaint(np.ascontiguousarray(f), mk, 4, cv2.INPAINT_TELEA)
                xs = np.where(wm.sum(0) > 0)[0]
                plains.append(((xs.min() + xs.max()) / 2, (y0 + y1) / 2, (y1 - y0) * 0.62, sp[7]))
    im = Image.fromarray(f).resize((W * S, H * S), Image.LANCZOS)
    im = apply_device(im, i)
    d = ImageDraw.Draw(im)
    for px, py, ph, ptxt in plains:
        ff = fontw(ph * S, 800)
        d.text((px * S, py * S), ptxt, font=ff, fill=(255, 255, 255), anchor="mm", stroke_width=int(1.5 * S), stroke_fill=(20, 20, 40))
    for cx, cy, wd, hh, col in draws:
        if wd: d.text((cx * S, cy * S), wd, font=fontw(hh * S, 500), fill=tuple(col), anchor="mm")
    for k_, sp in enumerate(SPECS):
        if not (sp[0] <= t <= sp[1]) or sp[2] == "plain": continue
        if not SPEC_INFO: precompute_specs()
        inf = SPEC_INFO.get(k_)
        if inf is None or not (inf["first"] - 1 <= i <= inf["last"]): continue
        box = inf["box"]; col = tuple(inf["col"]) if inf["col"] else NAVY
        draw_label(d, box, col, sp[7], big=(sp[2] == "whitebg" or sp[7] == "25 000"))
    txt = chunk_at(t)
    k, run = run_at(i)
    for b in sub_boxes(f, i):   # sous-titres allemands secondaires : recouverts aussi
        if run is not None:
            rb = run[2]
            if not (b[3] < rb[1] - 4 or b[1] > rb[3] + 4): continue
        rrect(d, (b[0] * S - 2, b[1] * S - 2, b[2] * S + 2, b[3] * S + 2), (255, 255, 255), 6 * S)
    if run is not None:
        sub_draw(d, k, run, txt or prev_chunk(t) or "")
        STATE["run"] = (k, run); STATE["bt"] = t
    elif txt and STATE.get("run") and t - STATE["bt"] < 1.0 and not any(a <= t <= b for a, b in NO_SUB) and not any(c0 - 10 <= (STATE["run"][1][2][1] + STATE["run"][1][2][3]) / 2 <= c1 + 10 for c0, c1 in claimed_at(t)):
        sub_draw(d, *STATE["run"], txt)
    return im

def main():
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", os.environ.get("SRC", "src.mp4"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W*S}x{H*S}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "video_fr_noaudio.mp4"], stdin=subprocess.PIPE)
    i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        enc.stdin.write(process(np.frombuffer(buf, np.uint8).reshape(H, W, 3), i).tobytes())
        i += 1
        if i % 1000 == 0: print(i, flush=True)
    enc.stdin.close(); enc.wait(); print("done", i)

if __name__ == "__main__":
    main()
