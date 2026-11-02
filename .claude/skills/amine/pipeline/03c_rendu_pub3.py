"""Pub 3 : sous-titres bleu marine + étiquettes recouverts en français, appareil renommé Vigilair."""
import json, subprocess, re, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 360, 640, 2, 30
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
 (87.8, 88.8, "red",   40, 82, 0, 360, "JUSQU'À 70 PPM"),
 (87.8, 88.8, "plain", 484, 514, 0, 360, "TOTALEMENT MUET"),
 (105.1, 108.6, "red", 40, 82, 0, 360, "ENFANTS"),
 (105.1, 108.6, "plain", 480, 512, 0, 360, "30 À 50 PPM"),
 (140.0, 142.4, "navy", 112, 145, 0, 360, "NORME LÉGALE"),
 (140.0, 142.4, "red", 378, 412, 0, 360, "4 HEURES DE SILENCE"),
 (186.9, 188.4, "green", 38, 76, 0, 360, "ALARME DÈS 30 PPM"),
 (215.4, 218.15, "red",   272, 308, 0, 360, "MONOXYDE DE CARBONE"),
 (215.4, 218.15, "green", 324, 360, 0, 360, "GAZ NATUREL"),
 (215.4, 218.15, "green", 376, 412, 0, 360, "PROPANE"),
 (215.4, 218.15, "green", 427, 463, 0, 360, "GAZ INFLAMMABLES"),
 (242.6, 248.8, "navy",  120, 158, 0, 360, "PLUS DE"),
 (242.6, 248.8, "navy",  162, 224, 0, 360, "25 000"),
 (242.6, 248.8, "navy",  258, 298, 0, 360, "FAMILLES"),
 (248.6, 253.2, "red",    88, 142, 0, 360, "MOINS 60 %"),
 (248.6, 253.2, "navy",  460, 505, 0, 360, "SUR 3 DÉTECTEURS"),
 (257.9, 264.0, "circle", 272, 306, 120, 240, "JOURS"),
 (257.9, 264.0, "navy",  375, 412, 0, 360, "SATISFAIT OU REMBOURSÉ"),
 (280.2, 285.0, "whitebg", 112, 166, 20, 340, "VIGILAIR"),
 (280.2, 285.0, "red",   436, 474, 0, 360, "MOINS 60 % SUR 3 DÉTECTEURS"),
 (280.2, 285.0, "blue",  490, 528, 0, 360, "LIEN EN BAS"),
]
NO_SUB = [(140.0, 142.4), (242.6, 253.2), (257.9, 264.0), (280.2, 285.0)]
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
seq = json.load(open("words.json"))
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

NB = json.load(open("navyboxes.json"))
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
DEV_SHOTS = [(175.13, 179.23), (213.7, 220.4), (226.3, 236.1), (248.7, 253.1), (280.3, 284.9)]

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

KEYS = [  # (début, fin, t_clé, [(élément, cx, cy, largeur)])  — positions relevées à la main
    (213.7, 215.6, 214.5, [("row", 182, 378, 105), ("gas", 182, 453, 85), ("logo", 182, 493, 60)]),
    (215.6, 218.1, 216.8, [("row", 184, 165, 67), ("gas", 177, 209.5, 47), ("logo", 179, 234.5, 31)]),
    (218.1, 220.4, 219.3, [("row", 186, 452, 228), ("gas", 183, 581, 165)]),
    (175.13, 177.0, 175.6, [("row", 165, 335, 140), ("gas", 165, 404, 100), ("logo", 165, 445, 65)]),
    (226.4, 230.7, 228.5, [("row", 180, 367, 72), ("gas", 180, 406, 54), ("logo", 180, 427, 36)]),
    (230.7, 232.17, 231.0, [("row", 178, 357, 55), ("gas", 182, 395, 46), ("logo", 184, 420, 33)]),
    (233.6, 236.1, 235.0, [("row", 182, 315, 93), ("gas", 179, 385, 68), ("logo", 180, 426, 49)]),
    (248.7, 253.1, 251.0, [("row", 90, 350, 45), ("gas", 93, 384, 38), ("logo", 91, 406, 28),
                           ("row", 182, 350, 60), ("gas", 179, 392, 47), ("logo", 180, 416, 30),
                           ("row", 260, 350, 40), ("gas", 261, 384, 38), ("logo", 264, 406, 24)]),
    (280.3, 284.9, 282.5, [("row", 179, 307, 77), ("gas", 178, 362, 58), ("logo", 180, 392, 40)]),
]
LINEAR = [  # plan en zoom : interpolation linéaire entre deux relevés manuels
    (232.17, 233.62, 232.25, 233.55,
     [("row", 180, 330, 90, 180, 346, 95), ("gas", 180, 393, 68, 180, 411, 75), ("logo", 180, 431, 48, 180, 450, 55)]),
    (177.0, 179.23, 177.05, 178.9,
     [("row", 186, 440, 222, 185, 465, 235), ("gas", 186, 565, 162, 185, 597, 170), ("logo", 185, 637, 110, 185, 668, 115)]),
]
HREL = {"row": 0.085, "gas": 0.13, "logo": 0.24}

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
            for sc in ps * np.array([0.94, 0.97, 1.0, 1.03, 1.06]):
                t = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
                r = int(0.35 * w * sc + 12)
                X0, Y0 = int(max(0, pcx - t.shape[1] / 2 - r)), int(max(0, pcy - t.shape[0] / 2 - r))
                X1, Y1 = int(min(W, pcx + t.shape[1] / 2 + r)), int(min(H, pcy + t.shape[0] / 2 + r))
                win = g[Y0:Y1, X0:X1]
                if win.shape[0] <= t.shape[0] or win.shape[1] <= t.shape[1]: continue
                m = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(m)
                if best is None or mv > best[0]: best = (mv, X0 + ml[0] + t.shape[1] / 2, Y0 + ml[1] + t.shape[0] / 2, sc)
            if best is not None and best[0] > 0.35:
                pcx, pcy, ps = best[1], best[2], best[3]
            res[k] = (pcx, pcy, ps)
            k += direction
    return res

def _smooth(arr, a=7, b=4):
    arr = np.array(arr, dtype=float)
    med = np.array([np.median(arr[max(0, k - a):k + a + 1], axis=0) for k in range(len(arr))])
    return np.array([med[max(0, k - b):k + b + 1].mean(axis=0) for k in range(len(arr))])

def build_devtrack():
    path = "devtrack.json"
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    allf = {}; i = 0
    want = set()
    for t0, t1, *_ in KEYS + LINEAR:
        want.update(range(int(t0 * FPS), int(t1 * FPS) + 1))
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        if i in want: allf[i] = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        i += 1
    out = {}
    for t0, t1, tk, els in KEYS:
        idx = [j for j in range(int(t0 * FPS) + 1, int(t1 * FPS)) if j in allf]
        if not idx: continue
        gray = [cv2.cvtColor(allf[j], cv2.COLOR_RGB2GRAY).astype(np.float32) for j in idx]
        k0 = min(range(len(idx)), key=lambda q: abs(idx[q] - tk * FPS))
        for ei, (name, cx, cy, w) in enumerate(els):
            h = max(5, HREL[name] * w * 1.4)
            tr = _track(gray, k0, cx, cy, w, h)
            seq = [tr[q] for q in range(len(idx))]
            sm = _smooth(seq)
            # couleur du texte : médiane sur l'image clé
            f0 = allf[idx[k0]]; sub = f0[int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)].astype(int)
            L = sub.mean(2); S_ = sub.max(2) - sub.min(2); mm = (L > np.median(L) + 12) & (S_ < 60)
            col = [int(v) for v in np.median(sub[mm], axis=0)] if mm.sum() > 3 else [120, 120, 120]
            for q, j in enumerate(idx):
                x, y, sc = sm[q]
                ww = w * sc
                if name == "row":
                    for wd, rx in (("Marche", -0.38), ("Alarme", 0.0), ("Défaut", 0.38)):
                        out.setdefault(j, []).append([x + rx * ww, y, wd, HREL[name] * ww * 1.05, col, ww / 3 + 4])
                elif name == "gas":
                    out.setdefault(j, []).append([x, y, "Détecteur gaz & CO", HREL[name] * ww * 0.85, col, ww + 6])
                    if len([e for e in els if e[0] == "logo"]) == 0:   # logo hors champ sur le gros plan : déduit du texte gaz
                        out.setdefault(j, []).append([x, y + 0.44 * ww, "Vigilair", 0.13 * ww, col, 0.62 * ww])
                else:
                    out.setdefault(j, []).append([x, y, "Vigilair", HREL[name] * ww * 0.9, col, ww + 6])
    for t0, t1, ta, tb, els in LINEAR:
        for j in range(int(t0 * FPS) + 1, int(t1 * FPS) + 1):
            u = min(1, max(0, (j / FPS - ta) / (tb - ta)))
            f0 = allf.get(j)
            for name, xa, ya, wa, xb, yb, wb in els:
                x = xa + u * (xb - xa); y = ya + u * (yb - ya); ww = wa + u * (wb - wa)
                col = [120, 120, 120]
                if f0 is not None:
                    h = HREL[name] * ww * 1.4
                    sub = f0[int(y - h / 2):int(y + h / 2), int(x - ww / 2):int(x + ww / 2)].astype(int)
                    if sub.size:
                        L = sub.mean(2); S_ = sub.max(2) - sub.min(2); mm = (L > np.median(L) + 12) & (S_ < 60)
                        if mm.sum() > 3: col = [int(v) for v in np.median(sub[mm], axis=0)]
                if name == "row":
                    for wd, rx in (("Marche", -0.38), ("Alarme", 0.0), ("Défaut", 0.38)):
                        out.setdefault(j, []).append([x + rx * ww, y, wd, HREL[name] * ww * 1.05, col, ww / 3 + 4])
                elif name == "gas":
                    out.setdefault(j, []).append([x, y, "Détecteur gaz & CO", HREL[name] * ww * 0.85, col, ww + 6])
                else:
                    out.setdefault(j, []).append([x, y, "Vigilair", HREL[name] * ww * 0.9, col, ww + 6])
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
        w = wb * 1.15; h = 1.9 * hh
        x0, x1 = int(max(0, cx - w / 2)), int(min(W, cx + w / 2)); y0, y1 = int(max(0, cy - h / 2)), int(min(H, cy + h / 2))
        if x1 - x0 < 3 or y1 - y0 < 3: continue
        sub = f[y0:y1, x0:x1].astype(int)
        lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
        m = ((lum > np.median(lum) + 8) & (sat < 60)).astype(np.uint8)
        mask[y0:y1, x0:x1] |= cv2.dilate(m, np.ones((5, 5), np.uint8))
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
    return [extend([b[0] - 3, b[1] - 6, b[0] + b[2] + 3, b[1] + b[3] + 5], f) for b in bs]

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
            for j in range(s0, e0 + 1): RUN_OF.setdefault(j, k)
    k = RUN_OF.get(i)
    return (k, RUNS[k]) if k is not None else (None, None)

def sub_draw(d, k, run, txt):
    s0, e0, b = run
    x0, y0, x1, y1 = b
    f = fontw(15 * S, 700)
    if k not in RUN_W:
        texts = set(filter(None, [chunk_at(j / FPS) or prev_chunk(j / FPS) for j in range(s0, e0 + 1)]))
        RUN_W[k] = max([d.textlength(tt, font=f) for tt in texts] or [0])
    bw = max((x1 - x0) * S, RUN_W[k] + 26 * S)
    bw = min(bw, (W - 6) * S)
    cx = (x0 + x1) / 2 * S
    bx0 = min(max(3 * S, cx - bw / 2), W * S - 3 * S - bw); bx0 = min(bx0, x0 * S)
    bx1 = max(bx0 + bw, x1 * S)
    by0 = y0 * S - 2; by1 = y1 * S + 2
    rrect(d, (bx0, by0, bx1, by1), NAVY, min(9 * S, (by1 - by0) * 0.28))
    if txt:
        tw = d.textlength(txt, font=f)
        if tw > (bx1 - bx0) - 14 * S: f = fontw(15 * S * ((bx1 - bx0) - 14 * S) / tw, 700)
        d.text(((bx0 + bx1) / 2, (by0 + by1) / 2), txt, font=f, fill=(255, 255, 255), anchor="mm")

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

def process(f, i):
    t = i / FPS
    draws = []
    if any(a <= t <= b for a, b in DEV_SHOTS):
        f, draws = stable_device(f, i)
    plains = []
    for sp in SPECS:
        if sp[2] == "plain" and sp[0] <= t <= sp[1]:
            y0, y1 = sp[3], sp[4]
            wm = cmask("plain", f[y0:y1]).astype(np.uint8)
            if wm.sum() > 20:
                mk = np.zeros((H, W), np.uint8); mk[y0:y1] = cv2.dilate(wm, np.ones((5, 5), np.uint8))
                f = cv2.inpaint(np.ascontiguousarray(f), mk, 4, cv2.INPAINT_TELEA)
                xs = np.where(wm.sum(0) > 0)[0]
                plains.append(((xs.min() + xs.max()) / 2, (y0 + y1) / 2, (y1 - y0) * 0.62, sp[7]))
    im = Image.fromarray(f).resize((W * S, H * S), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for px, py, ph, ptxt in plains:
        ff = fontw(ph * S, 800)
        d.text((px * S, py * S), ptxt, font=ff, fill=(255, 255, 255), anchor="mm", stroke_width=int(1.5 * S), stroke_fill=(20, 20, 40))
    for cx, cy, wd, hh, col in draws:
        d.text((cx * S, cy * S), wd, font=fontw(hh * S, 500), fill=tuple(col), anchor="mm")
    for k_, sp in enumerate(SPECS):
        if not (sp[0] <= t <= sp[1]) or sp[2] == "plain": continue
        box = spec_box(sp, f)
        if box is None:
            box = SPEC_UNION.get(k_)
            if box is None: continue
        else:
            u = SPEC_UNION.get(k_)
            box = box if u is None else [min(box[0], u[0]), min(box[1], u[1]), max(box[2], u[2]), max(box[3], u[3])]
            SPEC_UNION[k_] = box
        col = median_color(sp[2], f, box) or NAVY
        draw_label(d, box, col, sp[7], big=(sp[2] == "whitebg" or sp[7] == "25 000"))
    txt = chunk_at(t)
    k, run = run_at(i)
    if run is not None:
        sub_draw(d, k, run, txt or prev_chunk(t) or "")
        STATE["run"] = (k, run); STATE["bt"] = t
    elif txt and STATE.get("run") and t - STATE["bt"] < 1.0 and not any(a <= t <= b for a, b in NO_SUB):
        sub_draw(d, *STATE["run"], txt)
    return im

def main():
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
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
