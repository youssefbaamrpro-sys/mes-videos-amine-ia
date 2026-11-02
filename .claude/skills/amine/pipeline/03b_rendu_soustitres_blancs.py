"""Pub 2 : sous-titres blancs (texte noir) et étiquettes allemandes recouverts en français."""
import json, subprocess, re, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, S, FPS = 360, 640, 2, 24
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts") + "/"
_fc = {}
def font(px, w=500):
    key = (max(6, int(px)), w)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(FD + f"Poppins-{w}.ttf", key[0])
    return _fc[key]

# ---------------- sous-titres FR ----------------
seq = json.load(open(os.environ.get("WORDS", "words_new.json")))
chunks = []; cur = []
for w in seq:
    if cur and ((w["w"][:1].isupper() and w["w"] not in ("Vigilair", "Vigilair,", "CO", "Vigilair.")) or w.get("blk") != cur[-1].get("blk")):
        chunks.append(cur); cur = []
    cur.append(w)
    txt = " ".join(x["w"] for x in cur)
    nxt_i = seq.index(w) + 1
    nxt_w = seq[nxt_i]["w"] if nxt_i < len(seq) else ""
    glued = w["w"].isdigit() and nxt_w[:1].isdigit()
    if not glued and (len(cur) >= 4 or len(txt) >= 20 or re.search(r"[,.;:?!…]$", w["w"])):
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
merged = []
for c in chunks:
    if merged and len(" ".join(x["w"] for x in c)) <= 3 and c[0].get("blk") == merged[-1][-1].get("blk"):
        merged[-1] = merged[-1] + c
    else:
        merged.append(c)
chunks = merged
CH = []
for i, c in enumerate(chunks):
    s = c[0]["s"]; e = c[-1]["e"]
    nxt = chunks[i + 1][0]["s"] if i + 1 < len(chunks) else 1e9
    CH.append((s, nxt if nxt - e < 2.5 else e + 0.4, " ".join(x["w"] for x in c)))
def chunk_at(t):
    for s, e, txt in CH:
        if s <= t < e: return txt
    return None
def prev_chunk(t):
    pc = [c for c in CH if c[0] <= t]
    return pc[-1][2] if pc else None

WB = json.load(open("wboxes.json"))

# ---------------- étiquettes ----------------
CLAIM = [  # (t0, t1, y0, y1) zones qui ne sont pas des sous-titres
    (130.7, 135.4, 110, 145), (160.8, 165.6, 420, 535), (189.6, 192.0, 488, 526),
    (208.0, 213.4, 55, 168), (208.0, 213.4, 488, 526)]
PILLS = [(428, "MONOXYDE DE CARBONE"), (457, "GAZ NATUREL"), (487, "PROPANE"), (516, "GAZ INFLAMMABLES")]

def rr(d, box, fill, rad, outline=None, width=0):
    d.rounded_rectangle([int(v) for v in box], radius=int(rad), fill=fill, outline=outline, width=width)

def text_center(d, cx, cy, txt, f, fill):
    d.text((cx, cy), txt, font=f, fill=fill, anchor="mm")

def german_box(f, tb):
    """Étend la boîte de texte allemand jusqu'aux bords de son cadre blanc."""
    x, y, w, h = [int(v) for v in tb]
    wm = (f[..., 0] > 225) & (f[..., 1] > 225) & (f[..., 2] > 225)
    y0, y1 = max(0, y), min(H, y + h)
    x0, x1 = max(0, x), min(W, x + w)
    while x0 > 0 and wm[y0:y1, x0 - 1].mean() > 0.6: x0 -= 1
    while x1 < W and wm[y0:y1, x1].mean() > 0.6: x1 += 1
    while y0 > 0 and wm[y0 - 1, x0:x1].mean() > 0.6: y0 -= 1
    while y1 < H and wm[y1, x0:x1].mean() > 0.6: y1 += 1
    return [x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4]

def cover_box(d, gb, txt):
    """Cadre blanc FR >= cadre allemand complet, texte noir centré."""
    x, y, w, h = gb
    f = font(16.5 * S, 500)
    tw = d.textlength(txt, font=f) if txt else 0
    if tw > (W - 24) * S:
        f = font(16.5 * S * (W - 24) * S / tw, 500); tw = (W - 24) * S
    cx = (x + w / 2) * S; cy = (y + h / 2) * S
    bw = max(w * S, tw + 20 * S); bh = max(h * S, 28 * S)
    x0 = min(max(3 * S, cx - bw / 2), W * S - 3 * S - bw)
    x0 = min(x0, x * S); x1 = max(x0 + bw, (x + w) * S)
    box = (x0, cy - bh / 2, x1, cy + bh / 2)
    rr(d, box, (255, 255, 255), 6 * S)
    if txt: text_center(d, (box[0] + box[2]) / 2, cy, txt, f, (12, 12, 12))

def white_box_text(d, tb, txt, weight=500, fsize=None, pad=(8, 6), minbox=None):
    """tb = boîte de texte allemand (x, y, w, h) en coords source."""
    x, y, w, h = tb
    fs = (fsize or h * 1.05) * S
    f = font(fs, weight)
    tw = d.textlength(txt, font=f)
    maxw = (W - 10) * S - 2 * pad[0] * S
    if tw > maxw:
        f = font(fs * maxw / tw, weight); tw = maxw
    cx = (x + w / 2) * S; cy = (y + h / 2) * S
    bw = max(tw + 2 * pad[0] * S, (w + 2 * pad[0]) * S)
    bh = max(f.size * 1.25 + pad[1] * S, (h + 2 * pad[1]) * S)
    x0 = min(max(4 * S, cx - bw / 2), W * S - 4 * S - bw)
    box = (x0, cy - bh / 2, x0 + bw, cy + bh / 2)
    rr(d, box, (255, 255, 255), 5 * S)
    text_center(d, (box[0] + box[2]) / 2, cy, txt, f, (12, 12, 12))

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
DEV_WIN = (150.7, 213.4)
DEV_SHOTS = [(150.7, 170.6), (174.2, 175.3), (177.7, 179.1), (183.8, 186.5), (189.6, 192.0), (208.0, 213.4)]

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
WMUL = {"Vigilair": 8.0, "Détecteur gaz & CO": 13.0, "Marche": 6.5, "Alarme": 6.5, "Défaut": 6.5}

def _raw_device(f, t):
    STATE.pop("gas", None)
    if 159.2 <= t <= 160.75 or 160.9 <= t <= 169.6 or 174.3 <= t <= 175.15 or 177.85 <= t <= 178.95:
        _, dr = led_fix(f, t)
    elif any(a <= t <= b for a, b in DEV_SHOTS):
        _, dr = device_fix(f)
    else:
        dr = []
    return dr

def build_devtrack():
    path = "devtrack.json"
    if os.path.exists(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    small = []; raw = {}; i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        small.append(cv2.resize(f, (45, 80)).astype(np.float32))
        t = i / FPS
        if any(a <= t <= b for a, b in DEV_SHOTS):
            raw[i] = [list(map(float, d[:2])) + [d[2], float(d[3])] + list(d[4]) for d in _raw_device(f, t)]
        i += 1
    n = i
    cuts = [0] + [j for j in range(1, n) if np.abs(small[j] - small[j - 1]).mean() > 22] + [n]
    out = {}
    for a, b in zip(cuts, cuts[1:]):
        frames = [j for j in range(a, b) if j in raw]
        if not frames: continue
        words = {}
        for j in frames:
            for cx, cy, wd, hh, r, g, bb in raw[j]:
                words.setdefault(wd, {})[j] = (cx, cy, hh, r, g, bb)
        for wd, obs in words.items():
            if len(obs) < max(4, 0.25 * len(frames)): continue
            js = np.array(sorted(obs)); vals = np.array([obs[j][:3] for j in js])
            col = tuple(int(v) for v in np.median(np.array([obs[j][3:] for j in js]), axis=0))
            span = range(min(frames), max(frames) + 1)
            # médiane glissante puis moyenne glissante, puis interpolation
            med = np.array([np.median(vals[max(0, k - 7):k + 8], axis=0) for k in range(len(js))])
            sm = np.array([med[max(0, k - 4):k + 5].mean(axis=0) for k in range(len(js))])
            for j in span:
                v = [float(np.interp(j, js, sm[:, c])) for c in range(3)]
                out.setdefault(j, []).append([v[0], v[1], wd, v[2], col])
    json.dump(out, open(path, "w"))
    return out

DEVTRACK = None
def stable_device(f, i):
    global DEVTRACK
    if DEVTRACK is None: DEVTRACK = build_devtrack()
    els = DEVTRACK.get(i, [])
    if not els: return f, []
    mask = np.zeros((H, W), np.uint8); draws = []
    for cx, cy, wd, hh, col in els:
        w = WMUL.get(wd, 8) * hh; h = 2.3 * hh
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

def _subs_for(i, boxes):
    t = i / FPS
    claimed = [(a, b) for t0, t1, a, b in CLAIM if t0 <= t <= t1]
    return [b for b in boxes if not any(a <= b[1] + b[3] / 2 <= c for a, c in claimed)]

def build_runs():
    path = "runs.json"
    if os.path.exists(path): return json.load(open(path))
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "src.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    gb = []; i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        subs = _subs_for(i, WB[i] if i < len(WB) else [])
        gb.append(german_box(f, max(subs, key=lambda b: b[2] * b[3])) if subs else None)
        i += 1
    runs = []; cur = None
    for i, b in enumerate(gb):
        if b is None: continue
        cy = b[1] + b[3] / 2
        if cur and i - cur["e"] <= 6 and abs(cy - cur["cy"]) <= 14:
            cur["e"] = i; cur["boxes"].append(b)
        else:
            if cur: runs.append(cur)
            cur = {"s": i, "e": i, "cy": cy, "boxes": [b]}
    if cur: runs.append(cur)
    out = []
    for r in runs:
        bs = np.array(r["boxes"])
        x0 = bs[:, 0].min(); y0 = bs[:, 1].min(); x1 = (bs[:, 0] + bs[:, 2]).max(); y1 = (bs[:, 1] + bs[:, 3]).max()
        out.append([max(0, r["s"] - 8), r["e"] + 2, [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]])
    json.dump(out, open(path, "w"))
    return out

RUNS = None
RUN_OF = {}
def run_at(i):
    global RUNS
    if RUNS is None:
        RUNS = build_runs()
        for k, (s0, e0, b) in enumerate(RUNS):
            for j in range(s0, e0 + 1): RUN_OF.setdefault(j, k)
    k = RUN_OF.get(i)
    return (k, RUNS[k]) if k is not None else (None, None)

RUN_W = {}
def run_width(d, k, run):
    if k in RUN_W: return RUN_W[k]
    s0, e0, b = run
    texts = set()
    for j in range(s0, e0 + 1):
        tt = chunk_at(j / FPS) or prev_chunk(j / FPS)
        if tt: texts.add(tt)
    f = font(16.5 * S, 500)
    tw = max([d.textlength(x, font=f) for x in texts] or [0])
    RUN_W[k] = min((W - 6) * S, max(b[2] * S, tw + 22 * S))
    return RUN_W[k]

def stable_box(d, k, run, txt):
    s0, e0, b = run
    x, y, w, h = b
    bw = run_width(d, k, run); bh = max(h * S, 28 * S)
    cx = (x + w / 2) * S; cy = (y + h / 2) * S
    x0 = min(max(3 * S, cx - bw / 2), W * S - 3 * S - bw)
    x0 = min(x0, x * S); x1 = max(x0 + bw, (x + w) * S)
    box = (x0, cy - bh / 2, x1, cy + bh / 2)
    rr(d, box, (255, 255, 255), 6 * S)
    if txt:
        f = font(16.5 * S, 500)
        tw = d.textlength(txt, font=f)
        if tw > (x1 - x0) - 16 * S: f = font(16.5 * S * ((x1 - x0) - 16 * S) / tw, 500)
        text_center(d, (x0 + x1) / 2, cy, txt, f, (12, 12, 12))

STATE = {"box": None, "bt": -9}

def process(f, i):
    t = i / FPS
    draws = []
    if any(a <= t <= b for a, b in DEV_SHOTS):
        f, draws = stable_device(f, i)
    im = Image.fromarray(f).resize((W * S, H * S), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for cx, cy, wd, hh, col in draws:
        text_center(d, cx * S, cy * S, wd, font(hh * S, 500), col)
    claimed = [(a, b) for t0, t1, a, b in CLAIM if t0 <= t <= t1]
    boxes = WB[i] if i < len(WB) else []
    # carte "Baumarkt Melder"
    if 130.7 <= t <= 135.4:
        for b in boxes:
            if 110 <= b[1] <= 145:
                white_box_text(d, b, "Détecteur de bricolage", weight=700, fsize=12, pad=(5, 3))
        rr(d, (99 * S, 205 * S, 141 * S, 222 * S), (255, 255, 255), 0)
        d.text((100 * S, 213.5 * S), "Alarme", font=font(10 * S, 500), fill=(214, 58, 46), anchor="lm")
    # pastilles de gaz
    if 160.8 <= t <= 165.6:
        for b in boxes:
            cy = b[1] + b[3] / 2
            for py, txt in PILLS:
                if abs(cy - (py + 5)) < 12:
                    white_box_text(d, b, txt, weight=800, fsize=b[3] * 1.0, pad=(9, 5))
    # pilule "lien"
    if 189.75 <= t <= 192.0 or 208.3 <= t <= 213.4:
        box = (14 * S, 491 * S, 346 * S, 521 * S)
        rr(d, box, (255, 255, 255), 15 * S, outline=(20, 20, 20), width=2 * S)
        f2 = font(11.5 * S, 800)
        txt = "LE LIEN EST SOUS LA VIDÉO"
        tw = d.textlength(txt, font=f2)
        cx = 180 * S - 8 * S
        text_center(d, cx, 506 * S, txt, f2, (15, 15, 15))
        ax = cx + tw / 2 + 12 * S
        d.line([(ax, 499 * S), (ax, 513 * S)], fill=(15, 15, 15), width=2 * S)
        d.line([(ax - 5 * S, 508 * S), (ax, 513 * S), (ax + 5 * S, 508 * S)], fill=(15, 15, 15), width=2 * S)
    # carte finale
    if 208.1 <= t <= 213.4:
        rr(d, (29 * S, 60 * S, 331 * S, 163 * S), (255, 255, 255), 10 * S)
        text_center(d, 180 * S, 88 * S, "PLUS DE 12 000", font(26 * S, 800), (12, 12, 12))
        text_center(d, 180 * S, 118 * S, "FAMILLES FRANÇAISES", font(13.5 * S, 800), (12, 12, 12))
        text_center(d, 180 * S, 142 * S, "ONT DÉJÀ FRANCHI LE PAS", font(12 * S, 800), (12, 12, 12))
    # sous-titres (cadre stable sur toute la durée du sous-titre allemand)
    txt = chunk_at(t)
    k, run = run_at(i)
    if run is not None:
        stable_box(d, k, run, txt or prev_chunk(t) or "")
        STATE["run"] = (k, run); STATE["bt"] = t
    elif txt and STATE.get("run") and t - STATE["bt"] < 1.0:
        kk, rn = STATE["run"]
        if not any(a - 12 <= rn[2][1] + rn[2][3] / 2 <= c + 12 for a, c in claimed):
            stable_box(d, kk, rn, txt)
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
        if i % 500 == 0: print(i, flush=True)
    enc.stdin.close(); enc.wait(); print("done", i)

if __name__ == "__main__":
    main()
