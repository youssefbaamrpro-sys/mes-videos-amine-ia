"""Rendu final : sous-titres/étiquettes allemands recouverts par du français (sans Vmake)."""
import json, subprocess, sys, re, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, S = 360, 640, 2          # source, facteur d'agrandissement
FPS = 30
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "Poppins-700.ttf")
NAVY = (8, 1, 52)
_fc = {}
def font(px):
    px = max(8, int(px))
    if px not in _fc: _fc[px] = ImageFont.truetype(FONT, px)
    return _fc[px]

# ---------- étiquettes (t0, t1, couleur, y0, y1, x0, x1, texte) en coordonnées source ----------
SPECS = [
 (0.0, 6.0, "red",   92, 128, 0, 360, "ASTUCE DE SURVIE :"),
 (0.0, 6.0, "navy", 130, 160, 0, 360, "Si tu es intoxiqué"),
 (0.0, 6.0, "navy", 160, 194, 0, 360, "au CO chez toi"),
 (0.0, 6.0, "navy", 196, 238, 0, 360, "NE FAIS PAS ÇA"),
 (45.5, 48.5, "red", 165, 210, 0, 360, "200 FOIS PLUS"),
 (63.5, 70.9, "navy", 108, 142, 0, 360, "NORME EN 50291"),
 (67.5, 71.0, "red", 462, 500, 0, 360, "TOTALEMENT MUET"),
 (69.0, 71.0, "red", 502, 540, 0, 360, "JUSQU'À 70 PPM"),
 (70.9, 74.4, "navy", 106, 152, 0, 360, "MÊME LÀ"),
 (71.5, 74.4, "red", 396, 440, 0, 360, "JUSQU'À 4 HEURES"),
 (72.5, 74.4, "red", 444, 486, 0, 360, "D'ATTENTE"),
 (74.3, 78.7, "orange", 338, 378, 0, 140, "LÉSIONS CÉRÉBRALES"),
 (74.3, 78.7, "red", 278, 316, 0, 138, "ÉVANOUISSEMENT"),
 (78.5, 80.3, "navy", 395, 450, 0, 360, "2 CHOSES"),
 (79.8, 84.5, "navy", 510, 552, 0, 360, "ALARME BIEN AVANT 70 PPM"),
 (87.5, 91.0, "navy", 510, 552, 0, 360, "PAS QUE LE CO"),
 (103.0, 105.5, "green", 412, 456, 0, 360, "4 FONCTIONS ESSENTIELLES"),
 (102.8, 105.5, "navy", 466, 500, 0, 360, "EN 1 APPAREIL"),
 (105.4, 107.2, "green", 498, 542, 0, 360, "CAPTEUR GRADE 3"),
 (112.8, 115.8, "green", 496, 542, 0, 360, "ALERTE PRÉCOCE"),
 (115.8, 122.7, "navy", 104, 152, 0, 360, "4 CAPTEURS"),
 (116.6, 122.7, "red", 310, 352, 0, 360, "MONOXYDE DE CARBONE"),
 (118.0, 122.7, "green", 352, 394, 0, 360, "GAZ NATUREL"),
 (118.7, 122.7, "green", 395, 437, 0, 360, "PROPANE"),
 (119.8, 122.7, "green", 438, 482, 0, 360, "GAZ INFLAMMABLES"),
 (120.5, 122.7, "navy", 494, 534, 0, 360, "4 DANGERS SUR 1 ÉCRAN"),
 (130.3, 132.1, "green", 498, 542, 0, 360, "30 SECONDES"),
 (131.8, 134.6, "red", 358, 395, 0, 360, "PAS D'ÉLECTRICIEN"),
 (132.6, 134.6, "red", 400, 438, 0, 360, "PAS D'ÉCHELLE"),
 (133.3, 134.6, "red", 444, 482, 0, 360, "PAS DE WIFI"),
 (147.5, 150.1, "white", 106, 144, 60, 300, "EN 1 HEURE"),
 (102.8, 105.5, "navy", 104, 152, 0, 360, "VIGILAIR 4 EN 1"),
 (180.3, 182.5, "whitebg", 108, 162, 20, 340, "VIGILAIR 4 EN 1"),
 (147.5, 150.1, "red", 482, 524, 0, 360, "NIVEAUX MORTELS"),
 (166.0, 169.6, "red", 108, 156, 0, 360, "JUSQU'À -50 %"),
 (166.0, 169.6, "navy", 456, 492, 0, 360, "SUR LE PACK FAMILLE"),
 (169.4, 173.3, "circle", 244, 286, 82, 278, "GARANTIE"),
 (169.4, 173.3, "navy", 390, 432, 0, 360, "À VIE"),
 (172.0, 173.3, "green", 442, 488, 0, 360, "REMBOURSÉ À 100 %"),
 (175.0, 180.3, "red", 496, 542, 0, 360, "CLIQUE EN BAS"),
 (180.3, 182.5, "red", 443, 484, 0, 360, "JUSQU'À -50 %"),
 (180.3, 182.5, "blue", 494, 536, 0, 360, "LIEN EN BAS"),
]

def cmask(name, f):
    r, g, b = [f[..., c].astype(int) for c in range(3)]
    if name == "navy":   return (b > 22) & (b < 120) & (r < 60) & (g < 55) & (b - r > 15)
    if name == "red":    return (r > 170) & (g < 90) & (b < 100)
    if name in ("green", "circle"): return (g > 100) & (r < 100) & (b < 120) & (g - r > 45)
    if name == "orange": return (r > 200) & (g > 110) & (g < 200) & (b < 90)
    if name == "blue":   return (b > 160) & (r < 100) & (g > 50) & (g < 160)
    if name in ("white", "whitebg"):  return (r > 220) & (g > 220) & (b > 220)

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

# ---------- sous-titres FR ----------
words = json.load(open("words_fr.json"))
chunks = []; cur = []
for w in words:
    cur.append(w)
    txt = " ".join(x["w"] for x in cur)
    end_punct = re.search(r"[,.;:?!…]$", w["w"])
    if len(cur) >= 3 or len(txt) >= 18 or end_punct:
        chunks.append(cur); cur = []
if cur: chunks.append(cur)
CH = []
for i, c in enumerate(chunks):
    s = c[0]["s"]; e = c[-1]["e"]
    nxt = chunks[i + 1][0]["s"] if i + 1 < len(chunks) else 1e9
    e2 = nxt if nxt - e < 2.5 else e + 0.4
    txt = " ".join(x["w"] for x in c)
    txt = re.sub(r"[,.;…]+$", "", txt)
    txt = re.sub(r"^(Un|Deux) un ", r"\1 : un ", txt)
    CH.append((s, e2, txt, c[0]["seg"]))

def chunk_at(t, nearest=False):
    for s, e, txt, seg in CH:
        if s <= t < e: return txt
    if nearest:
        best = min(CH, key=lambda c: min(abs(c[0] - t), abs(c[1] - t)))
        return best[2]
    return None

NB = json.load(open("navyboxes.json"))
NO_SUB = [(63.5, 78.4), (115.8, 122.6), (147.5, 150.1), (166.0, 173.3), (180.3, 182.5)]

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

STATE={'y':448}
import cv2
_lg = np.array(Image.open("f1045.png").convert("L"))[372:385, 158:203].astype(np.float32)
LOGO_WIN = [(102.8, 105.6), (115.8, 122.8), (165.3, 169.8), (180.2, 182.5)]
def find_logos(f):
    g = cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY).astype(np.float32)
    found = []
    for sc in np.arange(0.45, 1.65, 0.07):
        tp = cv2.resize(_lg, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if tp.shape[0] < 4 or tp.shape[1] < 12: continue
        res = cv2.matchTemplate(g, tp, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res > 0.62)
        for y, x in zip(ys, xs):
            found.append((float(res[y, x]), int(x), int(y), tp.shape[1], tp.shape[0]))
    found.sort(reverse=True); keep = []
    for sc_, x, y, w, h in found:
        if all(abs(x + w/2 - (k[1] + k[3]/2)) > 15 or abs(y - k[2]) > 10 for k in keep):
            keep.append((sc_, x, y, w, h))
        if len(keep) >= 3: break
    return keep
def fix_logos(f, t):
    if not any(a <= t <= b for a, b in LOGO_WIN): return f, []
    L = find_logos(f)
    if not L: return f, []
    f = f.copy(); out = []
    for sc_, x, y, w, h in L:
        x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(W, x + w + 2), min(H, y + h + 2)
        sub = f[y0:y1, x0:x1]
        lum = sub.mean(2)
        bgl = np.median(lum)
        m = (lum > bgl + 18).astype(np.uint8)
        if m.sum() < 5: continue
        col = tuple(int(v) for v in np.median(sub[m.astype(bool)], axis=0))
        mask = np.zeros((H, W), np.uint8); mask[y0:y1, x0:x1] = cv2.dilate(m, np.ones((3, 3), np.uint8))
        f = cv2.inpaint(np.ascontiguousarray(f), mask, 3, cv2.INPAINT_TELEA)
        out.append((x, y, w, h, col, "Vigilair", 0.95))
        # small German labels on the device, positioned relative to the logo
        sc = w / 45.0; cx = x + w / 2; ty = y
        regs = [(-81, -70, -42, 45, [("Marche", -28), ("Alarme", 0), ("Défaut", 29)], 5.5),
                (-30, -18, -34, 34, [("Détecteur gaz & CO", 0)], 7.0)]
        for dy0, dy1, dx0, dx1, words, fh in regs:
            X0, X1 = int(cx + dx0 * sc), int(cx + dx1 * sc); Y0, Y1 = int(ty + dy0 * sc), int(ty + dy1 * sc)
            X0, Y0 = max(0, X0), max(0, Y0); X1, Y1 = min(W, X1), min(H, Y1)
            if X1 - X0 < 6 or Y1 - Y0 < 3: continue
            sub = f[Y0:Y1, X0:X1].astype(int)
            lum = sub.mean(2); sat = sub.max(2) - sub.min(2)
            m2 = ((lum > np.median(lum) + 14) & (sat < 45)).astype(np.uint8)
            if m2.sum() < 3: continue
            colr = tuple(int(v) for v in np.median(sub[m2.astype(bool)], axis=0))
            mk = np.zeros((H, W), np.uint8); mk[Y0:Y1, X0:X1] = cv2.dilate(m2, np.ones((3, 3), np.uint8))
            f = cv2.inpaint(np.ascontiguousarray(f), mk, 3, cv2.INPAINT_TELEA)
            for wd, dx in words:
                out.append((cx + dx * sc - 1, Y0, 2, (Y1 - Y0), colr, wd, fh / max(1, (Y1 - Y0) / sc) ))
    return f, out

def process(f,i):
    t = i / FPS
    f, logos = fix_logos(f, t)
    im = Image.fromarray(f).resize((W * S, H * S), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    for x, y, w, h, col, word, k in logos:
        fl = font(h * k * S)
        d.text(((x + w / 2) * S, (y + h * 0.82) * S), word, font=fl, fill=col, anchor="ms")
    claimed = []
    for sp in SPECS:
        if not (sp[0] <= t <= sp[1]): continue
        claimed.append((sp[3] - 4, sp[4] + 4))
        box = spec_box(sp, f)
        if box is None: continue
        col = median_color(sp[2], f, box) or NAVY
        draw_label(d, box, col, sp[7], big=(sp[2] == "whitebg"))
    # sous-titres allemands -> français
    if any(a <= t <= b for a, b in NO_SUB): return im
    subs = [b for b in NB[i] if not any(b[1] < c1 and b[1] + b[3] > c0 for c0, c1 in claimed)]
    subs = [b for b in subs if b[1] > 380 or (6.0 < t < 11.6 and 170 < b[1] < 230)]
    txt = chunk_at(t)
    if subs:
        b = max(subs, key=lambda b: b[2] * b[3])
        box = extend([b[0] - 3, b[1] - 7, b[0] + b[2] + 3, b[1] + b[3] + 5], f)
        STATE['y'] = (box[1] + box[3]) / 2
        pb = STATE.get('box')
        if pb is not None and t - STATE.get('bt', -9) < 0.5 and abs(pb[1] - box[1]) < 15:
            box = [min(box[0], pb[0]), min(box[1], pb[1]), max(box[2], pb[2]), max(box[3], pb[3])] if t - STATE.get('bt', -9) < 0.1 else box
        STATE['box'] = box; STATE['bt'] = t
        if txt is None:
            prevc = [c for c in CH if c[0] <= t]
            txt = prevc[-1][2] if prevc else ""

        # le fond français couvre toujours entièrement l'allemand
        draw_label(d, box, NAVY, txt)
        for b2 in subs:
            if b2 is not b:
                draw_label(d, [b2[0]-3, b2[1]-3, b2[0]+b2[2]+3, b2[1]+b2[3]+3], NAVY, "")
    elif txt is not None and t > 6.0 and not any(c0 < STATE['y'] + 20 and c1 > STATE['y'] - 20 for c0, c1 in claimed):
        pb = STATE.get('box')
        if pb is not None and t - STATE.get('bt', -9) < 0.8:
            draw_label(d, [pb[0] - 2, pb[1] - 2, pb[2] + 2, pb[3] + 2], NAVY, txt)
        else:
            h = 32
            draw_label(d, [180, STATE['y'] - h / 2, 180, STATE['y'] + h / 2], NAVY, txt)
    return im

def main(start=0, end=None):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "en.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W*S}x{H*S}", "-r", str(FPS), "-i", "-",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "video_fr_noaudio.mp4"], stdin=subprocess.PIPE)
    i = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3: break
        im = process(np.frombuffer(buf, np.uint8).reshape(H, W, 3), i)
        enc.stdin.write(im.tobytes())
        i += 1
        if i % 500 == 0: print(i, flush=True)
    enc.stdin.close(); enc.wait()
    print("done", i)

if __name__ == "__main__":
    main()
