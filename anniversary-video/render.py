"""Anniversary slideshow renderer: python3 -I render.py <photos_dir> <fonts_dir> <out.mp4> <W> <H> [audio.mp3] [audio_start_sec]"""
import sys, os, subprocess, math
import numpy as np, cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

SRC, FONTS, OUT, W, H = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
AUDIO = sys.argv[6] if len(sys.argv) > 6 else None
ASTART = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
FPS, TOTAL, XF = 30, 120.0, 0.8
VERT = H > W
S = min(W, H) / 1080.0
SCRIPT = f"{FONTS}/fontsource-great-vibes-5.0.18/files/great-vibes-latin-400-normal.woff"
SERIF_I = f"{FONTS}/cg/package/files/cormorant-garamond-latin-500-italic.woff"
SERIF = f"{FONTS}/cg/package/files/cormorant-garamond-latin-600-normal.woff"

# (file, poem or None) in story order
SEQ = [
    ("WA0038", "Some souls just know\nthey were made to meet"),
    ("WA0035", None), ("WA0072", None),
    ("WA0053", "Even miles apart,\nyour voice felt like home"),
    ("WA0041", "And then\nthere was us"),
    ("WA0044", None), ("WA0052", None),
    ("WA0054", "You are my favourite\nplace to be"),
    ("WA0056", "Two paths,\none direction"),
    ("WA0065", None),
    ("WA0064", "Hold my hand\nand the world gets quiet"),
    ("WA0066", None), ("WA0086", None),
    ("WA0077", "In the little moments\nI found the biggest love"),
    ("WA0088", "Life is sweeter\nwith you beside me"),
    ("WA0089", None), ("WA0091", None),
    ("WA0061", "Under every sky,\nit's always you"),
    ("WA0057", None), ("WA0005", None),
    ("WA0084", "You make me smile\nwithout even trying"),
    ("WA0082", None), ("WA0073", None),
    ("WA0045", "Every day with you\nis my favourite day"),
    ("WA0002", None),
    ("WA0004", "Here's to every yesterday\nand all our tomorrows"),
]
files = {f[-10:-4]: os.path.join(SRC, f) for f in os.listdir(SRC)}
INTRO, OUTRO = 6.0, 7.4
PH = (TOTAL - INTRO - OUTRO) / len(SEQ)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.array(im)
    # strip black letterbox bars
    rows = np.where(a.mean(axis=(1, 2)) > 12)[0]; cols = np.where(a.mean(axis=(0, 2)) > 12)[0]
    a = a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return a

def face_center(a):
    h, w = a.shape[:2]; k = 800 / max(h, w)
    g = cv2.cvtColor(cv2.resize(a, (int(w * k), int(h * k))), cv2.COLOR_RGB2GRAY)
    fs = cascade.detectMultiScale(g, 1.1, 6, minSize=(40, 40))
    if len(fs) == 0: return 0.5, 0.42
    wsum = sum(fw * fh for _, _, fw, fh in fs)
    cx = sum((x + fw / 2) * fw * fh for x, y, fw, fh in fs) / wsum / g.shape[1]
    cy = sum((y + fh / 2) * fw * fh for x, y, fw, fh in fs) / wsum / g.shape[0]
    return cx, cy

M = 1.12  # canvas margin for Ken Burns
CW, CH = int(W * M), int(H * M)

def build_canvas(a):
    """Return canvas (CH,CW,3) with photo framed, and focus point in canvas coords."""
    h, w = a.shape[:2]; fx, fy = face_center(a)
    ra, rt = w / h, CW / CH
    loss = 1 - min(ra, rt) / max(ra, rt)
    if loss < 0.30:  # cover-crop, centred on faces
        if ra > rt:
            nw = int(h * rt); x0 = int(np.clip(fx * w - nw / 2, 0, w - nw)); a2 = a[:, x0:x0 + nw]
            fx = (fx * w - x0) / nw
        else:
            nh = int(w / rt); y0 = int(np.clip(fy * h - nh * 0.42, 0, h - nh)); a2 = a[y0:y0 + nh]
            fy = (fy * h - y0) / nh
        c = cv2.resize(a2, (CW, CH), interpolation=cv2.INTER_AREA)
        return c, (fx * CW, fy * CH)
    # fit inside on soft blurred backdrop
    k = max(CW / w, CH / h); bg = cv2.resize(a, (int(w * k) + 2, int(h * k) + 2))
    y0, x0 = (bg.shape[0] - CH) // 2, (bg.shape[1] - CW) // 2
    bg = bg[y0:y0 + CH, x0:x0 + CW]
    bg = cv2.GaussianBlur(bg, (0, 0), 40 * S).astype(np.float32) * 0.55
    k = min(CW * 0.92 / w, CH * 0.86 / h) if not VERT else min(CW * 0.94 / w, CH * 0.9 / h)
    pw, ph = int(w * k), int(h * k); p = cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
    px, py = (CW - pw) // 2, (CH - ph) // 2
    if VERT: py = int(CH * 0.47 - ph / 2); py = max(int(CH * 0.06), py)
    # soft shadow
    sh = np.zeros((CH, CW), np.float32); sh[py + 12:py + ph + 12, px + 6:px + pw + 6] = 1
    sh = cv2.GaussianBlur(sh, (0, 0), 18 * S)[..., None]
    bg = bg * (1 - 0.6 * sh)
    b = int(6 * S)  # thin white border
    bg[py - b:py + ph + b, px - b:px + pw + b] = 245
    bg[py:py + ph, px:px + pw] = p
    return bg.clip(0, 255).astype(np.uint8), (px + fx * pw, py + fy * ph)

def text_layer(lines, font, size, color=(255, 255, 255), spacing=1.15, shadow=True):
    f = ImageFont.truetype(font, int(size * S))
    asc, desc = f.getmetrics(); lh = int((asc + desc) * spacing)
    ws = [f.getbbox(l)[2] for l in lines]; tw, th = max(ws) + 80, lh * len(lines) + 60
    img = Image.new("RGBA", (tw, th), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text(((tw - ws[i]) / 2, 30 + i * lh), l, font=f, fill=color + (255,))
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0)); sh.putalpha(img.getchannel("A").point(lambda v: int(v * 0.85)))
        sh = sh.filter(ImageFilter.GaussianBlur(6 * S)); out = Image.new("RGBA", img.size); out.alpha_composite(sh); out.alpha_composite(img); img = out
    return np.array(img).astype(np.float32) / 255.0

def paste(frame, layer, cx, cy, alpha):
    if alpha <= 0: return
    lh, lw = layer.shape[:2]; x0, y0 = int(cx - lw / 2), int(cy - lh / 2)
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + lw), min(H, y0 + lh)
    L = layer[ya - y0:yb - y0, xa - x0:xb - x0]; A = L[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - A) + L[..., :3] * 255 * A

def ease(t): return t * t * (3 - 2 * t)

# vignette + bottom gradient for legibility
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
GRAD = (1 - 0.55 * np.clip((yy / H - 0.55) / 0.45, 0, 1) ** 1.6)[..., None]

# ---------- beat grid (Perfect: ~63.4 BPM) ----------
# defaults fit "Perfect" (63.4 BPM); override via env for other songs
BPM_P = float(os.environ.get("BEAT_P", 0.9464)); BPM_PH = float(os.environ.get("BEAT_PH", 0.156))
INTRO_BEATS = int(os.environ.get("INTRO_BEATS", 6))
B_POEM = int(os.environ.get("BEATS_POEM", 5)); B_PLAIN = int(os.environ.get("BEATS_PLAIN", 4))
def beat(k): return BPM_PH + k * BPM_P - ASTART
SHORT_POEMS = set(os.environ.get("SHORT_POEMS", "WA0041,WA0056,WA0061").split(",")) - {""}
beats_per = [B_POEM if (poem and k not in SHORT_POEMS) else B_PLAIN for k, poem in SEQ]
TRANS = ["zoom", "slide", "fade", "heart", "wipe", "flash", "circle", "slideup", "fade", "zoom",
         "heart", "slide", "flash", "circle", "wipe", "fade", "slide", "heart", "zoom", "slideup",
         "flash", "slide", "circle", "fade", "wipe", "heart", "fade"]  # index i = transition INTO seg i+1

def ease(t): return t * t * (3 - 2 * t)
def ease2(t): t = min(max(t, 0), 1); return 0.5 - 0.5 * math.cos(math.pi * t)

# ---------- build segments ----------
segs = []  # dict(start, dur, kind, data, trans)
b = INTRO_BEATS
segs.append(dict(s=0.0, d=beat(b) - XF / 2, kind="intro", data=None, tr=None))
for i, (k, poem) in enumerate(SEQ):
    print("prep", k, flush=True)
    c, focus = build_canvas(load(files[k]))
    s0 = beat(b) - XF / 2; b += beats_per[i]; s1 = beat(b) - XF / 2
    segs.append(dict(s=s0, d=s1 - s0, kind="photo", data=(c, focus, poem, i), tr=TRANS[i]))
segs.append(dict(s=beat(b) - XF / 2, d=TOTAL - (beat(b) - XF / 2), kind="outro", data=None, tr=TRANS[len(SEQ)]))
print("outro starts", segs[-1]["s"])

# ---------- text ----------
def word_layers(text, font, size, color=(255, 255, 255), spacing=1.15):
    f = ImageFont.truetype(font, int(size * S)); asc, desc = f.getmetrics(); lh = (asc + desc) * spacing
    lines = text.split("\n"); out = []; sp = f.getlength(" ")
    for li, line in enumerate(lines):
        words = line.split(" "); ws = [f.getlength(w) for w in words]
        x = -(sum(ws) + sp * (len(words) - 1)) / 2; y = (li - (len(lines) - 1) / 2) * lh
        for w, wl in zip(words, ws):
            out.append((text_layer([w], font, size, color), x + wl / 2, y)); x += wl + sp
    return out

POEMS = {k: word_layers(p, SERIF_I, 74 if VERT else 66) for k, p in SEQ if p}
title = text_layer(["Happy Anniversary"], SCRIPT, 150 if VERT else 140, color=(255, 236, 220))
date = text_layer(["27  ·  NOVEMBER"], SERIF, 52, color=(245, 225, 205))
o1a = text_layer(["Happy Anniversary,"], SCRIPT, 130 if VERT else 120, color=(255, 236, 220))
o1b = text_layer(["my love"], SCRIPT, 130 if VERT else 120, color=(255, 236, 220))
o2 = word_layers("Forever starts with us", SERIF_I, 60, color=(245, 225, 205))
first_c = build_canvas(load(files["WA0004"]))[0]
BLUR = cv2.GaussianBlur(cv2.resize(first_c, (W, H)), (0, 0), 30 * S).astype(np.float32) * 0.45

def paste(frame, layer, cx, cy, alpha, reveal=None, scale=1.0):
    if alpha <= 0.003: return
    if scale != 1.0:
        layer = cv2.resize(layer, (max(1, int(layer.shape[1] * scale)), max(1, int(layer.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    lh, lw = layer.shape[:2]; x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + lw), min(H, y0 + lh)
    if xb <= xa or yb <= ya: return
    L = layer[ya - y0:yb - y0, xa - x0:xb - x0]; A = L[..., 3:4] * alpha
    if reveal is not None:  # handwriting-style left-to-right reveal with soft edge
        fe = 0.12 * lw; xs = np.arange(xa - x0, xb - x0, dtype=np.float32)
        A = A * np.clip((reveal * (lw + fe) - xs) / fe, 0, 1)[None, :, None]
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - A) + L[..., :3] * 255 * A

def add_glow(frame, layer, cx, cy, alpha):  # additive sprite
    if alpha <= 0.003: return
    lh, lw = layer.shape[:2]; x0, y0 = int(cx - lw / 2), int(cy - lh / 2)
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + lw), min(H, y0 + lh)
    if xb <= xa or yb <= ya: return
    frame[ya:yb, xa:xb] += layer[ya - y0:yb - y0, xa - x0:xb - x0] * alpha

# ---------- transition fields ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
U = min(W, H) / 2
r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
VIG = (1 - 0.28 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
GRAD = (1 - 0.55 * np.clip((yy / H - 0.55) / 0.45, 0, 1) ** 1.6)[..., None]
R_CIRC = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / U
R_WIPE = (xx / W * 0.6 + yy / H * 0.4) * 1.0

def heart_field():
    q = 4; gy, gx = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
    X = (gx * q - W / 2) / U; Y = -(gy * q - H / 2) / U
    R = np.full(X.shape, 99.0, np.float32)
    for sc in np.linspace(3.5, 0.02, 400):
        x, y = X / sc, Y / sc + 0.15
        inside = (x * x + y * y - 1) ** 3 - x * x * y ** 3 <= 0
        R[inside] = sc
    return cv2.resize(R, (W, H), interpolation=cv2.INTER_LINEAR)
R_HEART = heart_field()

def mask_reveal(R, p, rmax, feather):
    rr = p * (rmax + feather)
    m = np.clip((rr - R) / feather, 0, 1)[..., None]
    edge = np.exp(-((R - rr + feather * 0.5) / (feather * 0.6)) ** 2)[..., None] * math.sin(math.pi * p)
    return m, edge

PINK = np.array([255, 170, 190], np.float32); WARM = np.array([255, 214, 170], np.float32)

def scale_img(img, z, cx=None, cy=None):
    cx = W / 2 if cx is None else cx; cy = H / 2 if cy is None else cy
    A = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
    return cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def mix(A, B, p, kind):
    e = ease2(p)
    if kind == "fade": return A * (1 - e) + B * e
    if kind in ("slide", "slideup"):
        out = np.empty_like(A)
        if kind == "slide":
            off = int(e * W); out[:, :W - off] = A[:, off:]; out[:, W - off:] = B[:, :off]
        else:
            off = int(e * H); out[:H - off] = A[off:]; out[H - off:] = B[:off]
        return out
    if kind == "zoom":
        a2 = scale_img(A, 1 + 0.25 * e); b2 = scale_img(B, 1.3 - 0.3 * e)
        return a2 * (1 - e) + b2 * e
    if kind == "flash":
        g = math.sin(math.pi * p) ** 2
        base = A * (1 - e) + B * e
        return base * (1 - 0.4 * g) + WARM * (0.55 * g)
    R, rmax, fth = {"heart": (R_HEART, float(R_HEART.max()), 0.12),
                    "circle": (R_CIRC, float(R_CIRC.max()), 0.10),
                    "wipe": (R_WIPE, 1.0, 0.12)}[kind]
    m, edge = mask_reveal(R, p, rmax, fth)
    out = A * (1 - m) + B * m
    return out + edge * PINK * (0.55 if kind != "wipe" else 0.3)

# ---------- particles ----------
rng = np.random.default_rng(27)
def make_heart(px, color=(255, 120, 150), glow=True):
    n = px * 2; gy, gx = np.mgrid[0:n, 0:n].astype(np.float32)
    x = (gx - n / 2) / (px * 0.42); y = -(gy - n / 2) / (px * 0.42) + 0.15
    v = (x * x + y * y - 1) ** 3 - x * x * y ** 3
    a = (v <= 0).astype(np.float32); a = cv2.GaussianBlur(a, (0, 0), max(0.7, px * 0.025))
    if glow: a = np.maximum(a, cv2.GaussianBlur(a, (0, 0), px * 0.18) * 0.6)
    L = np.zeros((n, n, 4), np.float32); L[..., :3] = np.array(color) / 255.0; L[..., 3] = a
    return L
def make_spark(px):
    n = px * 2; gy, gx = np.mgrid[0:n, 0:n].astype(np.float32); dx, dy = (gx - n / 2) / px, (gy - n / 2) / px
    g = np.exp(-(dx * dx + dy * dy) * 18) + 0.5 * np.exp(-(dx * dx) * 400 - dy * dy * 3) + 0.5 * np.exp(-(dy * dy) * 400 - dx * dx * 3)
    return (g[..., None] * np.array([255, 235, 200], np.float32)).astype(np.float32)

HEARTS = []
for k in range(14):
    sz = int(rng.uniform(16, 40) * S)
    col = [(255, 120, 150), (255, 160, 180), (255, 200, 210), (240, 90, 120)][k % 4]
    HEARTS.append(dict(spr=make_heart(sz, col), x=rng.uniform(0.04, 0.96) * W, spd=rng.uniform(40, 85) * S,
                       amp=rng.uniform(15, 45) * S, f=rng.uniform(0.3, 0.7), ph=rng.uniform(0, 1000), a=rng.uniform(0.35, 0.65)))
BURST = []  # extra hearts for intro & outro
for k in range(26):
    sz = int(rng.uniform(18, 52) * S)
    BURST.append(dict(spr=make_heart(sz, [(255, 120, 150), (255, 175, 190), (250, 100, 130)][k % 3]), x=rng.uniform(0.02, 0.98) * W,
                      spd=rng.uniform(70, 150) * S, amp=rng.uniform(20, 60) * S, f=rng.uniform(0.3, 0.8), ph=rng.uniform(0, 1), a=rng.uniform(0.45, 0.8)))
SPARKS = [dict(spr=make_spark(int(rng.uniform(10, 26) * S)), x=rng.uniform(0, W), y=rng.uniform(0, H),
               f=rng.uniform(0.4, 1.2), ph=rng.uniform(0, 6.28)) for _ in range(22)]
q = 8; LEAK = np.zeros((H // q, W // q, 3), np.float32)
gy, gx = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
for (cx, cy, rad, col) in [(0.2, 0.3, 0.35, (255, 150, 90)), (0.75, 0.6, 0.3, (255, 110, 140)), (0.5, 0.1, 0.25, (255, 200, 130))]:
    d2 = ((gx - cx * W / q) ** 2 + (gy - cy * H / q) ** 2) / (rad * max(W, H) / q) ** 2
    LEAK += np.exp(-d2)[..., None] * np.array(col, np.float32)

def overlays(out, T, burst):
    # drifting light leak
    dx = math.sin(T * 0.13) * W / q * 0.35; dy = math.cos(T * 0.09) * H / q * 0.25
    lk = cv2.warpAffine(LEAK, np.float32([[1, 0, dx], [0, 1, dy]]), (W // q, H // q), borderMode=cv2.BORDER_CONSTANT)
    out += cv2.resize(lk, (W, H), interpolation=cv2.INTER_LINEAR) * (0.10 + 0.05 * math.sin(T * 0.5))
    for sp in SPARKS:
        a = max(0.0, math.sin(T * sp["f"] * 2 * math.pi / 2 + sp["ph"])) ** 4 * 0.8
        add_glow(out, sp["spr"], sp["x"], sp["y"], a * burst)
    for h in HEARTS:
        y = H + 60 * S - ((T * h["spd"] + h["ph"]) % (H + 120 * S))
        x = h["x"] + h["amp"] * math.sin(T * h["f"] * 2 * math.pi + h["ph"])
        fadeedge = min(1, (H + 60 * S - y) / (200 * S), y / (300 * S))
        paste(out, h["spr"], x, y, h["a"] * max(0, fadeedge) * burst)
    if burst > 0:
        for h in BURST:
            y = H + 80 * S - ((T * h["spd"] + h["ph"] * H) % (H + 160 * S))
            x = h["x"] + h["amp"] * math.sin(T * h["f"] * 2 * math.pi + h["ph"] * 6)
            paste(out, h["spr"], x, y, h["a"] * burst * max(0, min(1, y / (300 * S))))

def beat_pulse(T):
    ph = ((T + ASTART - BPM_PH) % BPM_P); return math.exp(-ph * 5)

# ---------- per-segment render ----------
def render_photo(data, lt, d):
    c, (fx, fy), poem, i = data
    p = ease(min(1, lt / (d + XF)))
    zin = i % 2 == 0
    z = 1 + (M - 1) * (p if zin else 1 - p) * 0.9
    ww, wh = CW / z, CH / z
    tx = np.clip(fx - ww / 2, 0, CW - ww); ty = np.clip(fy - wh / 2, 0, CH - wh)
    cx0, cy0 = (CW - ww) / 2, (CH - wh) / 2
    x = cx0 + (tx - cx0) * 0.6; y = cy0 + (ty - cy0) * 0.6
    A = np.float32([[W / ww, 0, -x * W / ww], [0, H / wh, -y * H / wh]])
    f = cv2.warpAffine(c, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return f * VIG * (GRAD if poem else 1)

def seg_frame(sg, T, layers):
    lt = T - sg["s"]; d = sg["d"]
    if sg["kind"] == "photo":
        c, focus, poem, i = sg["data"]
        if poem:
            key = SEQ[i][0]; wl = POEMS[key]; ty = H * (0.80 if VERT else 0.82)
            fo = np.clip((d + XF * 0.3 - lt) / 0.5, 0, 1)
            for k, (L, dx, dy) in enumerate(wl):
                ta = np.clip((lt - 0.45 - k * 0.14) / 0.5, 0, 1)
                layers.append(("p", L, W / 2 + dx, ty + dy + (1 - ease(ta)) * 26 * S, ease(ta) * ease(fo), None, 1.0))
        return render_photo(sg["data"], lt, d)
    if sg["kind"] == "intro":
        f = BLUR * (0.5 + 0.5 * ease(min(1, lt / 1.5)))
        fade = np.clip((d + XF - lt) / 0.7, 0, 1); cy = H * 0.45
        layers.append(("p", title, W / 2, cy, fade, ease2(np.clip((lt - 0.4) / 2.2, 0, 1)), 1.0))
        hb = ease(np.clip((lt - 2.4) / 0.6, 0, 1))
        layers.append(("p", HEART_ICON, W / 2, cy + 125 * S, hb * fade, None, (0.6 + 0.4 * hb) * (1 + 0.18 * beat_pulse(T))))
        layers.append(("p", date, W / 2, cy + 210 * S + (1 - ease(np.clip((lt - 2.8) / 0.8, 0, 1))) * 20 * S,
                       ease(np.clip((lt - 2.8) / 0.8, 0, 1)) * fade, None, 1.0))
        return f
    # outro
    f = BLUR * (1 - 0.85 * ease(np.clip((lt - (d - 1.6)) / 1.6, 0, 1)))
    fo = 1 - ease(np.clip((lt - (d - 1.8)) / 1.5, 0, 1)); cy = H * 0.42; lh = 150 * S if VERT else 135 * S
    layers.append(("p", o1a, W / 2, cy - lh / 2, fo, ease2(np.clip((lt - 0.6) / 1.8, 0, 1)), 1.0))
    layers.append(("p", o1b, W / 2, cy + lh / 2, fo, ease2(np.clip((lt - 2.2) / 1.2, 0, 1)), 1.0))
    for k, (L, dx, dy) in enumerate(o2):
        ta = np.clip((lt - 3.2 - k * 0.15) / 0.5, 0, 1)
        layers.append(("p", L, W / 2 + dx, cy + lh + 95 * S + dy + (1 - ease(ta)) * 20 * S, ease(ta) * fo, None, 1.0))
    da = ease(np.clip((lt - 4.2) / 0.8, 0, 1))
    layers.append(("p", date, W / 2, cy + lh + 180 * S, da * fo, None, 1.0))
    layers.append(("p", HEART_ICON, W / 2, cy - lh - 90 * S, da * fo, None, 1 + 0.18 * beat_pulse(T)))
    return f

HEART_ICON = make_heart(int(34 * S), (255, 130, 160))

def frame_at(T):
    act = [sg for sg in segs if sg["s"] - 1e-6 <= T < sg["s"] + sg["d"] + XF]
    layers = []
    if len(act) == 1:
        out = seg_frame(act[0], T, layers)
    else:
        A_, B_ = act[0], act[-1]
        fa = seg_frame(A_, T, layers); fb = seg_frame(B_, T, layers)
        out = mix(fa, fb, (T - B_["s"]) / XF, B_["tr"])
    out = np.ascontiguousarray(out, dtype=np.float32)
    s0, sl = segs[0], segs[-1]
    burst = max(np.clip((s0["d"] + XF - T) / 1.0, 0, 1) * ease(np.clip(T / 1.2, 0, 1)),
                ease(np.clip((T - sl["s"]) / 1.5, 0, 1)) * (1 - ease(np.clip((T - (TOTAL - 1.5)) / 1.5, 0, 1))))
    overlays(out, T, burst)
    for (_, L, x, y, al, rev, sc) in layers: paste(out, L, x, y, al, rev, sc)
    return out.clip(0, 255).astype(np.uint8)

cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
if AUDIO:
    cmd += ["-ss", str(ASTART), "-i", AUDIO, "-map", "0:v", "-map", "1:a",
            "-af", f"afade=t=in:d=1.5,afade=t=out:st={TOTAL-4}:d=4", "-c:a", "aac", "-b:a", "256k", "-shortest"]
cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-movflags", "+faststart", "-t", str(TOTAL), OUT]
if os.environ.get("STILLS"):
    for T in map(float, os.environ["STILLS"].split(",")):
        Image.fromarray(frame_at(T)).save(f"{OUT}_{T:05.1f}.jpg", quality=85)
    sys.exit()
pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
N = int(TOTAL * FPS)
for n in range(N):
    pr.stdin.write(frame_at(n / FPS).tobytes())
    if n % 300 == 0: print("frame", n, "/", N, flush=True)
pr.stdin.close(); pr.wait(); print("done", OUT)
