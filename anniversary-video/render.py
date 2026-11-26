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

# ---------- build segments ----------
segs = []  # (start, dur, kind, data)
t = 0.0
segs.append((t, INTRO, "intro", None)); t += INTRO
for i, (k, poem) in enumerate(SEQ):
    print("prep", k, flush=True)
    c, focus = build_canvas(load(files[k]))
    tl = text_layer(poem.split("\n"), SERIF_I, 74 if VERT else 66) if poem else None
    segs.append((t, PH, "photo", (c, focus, tl, i))); t += PH
segs.append((t, OUTRO, "outro", None))

title = text_layer(["Happy Anniversary"], SCRIPT, 150 if VERT else 140, color=(255, 236, 220))
date = text_layer(["27  ·  NOVEMBER"], SERIF, 52, color=(245, 225, 205))
heart = text_layer(["♡"], "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60, color=(255, 200, 200))
o1 = text_layer(["Happy Anniversary,", "my love"], SCRIPT, 130 if VERT else 120, color=(255, 236, 220), spacing=1.0)
o2 = text_layer(["Forever starts with us"], SERIF_I, 60, color=(245, 225, 205))
first_c = build_canvas(load(files["WA0004"]))[0]
BLUR = cv2.GaussianBlur(cv2.resize(first_c, (W, H)), (0, 0), 30 * S).astype(np.float32) * 0.45

def render_photo(data, lt):
    c, (fx, fy), tl, i = data
    p = ease(min(1, lt / (PH + XF)))
    zin = i % 2 == 0
    z = 1 + (M - 1) * (p if zin else 1 - p) * 0.9  # zoom factor relative to output
    scale = 1 / (1 + (z - 1))
    ww, wh = CW * scale, CH * scale
    # pan gently toward focus
    tx = np.clip(fx - ww / 2, 0, CW - ww); ty = np.clip(fy - wh / 2, 0, CH - wh)
    cx0, cy0 = (CW - ww) / 2, (CH - wh) / 2
    x = cx0 + (tx - cx0) * 0.6; y = cy0 + (ty - cy0) * 0.6
    A = np.float32([[W / ww, 0, -x * W / ww], [0, H / wh, -y * H / wh]])
    f = cv2.warpAffine(c, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    return f, tl

def frame_at(T):
    out = np.zeros((H, W, 3), np.float32); wsum = 0
    layers = []
    for (s, d, kind, data) in segs:
        if not (s - 1e-6 <= T < s + d + XF) : continue
        lt = T - s
        a = 1.0
        if s > 0 and lt < XF: a = ease(lt / XF)
        if kind == "photo":
            f, tl = render_photo(data, lt)
            f = f * VIG * (GRAD if tl is not None else 1)
            out = out * (1 - a) + f * a
            if tl is not None:
                ta = np.clip((lt - 0.5) / 0.6, 0, 1) * np.clip((d + 0.2 - lt) / 0.6, 0, 1)
                layers.append((tl, W / 2, H * (0.80 if VERT else 0.82) - (1 - ta) * 12 * S, ease(ta) * 1.0))
        elif kind == "intro":
            f = BLUR * (0.6 + 0.4 * ease(min(1, lt / 1.5)))
            out = out * (1 - a) + f * a
            fade = np.clip((d - lt) / 0.6, 0, 1)
            layers.append((title, W / 2, H * 0.45, ease(np.clip((lt - 0.6) / 1.2, 0, 1)) * fade))
            layers.append((heart, W / 2, H * 0.45 + 120 * S, ease(np.clip((lt - 1.6) / 1, 0, 1)) * fade))
            layers.append((date, W / 2, H * 0.45 + 200 * S, ease(np.clip((lt - 2.0) / 1, 0, 1)) * fade))
        else:  # outro
            f = BLUR * (1 - 0.8 * ease(np.clip((lt - (d - 1.4)) / 1.4, 0, 1)))
            out = out * (1 - a) + f * a
            fo = 1 - ease(np.clip((lt - (d - 1.6)) / 1.4, 0, 1))
            layers.append((o1, W / 2, H * 0.44, ease(np.clip((lt - 0.8) / 1.4, 0, 1)) * fo))
            layers.append((o2, W / 2, H * 0.44 + 190 * S, ease(np.clip((lt - 2.0) / 1.2, 0, 1)) * fo))
            layers.append((date, W / 2, H * 0.44 + 275 * S, ease(np.clip((lt - 2.6) / 1.2, 0, 1)) * fo))
    for L, x, y, al in layers: paste(out, L, x, y, al)
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
