# face_shape.py
import numpy as np
import cv2

# ---- Head mask (silhouette) ----
def head_mask_from_crop(head_bgr: np.ndarray) -> np.ndarray:
    """Return binary mask (uint8 0/255) for head region (GrabCut + cleanup)."""
    h, w = head_bgr.shape[:2]
    if h < 30 or w < 30:
        return np.zeros((h, w), np.uint8)
    # gentle grabcut
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w*0.06), int(h*0.06), int(w*0.88), int(h*0.88))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(head_bgr, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    except Exception:
        # fallback with adaptive threshold on V channel
        hsv = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2]
        th = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, -7)
        mask = th

    # cleanup
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask*255, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # keep largest component
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask)
    c = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, thickness=-1)
    return out

# ---- Contour extraction ----
def contour_from_mask(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

# ---- Elliptic Fourier Descriptors (magnitude-only, normalized) ----
def efd_descriptor(contour, harmonics: int = 10) -> np.ndarray:
    """Return 2*harmonics magnitudes (cos/sin) normalized for scale/rotation."""
    if contour is None or len(contour) < harmonics*2:
        return np.zeros((1, 2*harmonics), dtype=np.float32)
    pts = contour[:,0,:].astype(np.float32)  # (N,2)
    # uniform resample along arc length
    d = np.sqrt(((np.roll(pts,-1,axis=0)-pts)**2).sum(1))
    s = np.cumsum(d); s = np.insert(s, 0, 0.0)[:-1]
    total = s[-1] + d[-1]
    n = max(200, harmonics*20)
    us = np.linspace(0, total, n, endpoint=False)
    xs = np.interp(us, s, pts[:,0]); ys = np.interp(us, s, pts[:,1])
    # center & scale
    xs -= xs.mean(); ys -= ys.mean()
    scale = np.sqrt((xs**2 + ys**2).mean()) + 1e-6
    xs /= scale; ys /= scale
    # FFT
    X = np.fft.rfft(xs); Y = np.fft.rfft(ys)
    # Take 1..harmonics magnitudes from both X and Y
    mags = []
    for k in range(1, harmonics+1):
        mags.append(np.abs(X[k])); mags.append(np.abs(Y[k]))
    v = np.array(mags, dtype=np.float32)[None, :]
    # L2 normalize
    nrm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    v = v / nrm
    return v.astype(np.float32)

# ---- Radial signature (shape context-ish) ----
def radial_shape_signature(contour, bins: int = 36) -> np.ndarray:
    """36-d radial distances from centroid to contour at equal angles, normalized."""
    if contour is None or len(contour) < 5:
        return np.zeros((1, bins), dtype=np.float32)
    pts = contour[:,0,:].astype(np.float32)
    c = pts.mean(0)  # centroid
    vecs = pts - c
    ang = (np.arctan2(vecs[:,1], vecs[:,0]) + np.pi)  # 0..2π
    dist = np.sqrt((vecs**2).sum(1))
    # normalize by RMS radius
    rms = np.sqrt((dist**2).mean()) + 1e-6
    dist /= rms
    # bin by angle
    sig = np.zeros((bins,), np.float32)
    counts = np.zeros((bins,), np.float32)
    idx = np.floor(ang / (2*np.pi) * bins).astype(int)
    idx = np.clip(idx, 0, bins-1)
    for i, r in zip(idx, dist):
        sig[i] += r; counts[i] += 1.0
    counts[counts==0] = 1.0
    sig /= counts
    # L2 normalize
    sig = sig / (np.linalg.norm(sig) + 1e-6)
    return sig[None, :].astype(np.float32)

# ---- Hu moments of head mask (log-scaled) ----
def hu_moments(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((1,7), dtype=np.float32)
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)  # log -> better scale
    hu = hu.astype(np.float32)[None, :]
    # standardize
    hu = (hu - hu.mean()) / (hu.std() + 1e-6)
    return hu.astype(np.float32)
