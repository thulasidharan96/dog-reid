# mp_face.py
# pip install mediapipe
import numpy as np
import cv2

try:
    import mediapipe as mp
    MP_OK = True
except Exception:
    MP_OK = False

_FACE = None

def _lazy_init():
    global _FACE
    if not MP_OK:
        return
    if _FACE is None:
        _FACE = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,        # faster; dog faces are approximate anyway
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )

def _procrustes_norm(pts):
    """Center + scale + rotate to canonical using Procrustes to make it pose/scale invariant."""
    pts = pts.astype(np.float32)
    c = pts.mean(axis=0, keepdims=True)
    pts = pts - c
    s = np.sqrt((pts**2).sum()/len(pts)) + 1e-6
    pts = pts / s
    # PCA to canonical rotation (2D)
    U, S, Vt = np.linalg.svd(pts, full_matrices=False)
    pts = pts @ Vt.T
    return pts

def _pairwise_log_hist(pts, bins=32):
    """Rotation/translation/scale-invariant log–distance histogram (compact 32-D)."""
    d = np.sqrt(((pts[None,:,:]-pts[:,None,:])**2).sum(-1) + 1e-6)
    triu = d[np.triu_indices(len(pts), 1)]
    triu = np.log(triu + 1e-6)
    h, edges = np.histogram(triu, bins=bins, range=(np.min(triu), np.max(triu)))
    h = h.astype(np.float32)
    if h.sum() > 0: h /= h.sum()
    return h[None, :]  # [1,bins]

def mp_face_descriptor(bgr, want_bins=32):
    """
    Returns (desc, ok). desc is [1, D] float32.
    ok=False when mediapipe isn't available or no landmarks are found.
    """
    if not MP_OK:
        return np.zeros((1, want_bins + 64), np.float32), False

    _lazy_init()
    if _FACE is None:
        return np.zeros((1, want_bins + 64), np.float32), False

    h, w = bgr.shape[:2]
    if h < 60 or w < 60:
        return np.zeros((1, want_bins + 64), np.float32), False

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = _FACE.process(rgb)
    if not res.multi_face_landmarks:
        return np.zeros((1, want_bins + 64), np.float32), False

    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([(p.x*w, p.y*h) for p in lm], dtype=np.float32)  # [468,2]
    pts = _procrustes_norm(pts)                                     # canonicalize

    # 1) pairwise log-distance histogram (32D by default)
    hist = _pairwise_log_hist(pts, bins=want_bins)                  # [1,32]

    # 2) compact landmark embedding via uniform subsample + linear projection (64D)
    idx = np.linspace(0, len(pts)-1, 64, dtype=np.int32)
    sub = pts[idx]                                                  # [64,2]
    sub = sub.reshape(1, -1)                                        # [1,128]
    # random but fixed projection to 64 dims (deterministic seed)
    rng = np.random.RandomState(1234)
    P = rng.normal(size=(128, 64)).astype(np.float32)
    proj = (sub @ P).astype(np.float32)                             # [1,64]
    # L2 norm both
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-6
        return x / n
    desc = np.concatenate([_norm(hist), _norm(proj)], axis=1)       # [1, 32+64 = 96]
    return desc.astype(np.float32), True
