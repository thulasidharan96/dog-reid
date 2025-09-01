# dog_reid.py  —  v5 angle-robust (ResNet-50 or EfficientNet-V2-S selectable)
import argparse, json, os, time, math
from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from ultralytics import YOLO
import faiss

from face_shape import (
    head_mask_from_crop, contour_from_mask,
    efd_descriptor, radial_shape_signature, hu_moments
)
from mp_face import mp_face_descriptor

# =========================
# Config
# =========================
INDEX_PATH = os.path.abspath("dog_index.faiss")
META_PATH  = os.path.abspath("dog_metadata.json")
INDEX_INFO = os.path.abspath("index_info.json")
CROPS_DIR  = os.path.abspath("crops")

# Detector
YOLO_WEIGHTS = "yolov8n.pt"   # you can set --yolo_s to auto use yolov8s.pt
CLASS_NAME   = "dog"
CONF_THRESH  = 0.20
IOU_THRESH   = 0.45

# Hist + thresholds
COLOR_BINS = (4, 4, 4)    # HSV -> 64 dims
DEFAULT_THRESHOLD = 0.86  # For ResNet-50 try 0.84 via --threshold 0.84

# Breed tweak
BREED_BOOST   = 1.02
BREED_PENALTY = 0.99

# FAISS defaults
DEFAULT_INDEX_TYPE = "flat"
DEFAULT_NLIST = 4096
DEFAULT_PQ_M  = 16
DEFAULT_PQ_BITS = 8

# Re-ranking + QE
RERANK_TOPK = 10   # neighbors for re-ranking
QE_TOPK     = 5    # neighbors for query expansion
QE_ALPHA    = 0.25 # how much neighbor mean influences query

# =========================
# Utils
# =========================
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_json(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_metadata(path: str) -> List[dict]:
    return load_json(path, [])

def save_metadata(path: str, data: List[dict]):
    ensure_dir(os.path.dirname(path))
    save_json(path, data)

def to_cosine_index(dim: int):
    return faiss.IndexFlatIP(dim)

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

# =========================
# Hand-crafted features
# =========================
def color_hist_hsv(bgr: np.ndarray, bins=COLOR_BINS) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, bins, [0,180, 0,256, 0,256]).flatten()
    if hist.sum() > 0: hist = hist / hist.sum()
    return hist.astype("float32").reshape(1, -1)

def banded_color_hsv(bgr: np.ndarray, bins=(2,2,2), bands: int = 3) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h < bands:
        return np.zeros((1, bins[0]*bins[1]*bins[2]*bands), dtype=np.float32)
    step = h // bands
    out = []
    for i in range(bands):
        y1 = i*step
        y2 = h if i == bands-1 else (i+1)*step
        out.append(color_hist_hsv(bgr[y1:y2, :], bins=bins))
    return np.concatenate(out, axis=1).astype("float32")

def edge_orientation_hist(bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    hist = np.zeros((bins,), dtype=np.float32)
    if mag.size:
        bin_idx = np.floor((ang.flatten() * bins) / (2*np.pi)).astype(int)
        bin_idx = np.clip(bin_idx, 0, bins-1)
        for b, m in zip(bin_idx, mag.flatten()):
            hist[b] += m
    s = float(hist.sum())
    if s > 0: hist /= s
    return hist.reshape(1, -1).astype("float32")

def shape_features(bbox: Tuple[int,int,int,int], frame_wh: Tuple[int,int]) -> np.ndarray:
    (x1, y1, x2, y2) = bbox
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    aspect = float(w) / float(h)
    fw, fh = frame_wh
    area_ratio = (w*h) / float(max(1, fw*fh))
    aspect = np.clip(aspect, 0.2, 5.0) / 5.0
    area_ratio = np.clip(area_ratio, 0.0, 0.5) / 0.5
    return np.array([[aspect, area_ratio]], dtype="float32")

def tail_visibility_score(crop_bgr: np.ndarray) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return np.array([[0.0]], dtype="float32")
    band = crop_bgr[int(h*0.55):, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 60, 160)
    bw = edges.shape[1]; strip = max(2, int(bw * 0.20))
    l_density = edges[:, :strip].mean() / 255.0
    r_density = edges[:, -strip:].mean() / 255.0
    score = float(np.clip(max(l_density, r_density), 0.0, 1.0))
    return np.array([[score]], dtype="float32")

def gray_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    p = hist / (hist.sum() + 1e-9)
    e = -(p[p>0] * np.log2(p[p>0])).sum()
    return float(np.clip(e / 8.0, 0.0, 1.0))

def texture_stats(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray)) / 255.0
    std  = float(np.std(gray)) / 255.0
    lap  = cv2.Laplacian(gray, cv2.CV_32F).var()
    lapn = float(np.clip(lap / 500.0, 0.0, 1.0))
    ent  = gray_entropy(gray)
    return np.array([[mean, std, lapn, ent]], dtype="float32")

def ear_color_hsv(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h < 10 or w < 10:
        return np.zeros((1,6), dtype="float32")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    maskL = np.zeros((h,w), dtype=np.uint8); maskR = np.zeros((h,w), dtype=np.uint8)
    triL = np.array([[0,0],[w//2,0],[0,h//2]], dtype=np.int32)
    triR = np.array([[w,0],[w//2,0],[w,h//2]], dtype=np.int32)
    cv2.fillConvexPoly(maskL, triL, 1); cv2.fillConvexPoly(maskR, triR, 1)
    mL = hsv[maskL==1]; mR = hsv[maskR==1]
    meanL = mL.mean(axis=0).astype(np.float32) if mL.size else np.array([0,0,0], dtype=np.float32)
    meanR = mR.mean(axis=0).astype(np.float32) if mR.size else np.array([0,0,0], dtype=np.float32)
    HnL = meanL[0]/180.0; SnL = meanL[1]/255.0; VnL = meanL[2]/255.0
    HnR = meanR[0]/180.0; SnR = meanR[1]/255.0; VnR = meanR[2]/255.0
    return np.array([[HnL,SnL,VnL,HnR,SnR,VnR]], dtype="float32")

def ear_edge_density(head_bgr: np.ndarray) -> np.ndarray:
    h, w = head_bgr.shape[:2]
    if h < 20 or w < 20: return np.zeros((1,2), dtype=np.float32)
    gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    TL = edges[:h//2, :w//2]; TR = edges[:h//2, w//2:]
    l = float(TL.mean()/255.0); r = float(TR.mean()/255.0)
    return np.array([[l, r]], dtype="float32")

def head_symmetry_score(head_bgr: np.ndarray) -> np.ndarray:
    h, w = head_bgr.shape[:2]
    if h < 20 or w < 20: return np.array([[0.5]], dtype=np.float32)
    gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
    L = gray[:, :w//2]; R = gray[:, w - w//2:]
    Rf = cv2.flip(R, 1)
    L = cv2.resize(L, (Rf.shape[1], Rf.shape[0]))
    diff = np.mean(np.abs(L.astype(np.float32) - Rf.astype(np.float32))) / 255.0
    score = float(np.clip(1.0 - diff*2.0, 0.0, 1.0))
    return np.array([[score]], dtype="float32")

def detect_eyes_nose_points(gray_head: np.ndarray) -> Tuple[float,float,float,float,float,float,float,float]:
    h, w = gray_head.shape
    if h < 24 or w < 24: return (-1,-1,-1,-1,-1,-1,-1,-1)
    eq = cv2.equalizeHist(gray_head)
    up = eq[:h//2, :]
    eL=(-1,-1,-1); eR=(-1,-1,-1)
    try:
        circles = cv2.HoughCircles(up, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(8,w//8), param1=100, param2=10,
                                   minRadius=2, maxRadius=max(4, w//12))
        if circles is not None:
            c = np.squeeze(circles, axis=0)
            if c.ndim == 1: c = c[None, ...]
            li = int(np.argmin(c[:,0])); ri = int(np.argmax(c[:,0]))
            eL = (float(c[li,0])/w, float(c[li,1])/h, float(c[li,2])/w)
            eR = (float(c[ri,0])/w, float(c[ri,1])/h, float(c[ri,2])/w)
            eL = (np.clip(eL[0],0,1), np.clip(eL[1],0,1), np.clip(eL[2],0,1))
            eR = (np.clip(eR[0],0,1), np.clip(eR[1],0,1), np.clip(eR[2],0,1))
    except Exception:
        pass
    band = eq[h//3:h, :]
    nx, ny = -1.0, -1.0
    if band.size > 0:
        _, th = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            ci = int(np.argmax([cv2.contourArea(c) for c in cnts]))
            M = cv2.moments(cnts[ci])
            if M["m00"] > 0:
                cx = (M["m10"]/M["m00"]); cy = (M["m01"]/M["m00"]) + (h//3)
                nx = np.clip(cx / w, 0, 1); ny = np.clip(cy / h, 0, 1)
    return (eL[0],eLy:=eL[1],eL[2], eR[0],eRy:=eR[1],eR[2], nx,ny)

def facial_color_from_points(head_bgr: np.ndarray,
                             geom: Tuple[float,float,float,float,float,float,float,float]) -> np.ndarray:
    h, w = head_bgr.shape[:2]
    hsv = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2HSV)
    eLx,eLy,eLr,eRx,eRy,eRr,nx,ny = geom
    def sample(ptx, pty, r=4):
        if ptx<0 or pty<0: return None
        cx = int(ptx * w); cy = int(pty * h)
        x1=max(0,cx-r); y1=max(0,cy-r); x2=min(w,cx+r); y2=min(h,cy+r)
        roi = hsv[y1:y2, x1:x2]
        return roi if roi.size>0 else None
    eyeL = sample(eLx,eLy); eyeR = sample(eRx,eRy); nose = sample(nx,ny)
    eye_v = 0.5
    if eyeL is not None or eyeR is not None:
        Vs=[]
        if eyeL is not None: Vs.append(eyeL[:,:,2].mean()/255.0)
        if eyeR is not None: Vs.append(eyeR[:,:,2].mean()/255.0)
        eye_v = float(np.mean(Vs))
    nose_h, nose_s = 0.0, 0.0
    if nose is not None:
        nose_h = float(nose[:,:,0].mean()/180.0)
        nose_s = float(nose[:,:,1].mean()/255.0)
    return np.array([[eye_v, nose_h, nose_s]], dtype="float32")

def facial_anatomy_metrics(geom: Tuple[float,float,float,float,float,float,float,float]) -> np.ndarray:
    eLx,eLy,eLr,eRx,eRy,eRr,nx,ny = geom
    if eLx<0 or eRx<0:
        return np.array([[0.5, 1.0, 0.5, 0.5, 0.5]], dtype=np.float32)
    interocular = float(np.clip(abs(eRx - eLx), 0.05, 0.9))
    eye_center_y = float(np.clip((eLy + eRy)/2.0, 0.0, 1.0))
    if eLr>0 and eRr>0:
        eye_size_ratio = float(np.clip(min(eLr,eRr)/max(eLr,eRr), 0.0, 1.0))
    else:
        eye_size_ratio = 1.0
    dx = eRx - eLx; dy = eRy - eLy
    tilt = abs(np.arctan2(dy, dx)) / (np.pi/2)
    eye_line_flatness = float(np.clip(1.0 - tilt, 0.0, 1.0))
    midx = (eRx + eLx)/2.0; midy = (eRy + eLy)/2.0
    dist = np.sqrt((nx - midx)**2 + (ny - midy)**2) / max(1e-3, interocular)
    eyes_to_nose = float(np.clip(1.0 - dist, 0.0, 1.0))
    return np.array([[interocular, eye_size_ratio, eye_line_flatness, eye_center_y, eyes_to_nose]], dtype=np.float32)

# =========================
# Pose/Align + TTA helpers
# =========================
def grabcut_foreground(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    if h < 40 or w < 40: return bgr
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w*0.06), int(h*0.06), int(w*0.88), int(h*0.88))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        out = bgr.copy()
        out[mask2 == 0] = (out[mask2 == 0] * 0.35).astype(out.dtype)
        return out
    except Exception:
        return bgr

def rotate_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    c = (w/2, h/2)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0,0]); sin = abs(M[0,1])
    nW = int((h*sin) + (w*cos))
    nH = int((h*cos) + (w*sin))
    M[0,2] += (nW/2) - c[0]
    M[1,2] += (nH/2) - c[1]
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

def align_head_by_eyes(head_bgr: np.ndarray, geom) -> np.ndarray:
    eLx,eLy,_, eRx,eRy,_, _,_ = geom
    h, w = head_bgr.shape[:2]
    if eLx<0 or eRx<0 or h<24 or w<24: 
        return head_bgr
    pL = (eLx*w, eLy*h); pR = (eRx*w, eRy*h)
    dx = pR[0]-pL[0]; dy = pR[1]-pL[1]
    angle = -math.degrees(math.atan2(dy, dx))
    rot = rotate_bound(head_bgr, angle)
    return rot

def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def fivecrop_tensor(pil_img: Image.Image, size: int, device: str):
    tnorm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = pil_img.resize((size, size))
    crops = [
        img.crop((0, 0, size//2, size//2)),
        img.crop((size//2, 0, size, size//2)),
        img.crop((0, size//2, size//2, size)),
        img.crop((size//2, size//2, size, size)),
        img.crop((size//4, size//4, size*3//4, size*3//4)),
    ]
    batch = torch.stack([tnorm(c) for c in crops], dim=0).to(device)
    return batch

# ---------- ORB/AKAZE + homography local check ----------
def _homography_inlier_ratio(kpa, desA, kpb, desB, norm, ratio=0.75):
    if desA is None or desB is None or len(kpa) < 8 or len(kpb) < 8:
        return 0.0
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn = bf.knnMatch(desA, desB, k=2)
    good = [m for m,n in knn if m.distance < ratio * n.distance]
    if len(good) < 8:
        return 0.0
    src = np.float32([kpa[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kpb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if mask is None:
        return 0.0
    inliers = int(mask.sum())
    return inliers / max(8, len(good))

def local_match_score(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    grayA = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    r1 = 0.0
    try:
        akaze = cv2.AKAZE_create()
        kpa, da = akaze.detectAndCompute(grayA, None)
        kpb, db = akaze.detectAndCompute(grayB, None)
        r1 = _homography_inlier_ratio(kpa, da, kpb, db, cv2.NORM_HAMMING)
    except Exception:
        r1 = 0.0
    r2 = 0.0
    try:
        orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
        kpa, da = orb.detectAndCompute(grayA, None)
        kpb, db = orb.detectAndCompute(grayB, None)
        r2 = _homography_inlier_ratio(kpa, da, kpb, db, cv2.NORM_HAMMING)
    except Exception:
        r2 = 0.0
    return float(np.clip(0.6*r1 + 0.4*r2, 0.0, 1.0))

# =========================
# Backbones
# =========================
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = torch.clamp(x, min=self.eps).pow(self.p)
        x = torch.mean(x, dim=(-1, -2))
        return x.pow(1.0 / self.p)

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 trunk with GeM pooling (better for ReID than avgpool).
    Output dim: 2048
    """
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = GeM()
        self.out_dim = 2048
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x)   # [B, 2048]
        return x

class EffBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 1280
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

# =========================
# Core ReID
# =========================
class DogReID:
    def _expand_square(self, bbox, W, H, pad=0.18):
        x1, y1, x2, y2 = bbox
        w = x2 - x1; h = y2 - y1
        cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
        side = max(w, h) * (1.0 + pad)
        x1n = int(max(0, cx - side / 2))
        y1n = int(max(0, cy - side / 2))
        x2n = int(min(W, cx + side / 2))
        y2n = int(min(H, cy + side / 2))
        return (x1n, y1n, x2n, y2n)

    def __init__(self,
                 index_path: str = INDEX_PATH,
                 meta_path: str = META_PATH,
                 crops_dir: str = CROPS_DIR,
                 threshold: float = DEFAULT_THRESHOLD,
                 use_yolo_s: bool = False,
                 enable_head_align: bool = True,
                 enable_angle_tta: bool = True,
                 enable_rerank: bool = True,
                 enable_qe: bool = True,
                 backbone_name: str = "resnet50"):

        ensure_dir(crops_dir)
        torch.backends.cudnn.benchmark = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            try: torch.cuda.set_device(0)
            except Exception: pass

        # Detector
        yolo_w = "yolov8s.pt" if use_yolo_s else YOLO_WEIGHTS
        self.yolo = YOLO(yolo_w)
        try:
            self.yolo.to(0 if self.device == "cuda" else "cpu")
            print(f"[info] YOLO model loaded to {self.device}")
        except Exception:
            pass

        self.class_names = self.yolo.model.names if hasattr(self.yolo, "model") else getattr(self.yolo, "names", {})

        # --- ReID backbone choice ---
        self.backbone_name = backbone_name.lower()
        if self.backbone_name == "resnet50":
            self.backbone = ResNet50Backbone().to(self.device).eval()
        elif self.backbone_name == "effv2s":
            self.backbone = EffBackbone().to(self.device).eval()
        else:
            raise ValueError("Unknown backbone_name; use 'resnet50' or 'effv2s'")

        # Breed classifier (kept EfficientNet-B0)
        self.cls_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        ).to(self.device).eval()
        self.imagenet_classes = models.EfficientNet_B0_Weights.IMAGENET1K_V1.meta.get("categories", None)

        self.tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.index_path = index_path
        self.meta_path = meta_path
        self.index_info_path = INDEX_INFO
        self.crops_dir = crops_dir
        self.threshold = threshold
        self.enable_head_align = enable_head_align
        self.enable_angle_tta  = enable_angle_tta
        self.enable_rerank     = enable_rerank
        self.enable_qe         = enable_qe
        ensure_dir(self.crops_dir)

        self.fused_dim = self._compute_fused_dim()
        self.metadata: List[dict] = load_metadata(self.meta_path)
        self.next_id = 1 + (max([m.get("id", 0) for m in self.metadata]) if self.metadata else 0)

        self.index, self.index_info = self._load_or_create_index()

        # GPU FAISS mirror (if available)
        self.gpu_res = None
        self.gpu_index = None
        self._maybe_init_gpu_index()

    def _compute_fused_dim(self) -> int:
        Dg = self.backbone.out_dim
        Dh = Dg; Dt = Dg
        Dcolor = COLOR_BINS[0]*COLOR_BINS[1]*COLOR_BINS[2]  # 64
        Dtex   = 4
        Dshape = 2
        Dtail  = 1
        Dear   = 6
        Dfgeo  = 6
        Dfcol  = 3
        Dband  = 24
        Dedge  = 8
        DearEdge = 2
        Dsym     = 1
        Dfanat   = 5
        Defd     = 20
        Drad     = 36
        Dhu      = 7
        Dmp      = 96
        return Dg + Dh + Dt + Dcolor + Dtex + Dshape + Dtail + Dear + Dfgeo + Dfcol + Dband + Dedge + DearEdge + Dsym + Dfanat + Defd + Drad + Dhu + Dmp

    def _migrate_index_from_metadata(self) -> faiss.Index:
        print("[info] Index dim changed; rebuilding from metadata crops…")
        index = to_cosine_index(self.fused_dim)
        added = 0
        for m in self.metadata:
            crop_path = m.get("crop")
            img = cv2.imread(crop_path) if crop_path else None
            if img is None and m.get("source"):
                img = cv2.imread(m["source"])
                if img is not None:
                    x1,y1,x2,y2 = m.get("bbox",[0,0,0,0])
                    img = img[y1:y2, x1:x2]
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]
            vec, _ = self.embed_with_features(img, (0,0,w,h), (w,h))
            index.add(vec); added += 1
        print(f"[info] Rebuilt {added} vectors.")
        return index

    def _load_or_create_index(self):
        info = load_json(self.index_info_path, {})
        if os.path.exists(self.index_path):
            idx = faiss.read_index(self.index_path)
            if idx.d != self.fused_dim:
                idx = self._migrate_index_from_metadata()
                info = {"type": "flat"}
                faiss.write_index(idx, self.index_path)
                save_json(self.index_info_path, info)
            elif not info:
                info = {"type": "flat"}
            return idx, info
        index = to_cosine_index(self.fused_dim)
        info = {"type": DEFAULT_INDEX_TYPE}
        faiss.write_index(index, self.index_path)
        save_json(self.index_info_path, info)
        return index, info

    def _maybe_init_gpu_index(self):
        if self.device == "cuda" and hasattr(faiss, "StandardGpuResources"):
            try:
                self.gpu_res = faiss.StandardGpuResources()
                self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
            except Exception:
                self.gpu_res = None
                self.gpu_index = None

    def _sync_gpu_index(self):
        if self.gpu_res is not None:
            try:
                self.gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
            except Exception:
                self.gpu_index = None

    # ---------- detection ----------
    def detect_dogs(self, image_bgr: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        dev = 0 if self.device == "cuda" else "cpu"
        res = self.yolo.predict(image_bgr, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False, device=dev)
        out = []
        for r in res:
            for b in r.boxes:
                cls_id = int(b.cls[0].item())
                label = self.class_names.get(cls_id) if isinstance(self.class_names, dict) else self.class_names[cls_id]
                if label == CLASS_NAME:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0].item())
                    out.append(((x1, y1, x2, y2), conf))
        return out

    # ---------- breed ----------
    def breed_topk(self, crop_bgr: np.ndarray, k: int = 3) -> List[str]:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        with torch.no_grad():
            t = self.tf(pil).unsqueeze(0).to(self.device)
            logits = self.cls_model(t)
            probs = torch.softmax(logits, dim=1)[0]
            idxs = torch.topk(probs, k=10).indices.cpu().numpy().tolist()
        labels = [self.imagenet_classes[i] if self.imagenet_classes else str(i) for i in idxs]
        doggy = [lbl for lbl in labels if any(w in lbl.lower() for w in [
            "dog","terrier","retriever","hound","shepherd","spaniel","bulldog","poodle","mastiff",
            "pug","husky","akita","dachshund","chow","beagle","collie","pinscher","spitz"
        ])]
        if not doggy: doggy = labels[:k]
        return doggy[:k]

    # ---------- deep embed with ANGLE TTA ----------
    def _deep_embed_tta(self, bgr: np.ndarray, angles: List[float]) -> np.ndarray:
        bgr = grabcut_foreground(bgr)
        feats = []
        with torch.no_grad():
            for ang in angles:
                img_r = rotate_bound(bgr, ang) if abs(ang) > 1e-3 else bgr
                for flip in (False, True):
                    rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if flip: pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                    for S in (224, 288):
                        batch = fivecrop_tensor(pil, S, self.device)   # [5,3,H,W]
                        feat = self.backbone(batch).detach()            # [5, D]
                        feats.append(feat)
            feats = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)  # [1,D]
        vec = feats.cpu().numpy().astype("float32")
        return l2_normalize(vec).astype("float32")

    def _region_crops(self, crop_bgr: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        H, W = crop_bgr.shape[:2]
        global_crop = crop_bgr
        head_crop = crop_bgr[:max(H//2, 1), :]
        y1 = int(H*0.35); y2 = int(H*0.85)
        x1 = int(W*0.15); x2 = int(W*0.85)
        torso_crop = crop_bgr[y1:y2, x1:x2] if (y2>y1 and x2>x1) else crop_bgr
        return global_crop, head_crop, torso_crop

    # ---------- fused embedding ----------
    def embed_with_features(self,
                            crop_bgr: np.ndarray,
                            bbox: Tuple[int,int,int,int],
                            frame_wh: Tuple[int,int]) -> Tuple[np.ndarray, Dict[str,Any]]:
        W, H = frame_wh
        bx = self._expand_square(bbox, W, H, pad=0.18)
        x1,y1,x2,y2 = bx
        crop_bgr = grabcut_foreground(crop_bgr) if crop_bgr.size else crop_bgr

        g_crop, h_crop, t_crop = self._region_crops(crop_bgr)

        # Advanced face shape descriptors
        mask = head_mask_from_crop(h_crop)
        cont = contour_from_mask(mask)
        efd  = efd_descriptor(cont, harmonics=10)       # [1,20]
        rad  = radial_shape_signature(cont, bins=36)    # [1,36]
        hum  = hu_moments(mask)                         # [1,7]

        # Eye geometry before alignment
        gray_head = cv2.cvtColor(h_crop, cv2.COLOR_BGR2GRAY) if h_crop.size>0 else np.zeros((10,10), np.uint8)
        eLx,eLy,eLr,eRx,eRy,eRr,nx,ny = detect_eyes_nose_points(gray_head)

        # Pose-robust angles
        angles = [-25, 0, 25] if self.enable_angle_tta else [0]

        # Head alignment
        h_aligned = align_head_by_eyes(h_crop, (eLx,eLy,eLr,eRx,eRy,eRr,nx,ny)) if self.enable_head_align else h_crop

        emb_g = self._deep_embed_tta(g_crop, angles)
        emb_h = self._deep_embed_tta(h_aligned, angles)     # aligned head for better angle invariance
        emb_t = self._deep_embed_tta(t_crop, angles)

        col   = color_hist_hsv(crop_bgr)
        band  = banded_color_hsv(crop_bgr)
        edgeh = edge_orientation_hist(crop_bgr)
        tex   = texture_stats(crop_bgr)
        shp   = shape_features(bx, frame_wh)
        tail  = tail_visibility_score(crop_bgr)
        earsC = ear_color_hsv(h_crop)
        earsE = ear_edge_density(h_crop)
        symm  = head_symmetry_score(h_crop)

        fgeo = np.array([[eLx if eLx>=0 else 0.5,
                          eLy if eLy>=0 else 0.5,
                          eRx if eRx>=0 else 0.5,
                          eRy if eRy>=0 else 0.5,
                          nx  if nx >=0 else 0.5,
                          ny  if ny >=0 else 0.5]], dtype="float32")
        fcol = facial_color_from_points(h_crop, (eLx,eLy,eLr,eRx,eRy,eRr,nx,ny))
        fanat = facial_anatomy_metrics((eLx,eLy,eLr,eRx,eRy,eRr,nx,ny))

        mp_desc, mp_ok = mp_face_descriptor(h_crop)  # [1,96] or zeros

        # Weights — emphasize head
        Wt = {
            "emb_g": 0.66, "emb_h": 0.27, "emb_t": 0.07,
            "col": 0.010, "band": 0.005, "edge": 0.003,
            "tex": 0.003, "shp": 0.002, "tail": 0.002,
            "earsC": 0.003, "earsE": 0.004,
            "fgeo": 0.022, "fcol": 0.012, "fanat": 0.014,
            "symm": 0.004,
            "efd": 0.012, "rad": 0.012, "hu": 0.006,
            "mp":  0.020,
        }
        wsum = float(sum(Wt.values()))

        fused = np.concatenate([
            emb_g * Wt["emb_g"],
            emb_h * Wt["emb_h"],
            emb_t * Wt["emb_t"],
            col   * Wt["col"],
            band  * Wt["band"],
            edgeh * Wt["edge"],
            tex   * Wt["tex"],
            shp   * Wt["shp"],
            tail  * Wt["tail"],
            earsC * Wt["earsC"],
            earsE * Wt["earsE"],
            fgeo  * Wt["fgeo"],
            fcol  * Wt["fcol"],
            fanat * Wt["fanat"],
            symm  * Wt["symm"],
            efd   * Wt["efd"],
            rad   * Wt["rad"],
            hum   * Wt["hu"],
            mp_desc * Wt["mp"],
        ], axis=1) / wsum

        fused = l2_normalize(fused).astype("float32")

        bundle = {
            "emb_g": emb_g, "emb_h": emb_h, "emb_t": emb_t,
            "col": col, "band": band, "edge": edgeh,
            "tex": tex, "shp": shp, "tail": float(tail[0,0]),
            "earsC": earsC, "earsE": earsE, "fgeo": fgeo, "fcol": fcol,
            "fanat": fanat, "symm": symm,
            "efd": efd, "rad": rad, "hu": hum,
            "mp": mp_desc,
            "quality_blur": float(tex[0,2]),
            "quality_entropy": float(tex[0,3])
        }
        return fused, bundle

    # ---------- breakdown + deep override ----------
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = l2_normalize(a.astype("float32")); b = l2_normalize(b.astype("float32"))
        return float((a @ b.T)[0,0])

    def rerank_scores(self, query_vec: np.ndarray, topk: int = RERANK_TOPK) -> Tuple[np.ndarray,np.ndarray]:
        k = min(topk, max(1, self.index.ntotal))
        sims, idxs = self.search(query_vec, k=k)
        if k <= 2: return sims, idxs
        q_neighbors = idxs[0].tolist()
        resc = sims.copy()
        for j, nid in enumerate(q_neighbors):
            if nid < 0: continue
            nvec = self.index.reconstruct(nid).reshape(1, -1).astype("float32")
            nsims, nidxs = self.search(nvec, k=min(10, self.index.ntotal))
            neigh_set = set(nidxs[0].tolist())
            inter = len(neigh_set.intersection(q_neighbors))
            jacc = inter / max(1, len(neigh_set.union(q_neighbors)))
            resc[0,j] = float(0.85*resc[0,j] + 0.15*jacc)
        order = np.argsort(-resc[0])
        return resc[:, order], idxs[:, order]

    def breakdown_similarity(self, A: Dict[str,Any], B: Dict[str,Any]) -> Dict[str,float]:
        s = {}
        s["deep_global"] = self.cosine(A["emb_g"], B["emb_g"])
        s["deep_head"]   = self.cosine(A["emb_h"], B["emb_h"])
        s["deep_torso"]  = self.cosine(A["emb_t"], B["emb_t"])
        s["color_hsv"]   = self.cosine(A["col"],   B["col"])
        s["band_color"]  = self.cosine(A["band"],  B["band"])
        s["edge_orient"] = self.cosine(A["edge"],  B["edge"])
        s["texture"]     = self.cosine(A["tex"],   B["tex"])
        s["shape"]       = self.cosine(A["shp"],   B["shp"])
        s["tail"]        = max(0.0, 1.0 - abs(A["tail"] - B["tail"]))
        s["ears_color"]  = self.cosine(A["earsC"], B["earsC"])
        s["ears_edge"]   = self.cosine(A["earsE"], B["earsE"])
        s["face_geom"]   = self.cosine(A["fgeo"],  B["fgeo"])
        s["face_color"]  = self.cosine(A["fcol"],  B["fcol"])
        s["face_anat"]   = self.cosine(A["fanat"], B["fanat"])
        s["symmetry"]    = max(0.0, 1.0 - abs(A["symm"] - B["symm"]))
        s["face_shape_efd"] = self.cosine(A["efd"], B["efd"])
        s["face_shape_rad"] = self.cosine(A["rad"], B["rad"])
        s["face_shape_hu"]  = self.cosine(A["hu"],  B["hu"])
        s["face_mp"]        = self.cosine(A["mp"],  B["mp"])

        deep_agg = (0.66*s["deep_global"] + 0.27*s["deep_head"] + 0.07*s["deep_torso"]) / (0.66+0.27+0.07)
        oth = (0.010*s["color_hsv"] + 0.005*s["band_color"] + 0.003*s["edge_orient"] +
               0.003*s["texture"] + 0.002*s["shape"] + 0.002*s["tail"] +
               0.003*s["ears_color"] + 0.004*s["ears_edge"] +
               0.022*s["face_geom"] + 0.012*s["face_color"] +
               0.014*s["face_anat"] + 0.004*s["symmetry"] +
               0.012*s["face_shape_efd"] + 0.012*s["face_shape_rad"] + 0.006*s["face_shape_hu"] +
               0.020*s["face_mp"])
        oth /= (0.010+0.005+0.003+0.003+0.002+0.002+0.003+0.004+0.022+0.012+0.014+0.004+0.012+0.012+0.006+0.020)

        overall = 0.93*deep_agg + 0.07*oth

        # Robust deep overrides
        if (s["deep_head"] >= 0.80 and (s["face_geom"] >= 0.72 or s["face_anat"] >= 0.70)) or s["deep_global"] >= 0.84:
            overall = max(overall, 0.88)
        if s["deep_head"] >= 0.78 and (s["face_shape_efd"] >= 0.72 or s["face_shape_rad"] >= 0.72 or s["face_shape_hu"] >= 0.70):
            overall = max(overall, 0.90)
        if s["deep_head"] >= 0.76 and s["face_mp"] >= 0.74:
            overall = max(overall, 0.90)

        s["overall"] = float(np.clip(overall, 0.0, 1.0))
        return s

    # ---------- index ops ----------
    def add_entry(self, vec: np.ndarray, meta: dict):
        self.index.add(vec)
        self.metadata.append(meta)
        self._sync_gpu_index()

    def search(self, vec: np.ndarray, k: int = 1):
        if self.index.ntotal == 0:
            return np.array([[0.0]], dtype="float32"), np.array([[-1]], dtype="int64")
        if self.gpu_index is not None:
            return self.gpu_index.search(vec, k)
        return self.index.search(vec, k)

    def save_all(self):
        ensure_dir(os.path.dirname(self.index_path))
        faiss.write_index(self.index, self.index_path)
        save_metadata(self.meta_path, self.metadata)

    def _reconstruct_all(self) -> np.ndarray:
        nt = self.index.ntotal; d  = self.index.d
        vecs = np.zeros((nt, d), dtype="float32")
        for i in range(nt):
            vecs[i, :] = self.index.reconstruct(i)
        return vecs

    def rebuild_index(self, target_type: str = "ivf", nlist: int = 4096, pq_m: int = 16, pq_bits: int = 8):
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty; add some dogs first.")
        all_vecs = self._reconstruct_all()
        d = all_vecs.shape[1]
        quantizer = faiss.IndexFlatIP(d)

        if target_type.lower() == "ivf":
            new_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        elif target_type.lower() == "ivfpq":
            new_index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("target_type must be 'ivf' or 'ivfpq'")

        ntrain = min(all_vecs.shape[0], max(10000, nlist * 40))
        idx = np.random.choice(all_vecs.shape[0], size=ntrain, replace=False)
        train_vecs = all_vecs[idx]
        new_index.train(train_vecs)
        new_index.add(all_vecs)

        self.index = new_index
        self.index_info = {"type": target_type.lower(), "nlist": nlist, "pq_m": pq_m, "pq_bits": pq_bits}
        faiss.write_index(self.index, self.index_path)
        save_json(self.index_info_path, self.index_info)
        self._sync_gpu_index()

    # ---------- registration ----------
    def register_images(self, image_paths: List[str]):
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[skip] {path} not readable"); continue
            dogs = self.detect_dogs(img)
            if not dogs:
                print(f"[info] no dog in {path}"); continue

            H, W = img.shape[:2]
            for (x1,y1,x2,y2), conf in dogs:
                bx = self._expand_square((x1,y1,x2,y2), W, H, pad=0.18)
                crop = img[bx[1]:bx[3], bx[0]:bx[2]]
                vec, bundle = self.embed_with_features(crop, bx, (W,H))
                breeds = self.breed_topk(crop, k=3)

                dog_id = self.next_id; self.next_id += 1
                ensure_dir(self.crops_dir)
                crop_path = os.path.join(CROPS_DIR, f"dog_{dog_id}.jpg")
                cv2.imwrite(crop_path, crop)

                meta = {
                    "id": dog_id,
                    "source": os.path.abspath(path),
                    "bbox": list(bx),
                    "conf": round(conf,3),
                    "crop": os.path.abspath(crop_path),
                    "breeds_top3": breeds,
                    "tail_score": round(bundle["tail"],3),
                    "ear_color_hsv": np.array(bundle["earsC"]).flatten().round(3).tolist(),
                    "ear_edge_density": np.array(bundle["earsE"]).flatten().round(3).tolist(),
                    "face_geom": np.array(bundle["fgeo"]).flatten().round(3).tolist(),
                    "face_color": np.array(bundle["fcol"]).flatten().round(3).tolist(),
                    "face_anatomy": np.array(bundle["fanat"]).flatten().round(3).tolist(),
                    "symmetry": float(bundle["symm"][0,0]),
                    "shape_feat": np.array(bundle["shp"]).flatten().round(3).tolist(),
                    "texture": np.array(bundle["tex"]).flatten().round(3).tolist(),
                    "embed_version": "v5_angle_align",
                    "backbone": self.backbone_name,
                    "ts": int(time.time()),
                }
                self.add_entry(vec, meta)
                print(f"[added] Dog-{dog_id} breeds={breeds} tail={bundle['tail']:.2f}")

        self.save_all()
        print(f"[saved] total={self.index.ntotal}")

    def _breed_adjust(self, sim: float, new_topk: List[str], known_topk: List[str]) -> float:
        if not new_topk or not known_topk: return sim
        overlap = any(a == b for a in new_topk for b in known_topk)
        return sim * (BREED_BOOST if overlap else BREED_PENALTY)

    def _quality_adapt_threshold(self, base_thr: float, bundle: Dict[str,Any]) -> float:
        blur = bundle.get("quality_blur", 0.0)      # higher is sharper (normalized)
        ent  = bundle.get("quality_entropy", 0.5)   # 0..1
        adj = 0.0
        if blur >= 0.35 and ent >= 0.55:
            adj = -0.015
        elif blur >= 0.25 and ent >= 0.50:
            adj = -0.008
        return float(np.clip(base_thr + adj, 0.70, 0.95))

    def _query_expand(self, qvec: np.ndarray, idxs: np.ndarray, sims: np.ndarray, topk: int = QE_TOPK) -> np.ndarray:
        k = min(topk, idxs.shape[1])
        if k < 2: return qvec
        ids = idxs[0, :k].tolist()
        neigh = []
        for nid in ids:
            if nid < 0: continue
            nvec = self.index.reconstruct(nid).reshape(1,-1).astype("float32")
            neigh.append(nvec)
        if not neigh: return qvec
        meanv = np.mean(np.concatenate(neigh, axis=0), axis=0, keepdims=True).astype("float32")
        qnew = l2_normalize((1.0 - QE_ALPHA)*qvec + QE_ALPHA*meanv).astype("float32")
        return qnew

    def check_images(self, image_paths: List[str], threshold: float = None, topk: int = 3):
        base_thr = threshold or self.threshold
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"[skip] {path} not readable"); continue
            dogs = self.detect_dogs(img)
            if not dogs:
                print(f"[info] no dog in {path}"); continue

            H, W = img.shape[:2]
            for (x1,y1,x2,y2), conf in dogs:
                bx = self._expand_square((x1,y1,x2,y2), W, H, pad=0.18)
                crop = img[bx[1]:bx[3], bx[0]:bx[2]]
                vec, bundle = self.embed_with_features(crop, bx, (W,H))
                breeds_new = self.breed_topk(crop, k=3)

                sims, idxs = self.search(vec, k=min(max(topk, RERANK_TOPK), max(1, self.index.ntotal)))
                if self.enable_rerank and self.index.ntotal >= 5:
                    sims, idxs = self.rerank_scores(vec, topk=sims.shape[1])

                if self.enable_qe and self.index.ntotal >= 3:
                    q2 = self._query_expand(vec, idxs, sims, topk=QE_TOPK)
                    sims2, idxs2 = self.search(q2, k=sims.shape[1])
                    if self.enable_rerank and self.index.ntotal >= 5:
                        sims2, idxs2 = self.rerank_scores(q2, topk=sims2.shape[1])
                    if sims2[0,0] > sims[0,0]:
                        sims, idxs = sims2, idxs2
                        vec = q2

                sim0, idx0 = float(sims[0,0]), int(idxs[0,0])

                thr = self._quality_adapt_threshold(base_thr, bundle)

                if idx0 >= 0:
                    known = self.metadata[idx0]
                    sim0 = self._breed_adjust(sim0, breeds_new, known.get("breeds_top3", []))

                    best_path = known.get("crop")
                    if best_path and os.path.exists(best_path):
                        best_img = cv2.imread(best_path)
                        if best_img is not None and best_img.size > 0:
                            loc = local_match_score(crop, best_img)
                            sim0 = float(np.clip(0.95*sim0 + 0.05*loc, 0.0, 1.0))
                            if sim0 < thr and (loc >= 0.58):
                                sim0 = max(sim0, 0.90)

                    tail_known = float(known.get("tail_score", 0.0))
                    if abs(tail_known - bundle["tail"]) > 0.5:
                        sim0 *= 0.995

                if idx0 >= 0 and sim0 >= thr:
                    known = self.metadata[idx0]
                    print(f"⚠️ Already captured: Dog-{known['id']} (sim={sim0:.3f} ≥ {thr:.2f}) "
                          f"breeds={known.get('breeds_top3', [])} "
                          f"tail_known={known.get('tail_score',0)} tail_new={bundle['tail']:.2f} "
                          f"source={os.path.basename(known['source'])}")
                else:
                    print(f"✅ New dog (sim={sim0:.3f} < {thr:.2f}) -> adding. breeds={breeds_new} tail={bundle['tail']:.2f}")
                    dog_id = self.next_id; self.next_id += 1
                    ensure_dir(self.crops_dir)
                    crop_path = os.path.join(CROPS_DIR, f"dog_{dog_id}.jpg")
                    cv2.imwrite(crop_path, crop)
                    meta = {
                        "id": dog_id,
                        "source": os.path.abspath(path),
                        "bbox": list(bx),
                        "conf": round(conf,3),
                        "crop": os.path.abspath(crop_path),
                        "breeds_top3": breeds_new,
                        "tail_score": round(bundle["tail"],3),
                        "ear_color_hsv": np.array(bundle["earsC"]).flatten().round(3).tolist(),
                        "ear_edge_density": np.array(bundle["earsE"]).flatten().round(3).tolist(),
                        "face_geom": np.array(bundle["fgeo"]).flatten().round(3).tolist(),
                        "face_color": np.array(bundle["fcol"]).flatten().round(3).tolist(),
                        "face_anatomy": np.array(bundle["fanat"]).flatten().round(3).tolist(),
                        "symmetry": float(bundle["symm"][0,0]),
                        "shape_feat": np.array(bundle["shp"]).flatten().round(3).tolist(),
                        "texture": np.array(bundle["tex"]).flatten().round(3).tolist(),
                        "embed_version": "v5_angle_align",
                        "backbone": self.backbone_name,
                        "ts": int(time.time()),
                    }
                    self.add_entry(vec, meta)
                    self.save_all()

    # ---------- webcam ----------
    def webcam_loop(self, cam_index=0, threshold=None):
        thr = threshold or self.threshold
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("[error] webcam not available"); return
        print("[info] webcam opened (q to quit)")
        while True:
            ok, frame = cap.read()
            if not ok: break
            H, W = frame.shape[:2]
            dogs = self.detect_dogs(frame)
            for (x1,y1,x2,y2), conf in dogs:
                bx = self._expand_square((x1,y1,x2,y2), W, H, pad=0.18)
                crop = frame[bx[1]:bx[3], bx[0]:bx[2]]
                vec, _ = self.embed_with_features(crop, bx, (W,H))
                sims, idxs = self.search(vec, k=1)
                sim, idx = float(sims[0,0]), int(idxs[0,0])
                label = f"New ({sim:.2f})"; color = (0,255,0)
                if idx >= 0 and sim >= thr:
                    known = self.metadata[idx]
                    label = f"Dog-{known['id']} ({sim:.2f})"
                    color = (0,0,255)
                cv2.rectangle(frame, (bx[0],bx[1]), (bx[2],bx[3]), color, 2)
                cv2.putText(frame, label, (bx[0], max(20,bx[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Dog ReID (angle-robust)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        cv2.destroyAllWindows()

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Street Dog Re-ID (angle-robust, head-aligned)")
    p.add_argument("--mode", choices=["register","check","webcam","rebuild_index"], required=True)
    p.add_argument("--images", nargs="*", help="image paths")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--index_path", default=INDEX_PATH)
    p.add_argument("--meta_path", default=META_PATH)
    p.add_argument("--crops_dir", default=CROPS_DIR)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--target_index", choices=["ivf","ivfpq"], default="ivf")
    p.add_argument("--nlist", type=int, default=DEFAULT_NLIST)
    p.add_argument("--pq_m", type=int, default=DEFAULT_PQ_M)
    p.add_argument("--pq_bits", type=int, default=DEFAULT_PQ_BITS)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--backbone", choices=["resnet50", "effv2s"], default="resnet50",
                   help="feature backbone for ReID embeddings")
    # toggles
    p.add_argument("--yolo_s", action="store_true", help="use yolov8s.pt for detection")
    p.add_argument("--no_head_align", action="store_true")
    p.add_argument("--no_angle_tta", action="store_true")
    p.add_argument("--no_rerank", action="store_true")
    p.add_argument("--no_qe", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    # wire custom paths (optional)
    global INDEX_PATH, META_PATH, CROPS_DIR
    INDEX_PATH = args.index_path
    META_PATH  = args.meta_path
    CROPS_DIR  = args.crops_dir

    app = DogReID(
        index_path=args.index_path,
        meta_path=args.meta_path,
        crops_dir=args.crops_dir,
        threshold=args.threshold,
        use_yolo_s=args.yolo_s,
        enable_head_align=not args.no_head_align,
        enable_angle_tta=not args.no_angle_tta,
        enable_rerank=not args.no_rerank,
        enable_qe=not args.no_qe,
        backbone_name=args.backbone
    )

    if args.mode == "register":
        if not args.images:
            print("Need --images"); return
        app.register_images(args.images)

    elif args.mode == "check":
        if not args.images:
            print("Need --images"); return
        app.check_images(args.images, threshold=args.threshold, topk=args.topk)

    elif args.mode == "webcam":
        app.webcam_loop(cam_index=args.camera, threshold=args.threshold)

    elif args.mode == "rebuild_index":
        app.rebuild_index(target_type=args.target_index, nlist=args.nlist, pq_m=args.pq_m, pq_bits=args.pq_bits)
        print(f"[ok] rebuilt index -> {args.target_index} (nlist={args.nlist}, pq_m={args.pq_m}, pq_bits={args.pq_bits})")

if __name__ == "__main__":
    main()
