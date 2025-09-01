# app.py
import os, io, time, shutil, tempfile, threading
from typing import List, Optional, Literal, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cv2
import numpy as np

# Import your existing modules
from dog_reid import DogReID, DEFAULT_THRESHOLD, DEFAULT_NLIST, DEFAULT_PQ_M, DEFAULT_PQ_BITS, CROPS_DIR, INDEX_PATH, META_PATH

# ----------------------------
# Config & Globals
# ----------------------------
# Storage: keep defaults or override via env
INDEX_PATH_CFG = os.getenv("DOG_INDEX_PATH", INDEX_PATH)
META_PATH_CFG  = os.getenv("DOG_META_PATH",  META_PATH)
CROPS_DIR_CFG  = os.getenv("DOG_CROPS_DIR",  CROPS_DIR)

os.makedirs(CROPS_DIR_CFG, exist_ok=True)
UPLOAD_DIR = os.getenv("DOG_UPLOAD_DIR", os.path.abspath("uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Thread-safety for index mutations
index_lock = threading.Lock()

# One shared ReID engine
ENGINE = DogReID(
    index_path=INDEX_PATH_CFG,
    meta_path=META_PATH_CFG,
    crops_dir=CROPS_DIR_CFG,
    threshold=float(os.getenv("DOG_DEFAULT_THRESHOLD", DEFAULT_THRESHOLD)),
    use_yolo_s=bool(int(os.getenv("DOG_USE_YOLO_S", "0"))),
    enable_head_align=bool(int(os.getenv("DOG_ENABLE_HEAD_ALIGN", "1"))),
    enable_angle_tta=bool(int(os.getenv("DOG_ENABLE_ANGLE_TTA", "1"))),
    enable_rerank=bool(int(os.getenv("DOG_ENABLE_RERANK", "1"))),
    enable_qe=bool(int(os.getenv("DOG_ENABLE_QE", "1"))),
    backbone_name=os.getenv("DOG_BACKBONE", "resnet50"),
)

# ----------------------------
# FastAPI setup
# ----------------------------
app = FastAPI(
    title="Dog ReID API",
    version="1.0.0",
    description="Angle-robust street dog re-identification (ResNet-50 / EfficientNet-V2-S) with FAISS index and neat JSON responses.",
)

# CORS (adjust for your frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("DOG_CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Schemas
# ----------------------------
class RegisterResult(BaseModel):
    added_count: int
    total_index: int
    items: List[Dict[str, Any]] = Field(
        description="Per-crop registration results (Dog ID, breeds, tail score, source)."
    )

class CheckHit(BaseModel):
    is_known: bool
    similarity: float
    threshold: float
    matched_id: Optional[int] = None
    matched_meta: Optional[Dict[str, Any]] = None
    breeds_new: List[str]
    tail_new: float
    decision: Literal["known", "new"]

class CheckItem(BaseModel):
    source_file: str
    detections: int
    results: List[CheckHit]

class CheckResponse(BaseModel):
    items: List[CheckItem]
    total_new_added: int
    total_index: int

class StatsResponse(BaseModel):
    index_total: int
    meta_total: int
    backbone: str
    threshold: float
    index_info: Dict[str, Any]

class RebuildRequest(BaseModel):
    target_index: Literal["ivf", "ivfpq"] = "ivf"
    nlist: int = DEFAULT_NLIST
    pq_m: int = DEFAULT_PQ_M
    pq_bits: int = DEFAULT_PQ_BITS

class RebuildResponse(BaseModel):
    ok: bool
    index_info: Dict[str, Any]
    total_index: int

# ----------------------------
# Helpers
# ----------------------------
def _save_upload_to_disk(up: UploadFile, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    # sanitize name
    fname = f"{int(time.time()*1000)}_{os.path.basename(up.filename or 'image')}"
    out_path = os.path.join(dest_dir, fname)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(up.file, f)
    return out_path

def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None or img.size == 0:
        raise ValueError(f"Cannot read image: {path}")
    return img

# A minimally adapted checker that returns structures instead of prints.
def _analyze_and_optionally_add(img_path: str,
                                threshold: Optional[float],
                                topk: int) -> CheckItem:
    img = _load_bgr(img_path)
    dogs = ENGINE.detect_dogs(img)
    out_results: List[CheckHit] = []

    if not dogs:
        return CheckItem(source_file=os.path.basename(img_path), detections=0, results=[])

    H, W = img.shape[:2]
    base_thr = threshold or ENGINE.threshold
    new_added = 0

    for (x1, y1, x2, y2), conf in dogs:
        bx = ENGINE._expand_square((x1, y1, x2, y2), W, H, pad=0.18)
        crop = img[bx[1]:bx[3], bx[0]:bx[2]]

        vec, bundle = ENGINE.embed_with_features(crop, bx, (W, H))
        breeds_new = ENGINE.breed_topk(crop, k=3)

        # Search (with rerank + QE like CLI)
        with index_lock:
            sims, idxs = ENGINE.search(vec, k=min(max(topk, 10), max(1, ENGINE.index.ntotal)))
        if ENGINE.enable_rerank and ENGINE.index.ntotal >= 5:
            sims, idxs = ENGINE.rerank_scores(vec, topk=sims.shape[1])

        if ENGINE.enable_qe and ENGINE.index.ntotal >= 3:
            q2 = ENGINE._query_expand(vec, idxs, sims)
            with index_lock:
                sims2, idxs2 = ENGINE.search(q2, k=sims.shape[1])
            if ENGINE.enable_rerank and ENGINE.index.ntotal >= 5:
                sims2, idxs2 = ENGINE.rerank_scores(q2, topk=sims2.shape[1])
            if sims2[0, 0] > sims[0, 0]:
                sims, idxs = sims2, idxs2
                vec = q2

        sim0 = float(sims[0, 0])
        idx0 = int(idxs[0, 0]) if sims.size else -1

        thr = ENGINE._quality_adapt_threshold(base_thr, bundle)

        matched_meta = None
        decision = "new"
        is_known = False

        if idx0 >= 0:
            known = ENGINE.metadata[idx0]
            # Breed-aware adjust
            sim0 = ENGINE._breed_adjust(sim0, breeds_new, known.get("breeds_top3", []))

            # Local match bonus with saved crop if exists
            best_path = known.get("crop")
            if best_path and os.path.exists(best_path):
                best_img = cv2.imread(best_path)
                if best_img is not None and best_img.size > 0:
                    loc = ENGINE.local_match_score(crop, best_img)  # type: ignore (bound method available)
                    sim0 = float(np.clip(0.95 * sim0 + 0.05 * loc, 0.0, 1.0))
                    if sim0 < thr and (loc >= 0.58):
                        sim0 = max(sim0, 0.90)

            # Tail penalty for extreme mismatch
            tail_known = float(known.get("tail_score", 0.0))
            if abs(tail_known - bundle["tail"]) > 0.5:
                sim0 *= 0.995

            if sim0 >= thr:
                is_known = True
                decision = "known"
                matched_meta = {
                    "id": known["id"],
                    "breeds_top3": known.get("breeds_top3", []),
                    "source": os.path.basename(known.get("source", "")),
                    "crop": os.path.basename(known.get("crop", "")),
                    "tail_score": known.get("tail_score", None),
                    "ts": known.get("ts", None),
                }

        if not is_known:
            # Add as new
            with index_lock:
                dog_id = ENGINE.next_id
                ENGINE.next_id += 1
                os.makedirs(ENGINE.crops_dir, exist_ok=True)
                crop_path = os.path.join(ENGINE.crops_dir, f"dog_{dog_id}.jpg")
                cv2.imwrite(crop_path, crop)
                meta = {
                    "id": dog_id,
                    "source": os.path.abspath(img_path),
                    "bbox": list(bx),
                    "conf": round(conf, 3),
                    "crop": os.path.abspath(crop_path),
                    "breeds_top3": breeds_new,
                    "tail_score": float(np.round(bundle["tail"], 3)),
                    "ear_color_hsv": np.array(bundle["earsC"]).flatten().round(3).tolist(),
                    "ear_edge_density": np.array(bundle["earsE"]).flatten().round(3).tolist(),
                    "face_geom": np.array(bundle["fgeo"]).flatten().round(3).tolist(),
                    "face_color": np.array(bundle["fcol"]).flatten().round(3).tolist(),
                    "face_anatomy": np.array(bundle["fanat"]).flatten().round(3).tolist(),
                    "symmetry": float(bundle["symm"][0, 0]),
                    "shape_feat": np.array(bundle["shp"]).flatten().round(3).tolist(),
                    "texture": np.array(bundle["tex"]).flatten().round(3).tolist(),
                    "embed_version": "v5_angle_align",
                    "backbone": ENGINE.backbone_name,
                    "ts": int(time.time()),
                }
                ENGINE.add_entry(vec, meta)
                ENGINE.save_all()

            matched_meta = {"id": dog_id, "crop": f"dog_{dog_id}.jpg"}
            decision = "new"

        out_results.append(CheckHit(
            is_known=is_known,
            similarity=float(round(sim0, 4)),
            threshold=float(round(thr, 2)),
            matched_id=(matched_meta["id"] if matched_meta else None),
            matched_meta=matched_meta,
            breeds_new=breeds_new,
            tail_new=float(round(float(bundle["tail"]), 3)),
            decision=decision,  # "known" or "new"
        ))

    return CheckItem(
        source_file=os.path.basename(img_path),
        detections=len(dogs),
        results=out_results
    )

# Monkey patch for local_match_score reference used above
def _local_match_score(a_bgr, b_bgr) -> float:
    return ENGINE.local_match_score(a_bgr, b_bgr)  # method exists on class
ENGINE.local_match_score = _local_match_score  # type: ignore

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "engine_device": ENGINE.device}

@app.get("/stats", response_model=StatsResponse)
def stats():
    return StatsResponse(
        index_total=int(ENGINE.index.ntotal),
        meta_total=len(ENGINE.metadata),
        backbone=ENGINE.backbone_name,
        threshold=float(ENGINE.threshold),
        index_info=ENGINE.index_info,
    )

@app.post("/register", response_model=RegisterResult)
async def register(
    files: List[UploadFile] = File(..., description="One or more images containing dogs"),
):
    tmpdir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    added_items = []
    added_count = 0

    try:
        for f in files:
            path = _save_upload_to_disk(f, tmpdir)
            img = _load_bgr(path)
            dogs = ENGINE.detect_dogs(img)

            if not dogs:
                added_items.append({
                    "source": os.path.basename(path),
                    "message": "no dog detected",
                    "added": 0
                })
                continue

            H, W = img.shape[:2]
            for (x1, y1, x2, y2), conf in dogs:
                bx = ENGINE._expand_square((x1, y1, x2, y2), W, H, pad=0.18)
                crop = img[bx[1]:bx[3], bx[0]:bx[2]]
                vec, bundle = ENGINE.embed_with_features(crop, bx, (W, H))
                breeds = ENGINE.breed_topk(crop, k=3)

                with index_lock:
                    dog_id = ENGINE.next_id
                    ENGINE.next_id += 1
                    os.makedirs(ENGINE.crops_dir, exist_ok=True)
                    crop_path = os.path.join(ENGINE.crops_dir, f"dog_{dog_id}.jpg")
                    cv2.imwrite(crop_path, crop)
                    meta = {
                        "id": dog_id,
                        "source": os.path.abspath(path),
                        "bbox": list(bx),
                        "conf": round(conf, 3),
                        "crop": os.path.abspath(crop_path),
                        "breeds_top3": breeds,
                        "tail_score": float(round(bundle["tail"], 3)),
                        "ear_color_hsv": np.array(bundle["earsC"]).flatten().round(3).tolist(),
                        "ear_edge_density": np.array(bundle["earsE"]).flatten().round(3).tolist(),
                        "face_geom": np.array(bundle["fgeo"]).flatten().round(3).tolist(),
                        "face_color": np.array(bundle["fcol"]).flatten().round(3).tolist(),
                        "face_anatomy": np.array(bundle["fanat"]).flatten().round(3).tolist(),
                        "symmetry": float(bundle["symm"][0, 0]),
                        "shape_feat": np.array(bundle["shp"]).flatten().round(3).tolist(),
                        "texture": np.array(bundle["tex"]).flatten().round(3).tolist(),
                        "embed_version": "v5_angle_align",
                        "backbone": ENGINE.backbone_name,
                        "ts": int(time.time()),
                    }
                    ENGINE.add_entry(vec, meta)
                    ENGINE.save_all()

                added_items.append({
                    "source": os.path.basename(path),
                    "dog_id": dog_id,
                    "breeds_top3": breeds,
                    "tail_score": float(round(bundle["tail"], 3)),
                    "bbox": list(map(int, bx)),
                })
                added_count += 1

        return RegisterResult(
            added_count=added_count,
            total_index=int(ENGINE.index.ntotal),
            items=added_items
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/check", response_model=CheckResponse)
async def check(
    files: List[UploadFile] = File(..., description="One or more images to check"),
    threshold: Optional[float] = Form(None),
    topk: int = Form(3),
):
    if topk <= 0:
        raise HTTPException(status_code=400, detail="topk must be >= 1")

    tmpdir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    items: List[CheckItem] = []
    total_new_added = 0

    try:
        for f in files:
            path = _save_upload_to_disk(f, tmpdir)
            item = _analyze_and_optionally_add(path, threshold, topk)
            items.append(item)
            # count new
            for r in item.results:
                if r.decision == "new":
                    total_new_added += 1

        return CheckResponse(
            items=items,
            total_new_added=total_new_added,
            total_index=int(ENGINE.index.ntotal)
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@app.post("/rebuild-index", response_model=RebuildResponse)
def rebuild_index(req: RebuildRequest):
    with index_lock:
        ENGINE.rebuild_index(
            target_type=req.target_index,
            nlist=req.nlist,
            pq_m=req.pq_m,
            pq_bits=req.pq_bits
        )
    return RebuildResponse(
        ok=True,
        index_info=ENGINE.index_info,
        total_index=int(ENGINE.index.ntotal),
    )
