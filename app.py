import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")  # writable in Spaces
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import io
import time
import glob
from typing import List, Tuple
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ---- Import your engine ----
from dog_reid import DogReID, DEFAULT_THRESHOLD, COLOR_BINS

# ----------------------------
# Helpers
# ----------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def save_uploaded_files(files, dest_dir: str) -> List[str]:
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for uf in files:
        data = uf.read()
        out_path = os.path.join(dest_dir, uf.name)
        with open(out_path, "wb") as f:
            f.write(data)
        paths.append(out_path)
    return paths

def glob_paths(pattern: str) -> List[str]:
    if not pattern: return []
    # glob on Windows-style paths is fine
    return glob.glob(pattern)

def show_meta_card(meta: dict):
    st.caption(f"Dog-{meta.get('id', '?')} • backbone={meta.get('backbone','?')} • conf={meta.get('conf','?')}")
    cols = st.columns(2)
    crop_path = meta.get("crop")
    if crop_path and os.path.exists(crop_path):
        with cols[0]:
            st.image(crop_path, caption=os.path.basename(crop_path), use_container_width=True)
    with cols[1]:
        st.write("**Breeds top3**:", meta.get("breeds_top3", []))
        st.write("**Tail score**:", meta.get("tail_score", None))
        st.write("**Symmetry**:", meta.get("symmetry", None))
        st.write("**Head geom**:", meta.get("face_geom", None))
        st.write("**Face anatomy**:", meta.get("face_anatomy", None))
        st.write("**Texture**:", meta.get("texture", None))

def ensure_session():
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "last_logs" not in st.session_state:
        st.session_state.last_logs = []
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Dog Re-ID UI", layout="wide", page_icon="🐶")
ensure_session()
st.title("🐶 Dog Re-ID — Angle-Robust UI")

with st.sidebar:
    st.header("Model Settings")
    backbone = st.selectbox("Backbone", ["resnet50", "effv2s"], index=0, help="ResNet-50 (stronger) or EfficientNet-V2-S (lighter)")
    threshold = st.slider("Decision threshold", 0.70, 0.95, DEFAULT_THRESHOLD, 0.005,
                          help="Similarities ≥ threshold are considered matches.")
    yolo_s = st.toggle("Use YOLOv8s (bigger detector)", value=False)
    enable_head_align = st.toggle("Enable head alignment", value=True)
    enable_angle_tta = st.toggle("Enable angle TTA", value=True)
    enable_rerank = st.toggle("Enable re-ranking", value=True)
    enable_qe = st.toggle("Enable query expansion", value=True)

    st.header("Index / Metadata")
    index_path = st.text_input("FAISS index path", "dog_index.faiss")
    meta_path  = st.text_input("Metadata JSON path", "dog_metadata.json")
    crops_dir  = st.text_input("Crops directory", "crops")

    colx1, colx2 = st.columns(2)
    with colx1:
        if st.button("Load / Init Engine", use_container_width=True):
            st.session_state.engine = DogReID(
                index_path=index_path,
                meta_path=meta_path,
                crops_dir=crops_dir,
                threshold=threshold,
                use_yolo_s=yolo_s,
                enable_head_align=enable_head_align,
                enable_angle_tta=enable_angle_tta,
                enable_rerank=enable_rerank,
                enable_qe=enable_qe,
                backbone_name=backbone
            )
            st.success(f"Engine loaded: backbone={backbone}, index_dim={st.session_state.engine.fused_dim}")

    with colx2:
        if st.button("Save Index & Metadata", use_container_width=True, disabled=(st.session_state.engine is None)):
            if st.session_state.engine:
                st.session_state.engine.save_all()
                st.success("Saved FAISS index & metadata.")

    st.markdown("---")
    st.caption("Tip: A/B test by using different index/meta file names for each backbone.")

tabs = st.tabs(["📥 Register", "🔎 Check / Search", "🎥 Webcam", "🗂️ Index Manager", "📚 Metadata Browser"])

# -----------------------------------
# Tab: Register
# -----------------------------------
with tabs[0]:
    st.subheader("Register images (add to index)")
    if st.session_state.engine is None:
        st.info("Load the engine first from the sidebar.")
    else:
        upl = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)
        glob_pat = st.text_input("...or use a path pattern (e.g., D:\\\\Dog-identify\\\\dogs\\\\*.jpg)", "")
        colr1, colr2 = st.columns(2)
        with colr1:
            auto_glob = st.button("Register from pattern", use_container_width=True)
        with colr2:
            from_upload = st.button("Register uploaded files", use_container_width=True)

        logs = st.empty()
        gallery = st.container()

        if auto_glob:
            paths = glob_paths(glob_pat)
            if not paths:
                st.warning("No files found for the given pattern.")
            else:
                st.write(f"Found **{len(paths)}** files.")
                # Run registration
                st.session_state.engine.register_images(paths)
                st.success("Registration complete.")
        if from_upload and upl:
            tmp_dir = os.path.join("tmp_uploads", time.strftime("%Y%m%d_%H%M%S"))
            paths = save_uploaded_files(upl, tmp_dir)
            st.write(f"Saved **{len(paths)}** uploaded files to {tmp_dir}.")
            st.session_state.engine.register_images(paths)
            st.success("Registration complete.")

# -----------------------------------
# Tab: Check / Search
# -----------------------------------
with tabs[1]:
    st.subheader("Check (search & optionally add)")
    if st.session_state.engine is None:
        st.info("Load the engine first from the sidebar.")
    else:
        colc0, colc1, colc2 = st.columns([1,1,1])
        with colc0:
            topk = st.number_input("Top-K", min_value=1, max_value=50, value=5, step=1)
        with colc1:
            add_if_new = st.toggle("Add as new if below threshold", value=False,
                                   help="When OFF: just search. When ON: add below-threshold as new entries.")
        with colc2:
            use_glob = st.toggle("Use path pattern instead of upload", value=False)

        paths = []
        if use_glob:
            pattern = st.text_input("Path pattern (e.g., D:\\\\Dog-identify\\\\queries\\\\*.jpg)", "")
            if st.button("Run Check on Pattern", use_container_width=True):
                paths = glob_paths(pattern)
                if not paths:
                    st.warning("No files found.")
        else:
            upl2 = st.file_uploader("Upload query images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="check_upl")
            if st.button("Run Check on Uploaded", use_container_width=True):
                if upl2:
                    tmp_dir = os.path.join("tmp_queries", time.strftime("%Y%m%d_%H%M%S"))
                    paths = save_uploaded_files(upl2, tmp_dir)
                else:
                    st.warning("Please upload images first.")

        if paths:
            eng: DogReID = st.session_state.engine
            st.write(f"Processing **{len(paths)}** image(s)...")
            for p in paths:
                img = cv2.imread(p)
                if img is None:
                    st.warning(f"Skip (not readable): {p}")
                    continue

                dogs = eng.detect_dogs(img)
                if not dogs:
                    st.info(f"No dog found: {os.path.basename(p)}")
                    continue

                for (x1,y1,x2,y2), conf in dogs:
                    bx = (x1, y1, x2, y2)
                    crop = img[y1:y2, x1:x2].copy()
                    vec, bundle = eng.embed_with_features(crop, bx, (img.shape[1], img.shape[0]))

                    sims, idxs = eng.search(vec, k=min(max(topk, 10), max(1, eng.index.ntotal)))
                    if eng.enable_rerank and eng.index.ntotal >= 5:
                        sims, idxs = eng.rerank_scores(vec, topk=sims.shape[1])

                    sim0, idx0 = float(sims[0,0]), int(idxs[0,0])
                    thr = eng._quality_adapt_threshold(eng.threshold, bundle)

                    # UI display
                    st.markdown(f"#### Query: {os.path.basename(p)}")
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption=f"Detected crop (conf={conf:.2f})", use_container_width=True)

                    # Show neighbors
                    st.write(f"Top-{topk} neighbors (cosine sim):")
                    ncols = st.columns(min(topk, 5))
                    for j in range(min(topk, idxs.shape[1])):
                        nid = int(idxs[0, j])
                        if nid < 0: continue
                        meta = eng.metadata[nid]
                        with ncols[j % len(ncols)]:
                            cp = meta.get("crop")
                            simv = float(sims[0, j])
                            st.caption(f"ID {meta.get('id','?')} • sim={simv:.3f}")
                            if cp and os.path.exists(cp):
                                st.image(cp, use_container_width=True)

                    # Decision + optional add
                    if idx0 >= 0 and sim0 >= thr:
                        meta = eng.metadata[idx0]
                        st.success(f"Match: Dog-{meta['id']} (sim={sim0:.3f} ≥ thr={thr:.2f})")
                    else:
                        st.warning(f"No strong match (top1={sim0:.3f} < thr={thr:.2f}).")
                        if add_if_new:
                            # Add to index as new entry
                            dog_id = eng.next_id; eng.next_id += 1
                            os.makedirs(eng.crops_dir, exist_ok=True)
                            crop_path = os.path.join(eng.crops_dir, f"dog_{dog_id}.jpg")
                            cv2.imwrite(crop_path, crop)
                            breeds_new = eng.breed_topk(crop, k=3)
                            meta = {
                                "id": dog_id,
                                "source": os.path.abspath(p),
                                "bbox": [x1,y1,x2,y2],
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
                                "backbone": eng.backbone_name,
                                "ts": int(time.time()),
                            }
                            eng.add_entry(vec, meta)
                            eng.save_all()
                            st.success(f"Added as new: Dog-{dog_id}")

# -----------------------------------
# Tab: Webcam
# -----------------------------------
with tabs[2]:
    st.subheader("Webcam (experimental)")
    if st.session_state.engine is None:
        st.info("Load the engine first from the sidebar.")
    else:
        cam_index = st.number_input("Camera index", min_value=0, max_value=8, value=0, step=1)
        colw1, colw2 = st.columns(2)
        start = colw1.button("Start Webcam")
        stop  = colw2.button("Stop Webcam")

        if start:
            st.session_state.webcam_running = True
        if stop:
            st.session_state.webcam_running = False

        frame_holder = st.empty()
        info_holder = st.empty()

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error("Webcam not available.")
                st.session_state.webcam_running = False
            else:
                info_holder.info("Press Stop Webcam to end.")
                eng: DogReID = st.session_state.engine
                while st.session_state.webcam_running:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    H, W = frame.shape[:2]
                    dogs = eng.detect_dogs(frame)
                    for (x1,y1,x2,y2), conf in dogs:
                        bx = (x1,y1,x2,y2)
                        crop = frame[y1:y2, x1:x2]
                        vec, _ = eng.embed_with_features(crop, bx, (W,H))
                        sims, idxs = eng.search(vec, k=1)
                        sim, idx = float(sims[0,0]), int(idxs[0,0])
                        label = f"New ({sim:.2f})"; color = (0,255,0)
                        if idx >= 0 and sim >= eng.threshold:
                            known = eng.metadata[idx]
                            label = f"Dog-{known['id']} ({sim:.2f})"
                            color = (0,0,255)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                        cv2.putText(frame, label, (x1, max(20,y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_holder.image(frame_rgb, use_container_width=True)
                cap.release()

# -----------------------------------
# Tab: Index Manager
# -----------------------------------
with tabs[3]:
    st.subheader("Index Manager (rebuild for speed/memory)")
    if st.session_state.engine is None:
        st.info("Load the engine first from the sidebar.")
    else:
        eng: DogReID = st.session_state.engine
        st.write(f"**Vectors in index:** {eng.index.ntotal}")
        st.write(f"**Current index type:** {eng.index_info.get('type','flat')}, **dim:** {eng.index.d}")

        target = st.selectbox("Target index", ["ivf", "ivfpq"], index=0)
        nlist = st.number_input("nlist (IVF clusters)", min_value=256, max_value=65536, value=4096, step=256)
        pq_m = st.number_input("IVFPQ M (subvectors)", min_value=4, max_value=64, value=16, step=1)
        pq_bits = st.selectbox("IVFPQ bits per subvector", [4, 6, 8], index=2)

        if st.button("Rebuild Index", type="primary"):
            if eng.index.ntotal == 0:
                st.warning("Index is empty; register some images first.")
            else:
                with st.status("Rebuilding index...", expanded=True) as status:
                    eng.rebuild_index(target_type=target, nlist=int(nlist), pq_m=int(pq_m), pq_bits=int(pq_bits))
                    st.write(f"Rebuilt to {target} (nlist={nlist}, pq_m={pq_m}, pq_bits={pq_bits})")
                    status.update(label="Done!", state="complete")
                st.success("Index rebuild complete.")

# -----------------------------------
# Tab: Metadata Browser
# -----------------------------------
with tabs[4]:
    st.subheader("Browse Metadata")
    if st.session_state.engine is None:
        st.info("Load the engine first from the sidebar.")
    else:
        eng: DogReID = st.session_state.engine
        st.write(f"Total entries: **{len(eng.metadata)}**")
        # Simple gallery
        grid = st.columns(4)
        for i, meta in enumerate(eng.metadata):
            with grid[i % 4]:
                show_meta_card(meta)
