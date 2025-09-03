import streamlit as st
import tempfile
import os
import cv2
import io
import contextlib
from dog_reid import DogReID, DEFAULT_THRESHOLD

st.set_page_config(page_title="Dog ReID Streamlit", layout="centered")
st.title("Dog ReID: Register & Check")

@st.cache_resource
def get_pipeline():
    return DogReID(
        backbone_name=st.session_state.get("backbone", "resnet50"),
        threshold=st.session_state.get("threshold", DEFAULT_THRESHOLD)
    )

backbone = st.selectbox("Select backbone", ["resnet50", "effv2s"], index=0, key="backbone")
threshold = st.slider("Set threshold", min_value=0.7, max_value=0.99, value=DEFAULT_THRESHOLD, step=0.01, key="threshold")
pipeline = get_pipeline()
mode = st.radio("Choose mode", ["Check", "Register"])

uploaded_files = st.file_uploader(
    "Upload dog images (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    temp_paths = []
    for file in uploaded_files:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(file.read())
        tfile.close()
        temp_paths.append(tfile.name)

    def show_check_results(results):
        if not results:
            st.warning("No results returned.")
            return
        for res in results:
            if isinstance(res, dict):
                st.markdown(f"**Matched Dog**: ID `{res.get('id', '-')}` | Score: `{res.get('score', '-')}`")
                if 'meta' in res:
                    st.json(res['meta'])
            elif isinstance(res, str) and res.startswith("No match"):
                st.info(res)
            else:
                st.text(str(res))

    def show_register_results(results):
        if not results:
            st.warning("No results returned.")
            return
        for res in results:
            if isinstance(res, dict):
                if res.get('status') == 'added':
                    st.success(f"New dog added: ID `{res.get('id', '-')}`")
                    if 'meta' in res:
                        st.json(res['meta'])
                elif res.get('status') == 'exists':
                    st.info(f"Dog already exists: ID `{res.get('id', '-')}`")
                    if 'meta' in res:
                        st.json(res['meta'])
                else:
                    st.text(str(res))
            elif isinstance(res, str):
                st.text(res)

    # --- Capture CLI output ---
    cli_output = io.StringIO()
    with contextlib.redirect_stdout(cli_output):
        if mode == "Check":
            st.subheader("Checking uploaded images...")
            try:
                results = pipeline.check_images(temp_paths, threshold=threshold, topk=3)
            except Exception as e:
                results = [f"Error: {e}"]
            show_check_results(results)
            st.success("Check complete.")
        elif mode == "Register":
            st.subheader("Registering new dogs...")
            try:
                results = pipeline.register_images(temp_paths)
            except Exception as e:
                results = [f"Error: {e}"]
            show_register_results(results)
            st.success("Registration complete.")

    # Show CLI output in UI
    st.markdown("### CLI Output")
    st.code(cli_output.getvalue())

    # Clean up temp files
    for path in temp_paths:
        os.remove(path)

st.markdown("---")
if st.button("Show Metadata"):
    import json
    meta_path = os.path.abspath("dog_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        st.json(metadata)
    else:
        st.warning("No metadata found.")