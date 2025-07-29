import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile

@st.cache_resource
def load_app_and_swapper():
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True)
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, swapper

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def swap_faces(src_img, tgt_img, app, swapper):
    src_faces = app.get(src_img)
    tgt_faces = app.get(tgt_img)
    if len(src_faces) == 0 or len(tgt_faces) == 0:
        raise Exception("No face detected in one or both images.")
    src_face = src_faces[0]
    result_img = tgt_img.copy()
    for face in tgt_faces:
        result_img = swapper.get(result_img, face, src_face, paste_back=True)
    return result_img

# Streamlit UI
st.title("InsightFace Face Swapper")

col1, col2 = st.columns(2)
with col1:
    src_file = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
with col2:
    tgt_file = st.file_uploader("Upload Target Face", type=["jpg", "jpeg", "png"])

if st.button("Swap Face"):
    if src_file and tgt_file:
        src_img = pil_to_cv2(Image.open(src_file).convert("RGB"))
        tgt_img = pil_to_cv2(Image.open(tgt_file).convert("RGB"))
        app, swapper = load_app_and_swapper()
        try:
            result_img = swap_faces(src_img, tgt_img, app, swapper)
            pil_result_img = cv2_to_pil(result_img)
            st.image(pil_result_img, caption="Swapped Image", use_container_width=True)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                pil_result_img.save(tmp_file.name, format="JPEG")
                tmp_file.seek(0)
                st.download_button(
                    label="Download Swapped Image",
                    data=tmp_file.read(),
                    file_name="swapped_image.jpg",
                    mime="image/jpeg"
                )
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload both source and target images.")
