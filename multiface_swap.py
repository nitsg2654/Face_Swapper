import streamlit as st
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import cv2
import numpy as np
from PIL import Image

@st.cache_resource
def load_models():
    swapper = get_model('./model/inswapper_128.onnx', download=False)
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, swapper

def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def draw_faces(image, faces):
    img = image.copy()
    for i, face in enumerate(faces):
        box = face.bbox.astype(int)
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (0,255,0), 2)
        cv2.putText(img, f'{i+1}', tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return img

st.title("Select and Swap Face with InsightFace")

col1, col2 = st.columns(2)
with col1:
    src_file = st.file_uploader("Upload Source Face", type=["jpg", "jpeg", "png"])
with col2:
    tgt_file = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])

if src_file and tgt_file:
    app, swapper = load_models()
    src_img = pil_to_cv2(Image.open(src_file).convert("RGB"))
    tgt_img = pil_to_cv2(Image.open(tgt_file).convert("RGB"))
    src_faces = app.get(src_img)
    tgt_faces = app.get(tgt_img)
    if len(src_faces) == 0 or len(tgt_faces) == 0:
        st.error("No face detected in one or both images.")
    else:
        st.subheader("Detected Faces in Target Image")
        tgt_img_with_boxes = draw_faces(tgt_img, tgt_faces)
        st.image(cv2_to_pil(tgt_img_with_boxes), caption="Target Image with Face Numbers", use_container_width=True)
        index = st.radio("Select face number to swap", list(range(1, len(tgt_faces)+1))) - 1
        if st.button("Swap Selected Face"):
            src_face = src_faces[0]
            selected_face = tgt_faces[index]
            result_img = swapper.get(tgt_img, selected_face, src_face, paste_back=True)
            st.image(cv2_to_pil(result_img), caption="Swapped Image", use_container_width=True)