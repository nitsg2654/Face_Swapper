import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Load InsightFace model
swapper = get_model('./model/inswapper_128.onnx', download=False)
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the current frame
    faces = app.get(frame)
    
    if len(faces) == 2:
        # Perform face swap (assuming 2 faces are detected)
        src_face = faces[0]
        tgt_face = faces[1]
        swapped_frame = swapper.get(frame, tgt_face, src_face, paste_back=True)
        
        # Display the swapped frame
        cv2.imshow("Face Swapping", swapped_frame)
    else:
        # If there are not exactly 2 faces, show the original frame
        cv2.imshow("Face Swap", frame)
    
    # Wait for the user to press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
