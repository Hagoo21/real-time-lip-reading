import torch
import numpy as np
import cv2
from collections import deque

from mouth_extractor import extract_mouth_region
from camera import get_camera

# Import CHAPLIN model utils
from chaplin.benchmarks.LRS3.models.LRS3_V_WER19_1 import load_model, decode

model = load_model('chaplin/benchmarks/LRS3/models/LRS3_V_WER19.1.ckpt')
model.eval()

cap = get_camera(1)  # Camo webcam
frame_buffer = deque(maxlen=29)

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0  # shape (112, 112)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    mouth = extract_mouth_region(frame)
    if mouth is None:
        continue

    processed = preprocess(mouth)
    frame_buffer.append(processed)

    if len(frame_buffer) == 29:
        input_tensor = np.stack(frame_buffer)[np.newaxis, ...]  # (1, 29, 112, 112)
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            transcript = decode(output)

        print("🧠 Transcribed:", transcript)
        cv2.putText(frame, transcript, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Lip Reader", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
