# !pip install -q fastapi uvicorn nest_asyncio python-multipart

import io
import threading
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import uvicorn
import nest_asyncio
import requests

nest_asyncio.apply()

app = FastAPI()
session = ort.InferenceSession("mobilenetv3_static_int8.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).numpy()
    start = time.perf_counter()
    logits = session.run(["logits"], {"pixel_values": tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    pred_idx = int(np.argmax(logits, axis=-1)[0])
    return {"predicted_class": class_labels[pred_idx], "inference_ms": round(elapsed_ms, 2)}

def run_server():
    server.run()

if "server_thread" in globals() and server_thread.is_alive():
    print("Stopping previous server...")
    server.should_exit = True
    server_thread.join(timeout=5)

config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
server = uvicorn.Server(config)
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)
print("Server running at http://127.0.0.1:8000")

images[0].save("test_dog.jpg")
with open("test_dog.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/predict", files={"file": f})
print(resp.status_code, resp.json())
