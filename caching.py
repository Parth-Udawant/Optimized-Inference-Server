# !pip install -q fastapi uvicorn nest_asyncio python-multipart

import io
import time
import hashlib
import threading
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import uvicorn
import nest_asyncio
import requests

nest_asyncio.apply()

PORT = 8001
session = ort.InferenceSession("mobilenetv3_static_int8.onnx", providers=["CPUExecutionProvider"])
cache = {}
app = FastAPI()

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    key = hashlib.sha256(contents).hexdigest()

    if key in cache:
        result = dict(cache[key])
        result["cache_hit"] = True
        return result

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).numpy()
    start = time.perf_counter()
    logits = session.run(["logits"], {"pixel_values": tensor})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    pred_idx = int(np.argmax(logits, axis=-1)[0])

    result = {"predicted_class": class_labels[pred_idx], "inference_ms": round(elapsed_ms, 2)}
    cache[key] = result
    out = dict(result)
    out["cache_hit"] = False
    return out

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)
print(f"Server running (with caching) at http://127.0.0.1:{PORT}")

images[0].save("test_dog.png")
for i in range(2):
    with open("test_dog.png", "rb") as f:
        resp = requests.post(f"http://127.0.0.1:{PORT}/predict", files={"file": f})
    print(f"Request {i+1}: {resp.json()}")
