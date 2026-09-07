# !pip install -q fastapi uvicorn nest_asyncio python-multipart

import io
import time
import hashlib
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
import uvicorn
import nest_asyncio
import requests

nest_asyncio.apply()

PORT = 8002

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 2
session = ort.InferenceSession(
    "mobilenetv3_static_int8.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"]
)

cache = {}
app = FastAPI()

BATCH_WINDOW_S = 0.02
MAX_BATCH_SIZE = 16
request_queue = asyncio.Queue()

async def batch_worker():
    while True:
        tensor, future = await request_queue.get()
        batch_tensors = [tensor]
        batch_futures = [future]

        deadline = time.monotonic() + BATCH_WINDOW_S
        while len(batch_tensors) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                t, f = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                batch_tensors.append(t)
                batch_futures.append(f)
            except asyncio.TimeoutError:
                break

        batch_input = np.concatenate(batch_tensors, axis=0)
        start = time.perf_counter()
        logits = session.run(["logits"], {"pixel_values": batch_input})[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        for i, fut in enumerate(batch_futures):
            pred_idx = int(np.argmax(logits[i]))
            fut.set_result({
                "predicted_class": class_labels[pred_idx],
                "batch_size": len(batch_tensors),
                "batch_inference_ms": round(elapsed_ms, 2),
            })

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())

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

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    await request_queue.put((tensor, future))
    result = await future

    cache[key] = result
    out = dict(result)
    out["cache_hit"] = False
    return out

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)
print(f"Batching server running at http://127.0.0.1:{PORT}")

augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])

N = 250
test_files = []
for i in range(N):
    base_img = images[i % len(images)]
    aug_img = augment(base_img)
    path = f"loadtest_{i}.png"
    aug_img.save(path)
    test_files.append(path)
print(f"Generated {N} distinct test images")

def load_test(port, files, max_workers=20):
    def send(path):
        with open(path, "rb") as f:
            r = requests.post(f"http://127.0.0.1:{port}/predict", files={"file": f})
        return r.json()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(send, files))
    total_s = time.perf_counter() - start
    return total_s, len(files) / total_s

print("\nLoad test WITHOUT batching (Phase 3 server, port 8001):")
total_before, throughput_before = load_test(8001, test_files)
print(f"  {N} requests in {total_before:.2f}s -> {throughput_before:.1f} req/s")

print("\nLoad test WITH batching + parallelism config (port 8002):")
total_after, throughput_after = load_test(PORT, test_files)
print(f"  {N} requests in {total_after:.2f}s -> {throughput_after:.1f} req/s")

print(f"\nSpeedup from batching: {throughput_after / throughput_before:.2f}x")
