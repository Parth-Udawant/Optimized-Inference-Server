# !pip install -q onnx onnxruntime torchvision pillow requests

import os
import json
import time
from io import BytesIO

import numpy as np
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import onnx
import onnxruntime as ort
from PIL import Image
import requests as pyrequests

device = torch.device("cpu")

weights = MobileNet_V3_Large_Weights.DEFAULT
model = mobilenet_v3_large(weights=weights)
model.eval()
preprocess = weights.transforms()
class_labels = weights.meta["categories"]

image_urls = [
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",  
]
images = []
for url in image_urls:
    resp = pyrequests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    images.append(img)
print(f"Loaded {len(images)} test images")

def predict(img, model):
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(dim=-1).item()

for img, url in zip(images, image_urls):
    pred = predict(img, model)
    print(f"{url.split('/')[-1]}: predicted '{class_labels[pred]}'")

x = preprocess(images[0]).unsqueeze(0)

for _ in range(10): 
    with torch.no_grad():
        _ = model(x)

n_runs = 100
latencies = []
for _ in range(n_runs):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    latencies.append((time.perf_counter() - start) * 1000)
latencies = np.array(latencies)
print(f"PyTorch FP32 latency — mean: {latencies.mean():.2f}ms, p95: {np.percentile(latencies, 95):.2f}ms")

torch.save(model.state_dict(), "mobilenetv3_fp32.pt")
size_mb = os.path.getsize("mobilenetv3_fp32.pt") / (1024 * 1024)
print(f"PyTorch FP32 size: {size_mb:.2f} MB")

results = {
    "PyTorch FP32": {
        "latency_mean_ms": float(latencies.mean()),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "size_mb": float(size_mb),
    }
}

onnx_path = "mobilenetv3.onnx"
torch.onnx.export(
    model,
    x,
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)

m = onnx.load(onnx_path, load_external_data=True)
onnx.save(m, onnx_path, save_as_external_data=False)
onnx.checker.check_model(onnx.load(onnx_path))

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_logits = session.run(["logits"], {"pixel_values": x.numpy()})[0]
with torch.no_grad():
    torch_logits = model(x).numpy()
max_diff = float(np.abs(torch_logits - onnx_logits).max())
print(f"Max logit diff (PyTorch vs ONNX): {max_diff:.6f}")
assert max_diff < 1e-3, "ONNX export diverges from PyTorch"
print("PASSED: ONNX matches PyTorch.")

onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

for _ in range(10):
    _ = session.run(["logits"], {"pixel_values": x.numpy()})
latencies = []
for _ in range(n_runs):
    start = time.perf_counter()
    _ = session.run(["logits"], {"pixel_values": x.numpy()})
    latencies.append((time.perf_counter() - start) * 1000)
latencies = np.array(latencies)
print(f"ONNX FP32 latency — mean: {latencies.mean():.2f}ms, p95: {np.percentile(latencies, 95):.2f}ms")
print(f"ONNX FP32 size: {onnx_size_mb:.2f} MB")

results["ONNX FP32"] = {
    "latency_mean_ms": float(latencies.mean()),
    "latency_p95_ms": float(np.percentile(latencies, 95)),
    "size_mb": float(onnx_size_mb),
}

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

preprocessed_path = "mobilenetv3-preprocessed.onnx"
try:
    quant_pre_process(onnx_path, preprocessed_path, auto_merge=True)
except Exception as e:
    print(f"Symbolic shape inference failed ({e}); falling back to basic shape inference.")
    quant_pre_process(onnx_path, preprocessed_path, skip_symbolic_shape=True)

pm = onnx.load(preprocessed_path)
del pm.graph.value_info[:]
cleaned_path = "mobilenetv3-preprocessed-cleaned.onnx"
onnx.save(pm, cleaned_path)

quantized_path = "mobilenetv3_int8.onnx"
quantize_dynamic(model_input=cleaned_path, model_output=quantized_path, weight_type=QuantType.QInt8)
print(f"Saved quantized model to {quantized_path}")

q_session = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])

for img, url in zip(images, image_urls):
    xi = preprocess(img).unsqueeze(0).numpy()
    pred = int(np.argmax(q_session.run(["logits"], {"pixel_values": xi})[0], axis=-1)[0])
    print(f"[INT8] {url.split('/')[-1]}: predicted '{class_labels[pred]}'")

for _ in range(10):
    _ = q_session.run(["logits"], {"pixel_values": x.numpy()})
latencies = []
for _ in range(n_runs):
    start = time.perf_counter()
    _ = q_session.run(["logits"], {"pixel_values": x.numpy()})
    latencies.append((time.perf_counter() - start) * 1000)
latencies = np.array(latencies)
q_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
print(f"INT8 latency — mean: {latencies.mean():.2f}ms, p95: {np.percentile(latencies, 95):.2f}ms")
print(f"INT8 size: {q_size_mb:.2f} MB")

results["ONNX Dynamic INT8"] = {
    "latency_mean_ms": float(latencies.mean()),
    "latency_p95_ms": float(np.percentile(latencies, 95)),
    "size_mb": float(q_size_mb),
}

with open("results_project2.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved results_project2.json:")
print(json.dumps(results, indent=2))
