import os
import json
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from torchvision import transforms

with open("results_project2.json") as f:
    results = json.load(f)

augment = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])
calib_tensors = []
for img in images:
    for _ in range(10):
        calib_tensors.append(preprocess(augment(img)).unsqueeze(0).numpy())

class CalibDataReader(CalibrationDataReader):
    def __init__(self, tensors):
        self.tensors = tensors
        self.enum_data = None
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.tensors)
        t = next(self.enum_data, None)
        return None if t is None else {"pixel_values": t}

preprocessed_path = "mobilenetv3-preprocessed-static.onnx"
try:
    quant_pre_process(onnx_path, preprocessed_path, auto_merge=True)
except Exception as e:
    print(f"Symbolic shape inference failed ({e}); falling back.")
    quant_pre_process(onnx_path, preprocessed_path, skip_symbolic_shape=True)

pm = onnx.load(preprocessed_path)
del pm.graph.value_info[:]
cleaned_path = "mobilenetv3-preprocessed-static-cleaned.onnx"
onnx.save(pm, cleaned_path)

static_path = "mobilenetv3_static_int8.onnx"
quantize_static(
    model_input=cleaned_path,
    model_output=static_path,
    calibration_data_reader=CalibDataReader(calib_tensors),
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
print(f"Saved static-quantized model to {static_path}")

s_session = ort.InferenceSession(static_path, providers=["CPUExecutionProvider"])
for img, url in zip(images, image_urls):
    xi = preprocess(img).unsqueeze(0).numpy()
    pred = int(np.argmax(s_session.run(["logits"], {"pixel_values": xi})[0], axis=-1)[0])
    print(f"[Static INT8] {url.split('/')[-1]}: predicted '{class_labels[pred]}'")

for _ in range(10):
    _ = s_session.run(["logits"], {"pixel_values": x.numpy()})
latencies = []
for _ in range(100):
    start = time.perf_counter()
    _ = s_session.run(["logits"], {"pixel_values": x.numpy()})
    latencies.append((time.perf_counter() - start) * 1000)
latencies = np.array(latencies)
size_mb = os.path.getsize(static_path) / (1024 * 1024)
print(f"Static INT8 latency — mean: {latencies.mean():.2f}ms, p95: {np.percentile(latencies, 95):.2f}ms")
print(f"Static INT8 size: {size_mb:.2f} MB")

results["ONNX Static INT8"] = {
    "latency_mean_ms": float(latencies.mean()),
    "latency_p95_ms": float(np.percentile(latencies, 95)),
    "size_mb": float(size_mb),
}
with open("results_project2.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
