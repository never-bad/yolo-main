"""临时测试脚本：验证 Transformers 集成的 GroundingDINO 下载与推理"""
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from PIL import Image
import torch

model_id = "IDEA-Research/grounding-dino-tiny"
t0 = time.time()
print("loading processor/model ...", flush=True)
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
model.eval()
print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

img = Image.open(
    r"c:\Users\41653\Desktop\yolo-main\yolo-main\backend\data\datasets\ds_20260811_095952\v1\images\train\1653754052098_jpg.rf.58d1b4d59e677be335b8effdf0e21265.jpg"
).convert("RGB")
caption = "person . car"
inputs = proc(images=img, text=caption, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
res = proc.post_process_grounded_object_detection(
    out, inputs.input_ids, box_threshold=0.3, text_threshold=0.25,
    target_sizes=[(img.height, img.width)],
)[0]
print("num detections:", len(res["boxes"]), flush=True)
print("boxes:", res["boxes"].tolist(), flush=True)
print("scores:", res["scores"].tolist(), flush=True)
print("labels:", res["labels"], flush=True)
print("DONE", flush=True)