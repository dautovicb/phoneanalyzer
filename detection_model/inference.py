import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path

MODEL_PATH = "detect_phone.onnx"
INPUT_SIZE = 512
CLASS_NAMES = {0: "objects", 1: "box", 2: "phone_back", 3: "phone_front"}
PHONE_CLASSES = {2, 3}


def load_model(model_path: str = str(MODEL_PATH)) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


def postprocess(dets, labels, orig_w, orig_h, threshold=0.5):
    dets = dets[0]     
    labels = labels[0]  

    exp = np.exp(labels - np.max(labels, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    class_ids = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    results = []
    for i in range(len(dets)):
        conf = confidences[i]
        cls_id = class_ids[i]
        if conf < threshold or cls_id not in PHONE_CLASSES:
            continue

        cx, cy, w, h = dets[i]
        x1 = max(0, int((cx - w / 2) * orig_w))
        y1 = max(0, int((cy - h / 2) * orig_h))
        x2 = min(orig_w, int((cx + w / 2) * orig_w))
        y2 = min(orig_h, int((cy + h / 2) * orig_h))

        results.append((CLASS_NAMES[cls_id], float(conf), x1, y1, x2, y2))

    results.sort(key=lambda r: r[1], reverse=True)
    return results


def detect_and_crop(image: Image.Image, session: ort.InferenceSession, threshold=0.5):
    orig_w, orig_h = image.size
    input_tensor = preprocess(image)
    dets, labels = session.run(None, {"input": input_tensor})
    detections = postprocess(dets, labels, orig_w, orig_h, threshold)

    crops = []
    for cls_name, conf, x1, y1, x2, y2 in detections:
        crops.append((cls_name, conf, image.crop((x1, y1, x2, y2))))
    return crops


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect phones and export crops")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    session = load_model(args.model)
    image = Image.open(args.image).convert("RGB")
    crops = detect_and_crop(image, session, args.threshold)

    if not crops:
        print("No phone detected.")
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        crops.sort(key=lambda c: c[1], reverse=True)  
        cls_name, conf, crop = crops[0]
        out_path = output_dir / f"{cls_name}_{conf:.2f}.jpg"
        crop.save(out_path)
