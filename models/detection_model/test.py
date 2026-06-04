import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
import onnxruntime as ort
from PIL import Image
from inference import preprocess, postprocess, load_model
from tqdm import tqdm
import torch

MODEL_PATH = "output/detect_phone_v2.onnx"
DATASET_PATH = "dataset/test"
ANNOTATIONS = DATASET_PATH + "/_annotations.coco.json"
INPUT_SIZE = 512

def load_coco(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
       
        x, y, w, h = ann['bbox']
        img_id_to_anns[img_id].append({
            "boxes": [x, y, x + w, y + h],
            "labels": ann['category_id']
        })
    
    img_info = {img['id']: img['file_name'] for img in data['images']}
    return img_id_to_anns, img_info

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def predict(session: ort.InferenceSession, threshold=0.4):
    gt_data, img_info = load_coco(ANNOTATIONS)
    metric = MeanAveragePrecision(iou_type="bbox")
    
    for img_id, filename in tqdm(img_info.items(), desc="Evaluating ONNX"):
        path = os.path.join(DATASET_PATH, filename)
        if not os.path.exists(path): continue
        
        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        input_tensor = preprocess(image)
        dets, labels = session.run(None, {"input": input_tensor})
        detections = postprocess(dets, labels, orig_w, orig_h, threshold)

        preds_boxes = []
        preds_scores = []
        preds_labels = []

        for cls_id, conf, x1, y1, x2, y2 in detections:
            preds_boxes.append([x1, y1, x2, y2])
            preds_scores.append(conf)
            preds_labels.append(cls_id)

        if not preds_boxes: 
            preds_dict = {"boxes": torch.tensor([]), "scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.int64)}
        else:
            preds_dict = {
                "boxes": torch.tensor(preds_boxes),
                "scores": torch.tensor(preds_scores),
                "labels": torch.tensor(preds_labels, dtype=torch.int64)
            }

        gt_list = gt_data.get(img_id, [])
        gt_dict = {
            "boxes": torch.tensor([g['boxes'] for g in gt_list]),
            "labels": torch.tensor([g['labels'] for g in gt_list], dtype=torch.int64)
        }
        
        metric.update([preds_dict], [gt_dict])
    results = metric.compute()
    print(f"\nONNX mAP @ 0.5: {results['map_50']:.4f}")
    print(f"ONNX mAP @ 0.5-0.95: {results['map']:.4f}")

def main():
    session = load_model(MODEL_PATH)
    predict(session, 0.5)

if __name__ == "__main__":
    main()