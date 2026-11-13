from __future__ import annotations
import os
import csv
import time
import shutil

# --- СНАЧАЛА окружение, ПОТОМ импорты HuggingFace ---
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TIMM_OFFLINE"] = "1"

from pathlib import Path
import cv2
import numpy as np
import torch
import psutil
from PIL import Image
from ensemble_boxes import weighted_boxes_fusion

# ==========================================
# НАСТРОЙКА РЕСУРСОВ
# ==========================================
# Менять только эту переменную под мощность ПК (рекомендуется 4, 8 или 12+)
# Это количество физических ядер, не логических
cores = 8

# 1. Ограничиваем PyTorch (внутренняя математика слоев)
# Даем ему ровно столько потоков, сколько ядер выделили.
torch.set_num_threads(cores)

# 2. Ограничиваем PyTorch (inter-op)
# Потоки для независимых операций. Если дать столько же, сколько cores,
# PyTorch начнет "тормозить" на переключении контекста.
# Для инференса оптимально 1 или 2, поэтому берем cores // 4 (но не меньше 1).
try:
    torch.set_num_interop_threads(max(1, cores // 4))
except Exception:
    pass

# 3. Ограничиваем OpenCV
# Для мелких операций (кропы, CLAHE) 1 поток работает быстрее,
# так как не конфликтует с потоками PyTorch.
cv2.setNumThreads(1)

# 4. Ограничиваем процесс на уровне ОС (CPU Affinity)
try:
    proc = psutil.Process()
    available = proc.cpu_affinity()
    if len(available) >= cores:
        proc.cpu_affinity(available[:cores])
        print(f"[limit] CPU affinity: {available[:cores]}")
    else:
        proc.cpu_affinity(available)
        print(f"[limit] CPU affinity: {available} (доступно меньше {cores} ядер, используем все)")
except Exception as e:
    print(f"[warn] cpu_affinity: {e}")

# 5. Понижаем приоритет процесса (чтобы ПК не зависал во время работы)
try:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print("[limit] priority: BELOW_NORMAL")
except Exception as e:
    print(f"[warn] nice: {e}")

# 4. Приоритет процесса
try:
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print("[limit] priority: BELOW_NORMAL")
except Exception as e:
    print(f"[warn] nice: {e}")

from signature_yolo import load_signature_model as load_yolo_sig_model, detect_signatures as detect_yolo_sigs
from signature_detr import load_signature_model as load_detr_sig_model, detect_signatures as detect_detr_sigs
from classifier_sigslip import load_signature_classifier
from stamp import load_stamp_model, detect_stamps

# ==========================================
# НАСТРОЙКИ
# ==========================================
DEBUG = True
CSV_DEBUG = False

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input_all"
OUTPUT_DIR = SCRIPT_DIR / "output"
DEBUG_DIR = SCRIPT_DIR / "debug"
CSV_PATH = DEBUG_DIR / "debug_log.csv"

DEFAULT_BOTTOM_FRAC = 0.7
YOLO_CONF = 0.02
DETR_CONF = 0.05
STAMP_CONF = 0.01

MIN_SIG_WIDTH_FRAC = 0.02
MIN_SIG_HEIGHT_FRAC = 0.01
MONSTER_OVERLAP_FRAC = 0.90
MAX_MERGE_INPUT = 30

STAGE_ORDER = [
    "YOLO", "DETR", "YOLO_TTA", "WBF", "MERGE", "MERGE_DROPPED",
    "CROP", "SIZE_OK", "STAMP_RAW", "STAMP", "SIG_ACCEPTED", "SIG_REJECTED",
]

# --- Кэш моделей для повторных вызовов ---
_MODELS = None


def _get_models(debug_mode: bool = False):
    global _MODELS
    if _MODELS is None:
        print("[+] Загружаем модели в память (единоразово)...")
        _MODELS = {
            "yolo_sig": load_yolo_sig_model(),
            "detr_sig": load_detr_sig_model(),
            "stamp": load_stamp_model(),
            "classifier": load_signature_classifier(debug=debug_mode)
        }
    return _MODELS


def _log_csv(writer, image_name, stage, idx, bbox, confidence, status):
    if writer is None:
        return
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    writer.writerow([
        image_name, stage, idx,
        int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)),
        int(round(w)), int(round(h)), f"{confidence:.4f}", status
    ])


def _log_summary(writer, image_name, stage_counts, reject_reasons):
    if writer is None:
        return
    ordered = [(k, stage_counts[k]) for k in STAGE_ORDER if k in stage_counts]
    for k, v in stage_counts.items():
        if k not in STAGE_ORDER:
            ordered.append((k, v))
    parts = [f"{k}={v}" for k, v in ordered]
    rej = [f"{k}:{v}" for k, v in sorted(reject_reasons.items()) if v > 0]
    rej_str = ",".join(rej) if rej else "none"
    writer.writerow([image_name, "SUMMARY", 0, 0, 0, 0, 0, 0, 0, 0.0, " | ".join(parts) + " || rejects: " + rej_str])


def _nms(boxes, scores, iou_threshold=0.3):
    if not boxes: return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def _is_same_stroke(gray, b1, b2):
    ux1 = int(max(0, min(b1[0], b2[0])))
    uy1 = int(max(0, min(b1[1], b2[1])))
    ux2 = int(min(gray.shape[1], max(b1[2], b2[2])))
    uy2 = int(min(gray.shape[0], max(b1[3], b2[3])))
    if ux2 <= ux1 or uy2 <= uy1: return False
    roi = gray[uy1:uy2, ux1:ux2]
    if roi.size == 0: return False
    scale = 1.0
    if roi.shape[0] > 400 or roi.shape[1] > 400:
        scale = 0.5
        roi_small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        roi_small = roi
    _, thresh = cv2.threshold(roi_small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, roi_small.shape[1] // 8), 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    thresh = cv2.subtract(thresh, horizontal)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False
    best_cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [best_cnt], -1, 255, -1)
    line_pixels = cv2.bitwise_and(thresh, mask)
    dist = cv2.distanceTransform(line_pixels, cv2.DIST_L2, 5)
    valid_dist = dist[dist > 0]
    if len(valid_dist) == 0: return False
    med_thickness = np.median(valid_dist) * 2.0 / scale
    if med_thickness < 1.5 or med_thickness > 8.0: return False
    cx, cy, cw, ch = cv2.boundingRect(best_cnt)
    if scale != 1.0:
        inv = 1.0 / scale
        cx, cy, cw, ch = int(cx * inv), int(cy * inv), int(cw * inv), int(ch * inv)
    cx1, cy1, cx2, cy2 = cx + ux1, cy + uy1, cx + cw + ux1, cy + ch + uy1

    def covers(c_b, b):
        ix1, iy1 = max(c_b[0], b[0]), max(c_b[1], b[1])
        ix2, iy2 = min(c_b[2], b[2]), min(c_b[3], b[3])
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = (b[2] - b[0]) * (b[3] - b[1])
            return inter / (area + 1e-5) > 0.5
        return False

    return covers((cx1, cy1, cx2, cy2), b1) and covers((cx1, cy1, cx2, cy2), b2)


def _merge_by_ink(image_pil, detections):
    if len(detections) <= 1: return detections, 0
    dropped = 0
    if len(detections) > MAX_MERGE_INPUT:
        dropped = len(detections) - MAX_MERGE_INPUT
        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)[:MAX_MERGE_INPUT]
    gray = np.array(image_pil.convert("L"))
    n = len(detections)
    bbox_with_margin = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        w, h = x2 - x1, y2 - y1
        mx, my = max(20, w * 0.5), max(20, h * 0.5)
        bbox_with_margin.append((x1 - mx, y1 - my, x2 + mx, y2 + my))

    def _bbox_overlap_fast(i, j):
        ax1, ay1, ax2, ay2 = bbox_with_margin[i]
        bx1, by1, bx2, by2 = bbox_with_margin[j]
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    parent = list(range(n))

    def find(i):
        if parent[i] != i: parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if not _bbox_overlap_fast(i, j): continue
            x1_1, y1_1, x2_1, y2_1 = detections[i]["bbox"]
            x1_2, y1_2, x2_2, y2_2 = detections[j]["bbox"]
            w1, w2, h1, h2 = x2_1 - x1_1, x2_2 - x1_2, y2_1 - y1_1, y2_2 - y1_2
            min_h, max_h = min(h1, h2), max(h1, h2)
            if max_h / (min_h + 1e-5) > 3.0: continue
            gap_x = max(0.0, x1_2 - x2_1, x1_1 - x2_2)
            gap_y = max(0.0, y1_2 - y2_1, y1_1 - y2_2)
            if gap_x > 0.5 * max(w1, w2) or gap_y > 0.5 * max(h1, h2): continue
            if _is_same_stroke(gray, detections[i]["bbox"], detections[j]["bbox"]):
                union(i, j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(detections[i])
    merged_dets = []
    for group in groups.values():
        merged_dets.append({
            "confidence": max(d["confidence"] for d in group),
            "bbox": [
                min(d["bbox"][0] for d in group), min(d["bbox"][1] for d in group),
                max(d["bbox"][2] for d in group), max(d["bbox"][3] for d in group),
            ],
            "merged_from": len(group),
        })
    return merged_dets, dropped


def _crop_to_ink(image_pil, detections, stem="", debug=False):
    if not detections: return []
    gray = np.array(image_pil.convert("L"))
    H, W = gray.shape
    cropped_dets = []
    dbg_dir = None
    if debug and stem:
        dbg_dir = DEBUG_DIR / stem
        dbg_dir.mkdir(parents=True, exist_ok=True)
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[1] < 10 or roi.shape[0] < 10:
            det["crop_idx"] = idx
            det["crop_contours"] = 0
            det["crop_final_size"] = (x2 - x1, y2 - y1)
            det["crop_fallback"] = True
            cropped_dets.append(det)
            continue
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, roi.shape[1] // 8), 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        thresh_clean = cv2.subtract(thresh, horizontal_lines)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        thresh_dil = cv2.dilate(thresh_clean, kernel_dilate, iterations=1)
        contours, _ = cv2.findContours(thresh_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_rects = []
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < 80 or (cw > roi.shape[1] * 0.9 and ch < 12): continue
            mask = np.zeros_like(thresh_clean)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            ink_pixels = cv2.countNonZero(cv2.bitwise_and(thresh_clean, mask))
            density = ink_pixels / (cw * ch + 1e-5)
            if density > 0.75: continue
            valid_rects.append((cx, cy, cw, ch))
        det["crop_fallback"] = False
        if valid_rects:
            min_cx = min(r[0] for r in valid_rects)
            min_cy = min(r[1] for r in valid_rects)
            max_cx = max(r[0] + r[2] for r in valid_rects)
            max_cy = max(r[1] + r[3] for r in valid_rects)
            pad = 8
            nx1, ny1 = max(x1, x1 + min_cx - pad), max(y1, y1 + min_cy - pad)
            nx2, ny2 = min(x2, x1 + max_cx + pad), min(y2, y1 + max_cy + pad)
            if (nx2 - nx1) > 15 and (ny2 - ny1) > 15:
                det["bbox"] = [float(nx1), float(ny1), float(nx2), float(ny2)]
                det["crop_final_size"] = (nx2 - nx1, ny2 - ny1)
            else:
                det["crop_final_size"] = (int(det["bbox"][2] - det["bbox"][0]), int(det["bbox"][3] - det["bbox"][1]))
            det["crop_contours"] = len(valid_rects)
        else:
            det["crop_contours"] = 0
            det["crop_final_size"] = (x2 - x1, y2 - y1)
            det["crop_fallback"] = True
        det["crop_idx"] = idx
        cropped_dets.append(det)
        if dbg_dir is not None:
            crop_file = dbg_dir / f"crop_{idx}.jpg"
            thresh_file = dbg_dir / f"thresh_{idx}.jpg"
            dbg_crop = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            if valid_rects: cv2.rectangle(dbg_crop, (min_cx, min_cy), (max_cx, max_cy), (0, 255, 0), 2)
            cv2.imwrite(str(crop_file), dbg_crop)
            cv2.imwrite(str(thresh_file), thresh_dil)
    return cropped_dets


def _wbf_fuse(yolo_dets, detr_dets, image_pil, bottom_frac=0.7, iou_thr=0.25):
    W, H = image_pil.size
    y_boundary = int(round(H * (1.0 - bottom_frac)))
    boxes_list, scores_list, labels_list = [], [], []
    for dets in (yolo_dets, detr_dets):
        boxes_norm, scores, labels = [], [], []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            if y_boundary > 0 and (y1 + y2) / 2.0 < y_boundary: continue
            nx1, ny1 = max(0.0, x1 / W), max(0.0, y1 / H)
            nx2, ny2 = min(1.0, x2 / W), min(1.0, y2 / H)
            if (nx2 - nx1) * W < 2 or (ny2 - ny1) * H < 2: continue
            boxes_norm.append([nx1, ny1, nx2, ny2])
            scores.append(float(d["confidence"]))
            labels.append(0)
        boxes_list.append(np.array(boxes_norm, dtype=np.float32).reshape(-1, 4))
        scores_list.append(np.array(scores, dtype=np.float32))
        labels_list.append(np.array(labels, dtype=np.int32))
    if all(len(b) == 0 for b in boxes_list): return []
    try:
        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=[2.5, 1.0], iou_thr=iou_thr, skip_box_thr=0.0,
            conf_type="avg",
        )
    except Exception as e:
        return []
    result = [{"confidence": float(s), "bbox": [float(b[0] * W), float(b[1] * H), float(b[2] * W), float(b[3] * H)]} for
              b, s in zip(fused_boxes, fused_scores)]
    if len(result) > 1:
        keep = _nms([r["bbox"] for r in result], [r["confidence"] for r in result], 0.4)
        result = [result[i] for i in keep]
    if len(result) > 1:
        result = sorted(result, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
        clean = []
        for det in result:
            dx1, dy1, dx2, dy2 = det["bbox"]
            det_area = (dx2 - dx1) * (dy2 - dy1)
            monster = False
            for kept in clean:
                kx1, ky1, kx2, ky2 = kept["bbox"]
                kept_area = (kx2 - kx1) * (ky2 - ky1)
                ix1, iy1 = max(dx1, kx1), max(dy1, ky1)
                ix2, iy2 = min(dx2, kx2), min(dy2, ky2)
                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / kept_area > MONSTER_OVERLAP_FRAC:
                        monster = True
                        break
                    if kept_area > 5000 and det_area > kept_area * 1.5:
                        if (inter / kept_area) > 0.4 and (inter / det_area) > 0.2:
                            monster = True
                            break
            if not monster: clean.append(det)
        result = clean
    return result


def draw_detections(image_pil, sig_dets, stamp_dets, bottom_frac, output_path, all_candidates=None,
                    raw_candidates=None):
    img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]
    y_offset = int(round(H * (1.0 - bottom_frac)))
    if y_offset > 0: cv2.line(img_bgr, (0, y_offset), (W, y_offset), (255, 0, 0), 2)
    approved_bboxes = set((int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in [d["bbox"] for d in sig_dets])
    if raw_candidates:
        for cand in raw_candidates:
            cx1, cy1, cx2, cy2 = map(int, cand["bbox"])
            cv2.rectangle(img_bgr, (cx1, cy1), (cx2, cy2), (150, 150, 150), 1)
    if all_candidates:
        for cand in all_candidates:
            cx1, cy1, cx2, cy2 = map(int, cand["bbox"])
            if (cx1, cy1, cx2, cy2) in approved_bboxes: continue
            cv2.rectangle(img_bgr, (cx1, cy1), (cx2, cy2), (0, 0, 255), 1)
            cv2.putText(img_bgr, f"Rej: {cand['confidence']:.2f}", (cx1 + 5, cy2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
    for det in stamp_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(img_bgr, f"Stamp: {det['confidence']:.2f}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 0, 255), 2)
    for det in sig_dets:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_bgr, f"Sig: {det['confidence']:.2f}", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
    cv2.imwrite(str(output_path), img_bgr)


# =====================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ
# =====================================================================
def process_image(image_pil, stem, sig_yolo_model, sig_detr_model, stamp_model, classifier,
                  output_dir, csv_writer=None, debug: bool = DEBUG, save_output: bool = True):
    print(f"\n=== Обработка: {stem} ===")
    W, H = image_pil.size
    stage_counts, reject_reasons = {}, {}

    # 1. Детекция
    yolo_dets = detect_yolo_sigs(sig_yolo_model, image_pil, bottom_frac=DEFAULT_BOTTOM_FRAC, conf=YOLO_CONF)
    detr_dets = detect_detr_sigs(sig_detr_model, image_pil, bottom_frac=DEFAULT_BOTTOM_FRAC, conf=DETR_CONF)
    for i, d in enumerate(yolo_dets): _log_csv(csv_writer, stem, "YOLO", i, d["bbox"], d["confidence"], "candidate")
    for i, d in enumerate(detr_dets): _log_csv(csv_writer, stem, "DETR", i, d["bbox"], d["confidence"], "candidate")
    stage_counts["YOLO"], stage_counts["DETR"] = len(yolo_dets), len(detr_dets)
    print(f"  кандидатов: YOLO={len(yolo_dets)}, DETR={len(detr_dets)}")

    # 2. WBF
    fused_wbf = _wbf_fuse(yolo_dets, detr_dets, image_pil, bottom_frac=DEFAULT_BOTTOM_FRAC, iou_thr=0.25)
    if not fused_wbf:
        yolo_dets_tta = detect_yolo_sigs(sig_yolo_model, image_pil, bottom_frac=DEFAULT_BOTTOM_FRAC, conf=YOLO_CONF,
                                         use_tta=True)
        for i, d in enumerate(yolo_dets_tta): _log_csv(csv_writer, stem, "YOLO_TTA", i, d["bbox"], d["confidence"],
                                                       "candidate")
        stage_counts["YOLO_TTA"] = len(yolo_dets_tta)
        fused_wbf = _wbf_fuse(yolo_dets_tta, detr_dets, image_pil, bottom_frac=DEFAULT_BOTTOM_FRAC, iou_thr=0.25)
    for i, d in enumerate(fused_wbf): _log_csv(csv_writer, stem, "WBF", i, d["bbox"], d["confidence"], "candidate")
    stage_counts["WBF"] = len(fused_wbf)
    print(f"  после WBF: {len(fused_wbf)}")

    # 3. Склейка
    fused_merged, merge_dropped = _merge_by_ink(image_pil, fused_wbf)
    for i, d in enumerate(fused_merged):
        merged_from = d.get("merged_from", 1)
        status = f"candidate(merged_from={merged_from})" if merged_from > 1 else "candidate"
        _log_csv(csv_writer, stem, "MERGE", i, d["bbox"], d["confidence"], status)
    stage_counts["MERGE"], stage_counts["MERGE_DROPPED"] = len(fused_merged), merge_dropped
    print(f"  после склейки: {len(fused_merged)} (отброшено: {merge_dropped})")

    # 4. Обрезка по чернилам
    fused_cropped = _crop_to_ink(image_pil, fused_merged, stem=stem, debug=debug)
    for i, d in enumerate(fused_cropped):
        contours, fs, fallback = d.get("crop_contours", 0), d.get("crop_final_size", (0, 0)), d.get("crop_fallback",
                                                                                                    False)
        status = f"candidate(contours={contours},final={int(fs[0])}x{int(fs[1])},fallback={fallback})"
        _log_csv(csv_writer, stem, "CROP", i, d["bbox"], d["confidence"], status)
    stage_counts["CROP"] = len(fused_cropped)
    print(f"  после обрезки: {len(fused_cropped)}")

    # 5. Фильтр по размеру
    fused = []
    for i, d in enumerate(fused_cropped):
        w, h = d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]
        if w < W * MIN_SIG_WIDTH_FRAC or h < H * MIN_SIG_HEIGHT_FRAC:
            _log_csv(csv_writer, stem, "SIZE_FILTER", i, d["bbox"], d["confidence"], f"REJECTED_size_{int(w)}x{int(h)}")
        else:
            fused.append(d)
            _log_csv(csv_writer, stem, "SIZE_FILTER", i, d["bbox"], d["confidence"], f"passed_{int(w)}x{int(h)}")
    stage_counts["SIZE_OK"] = len(fused)

    # 6. Штампы: сырые детекции
    stamp_dets_raw = detect_stamps(
        stamp_model, image_pil,
        bottom_frac=DEFAULT_BOTTOM_FRAC,
        conf_thres=STAMP_CONF, debug=debug,
    )
    for i, d in enumerate(stamp_dets_raw):
        _log_csv(csv_writer, stem, "STAMP_RAW", i, d["bbox"], d["confidence"], "candidate")
    stage_counts["STAMP_RAW"] = len(stamp_dets_raw)

    # 7. ЕДИНЫЙ ИНФЕРЕНС (подписи + печати одним батчем)
    bboxes = [det["bbox"] for det in fused]
    crop_indices = [det.get("crop_idx", i) for i, det in enumerate(fused)]

    sig_results, stamp_verdicts = classifier.predict_all(
        image_pil, bboxes, stamp_dets_raw,
        stem=stem if debug else "", crop_indices=crop_indices
    )

    # 7a. Обрабатываем результаты верификации штампов
    stamp_dets: list[dict] = []
    if stamp_dets_raw and stamp_verdicts:
        for i, (d, v) in enumerate(zip(stamp_dets_raw, stamp_verdicts)):
            if v.get("is_stamp"):
                stamp_dets.append(d)
                _log_csv(csv_writer, stem, "STAMP_VERIFIED", i, d["bbox"], v["p_stamp"], f"KEEP_{v['reason']}")
            else:
                _log_csv(csv_writer, stem, "STAMP_VERIFIED", i, d["bbox"], v["p_stamp"], f"DROP_{v['reason']}")
                rej_key = f"stamp_{v['reason']}"
                reject_reasons[rej_key] = reject_reasons.get(rej_key, 0) + 1

    stage_counts["STAMP"] = len(stamp_dets)
    print(f"  штампов (после верификации): {len(stamp_dets)}")

    # 7b. Обрабатываем результаты классификации подписей
    signatures = []
    if fused and sig_results:
        for i, (det, res) in enumerate(zip(fused, sig_results)):
            label, conf, reason = res[0], res[1], (res[2] if len(res) > 2 else None)
            real_status = (f"ACCEPTED_{reason}" if label == "signature" else f"REJECTED_{reason}") if reason else (
                "ACCEPTED" if label == "signature" else "REJECTED_clip")
            _log_csv(csv_writer, stem, "CLIP", i, det["bbox"], conf, real_status)
            if label == "signature":
                signatures.append({"bbox": det["bbox"], "confidence": conf, "detector_confidence": det["confidence"]})
            else:
                rej_key = reason if reason else "clip_reject"
                reject_reasons[rej_key] = reject_reasons.get(rej_key, 0) + 1

    stage_counts["SIG_ACCEPTED"] = len(signatures)
    stage_counts["SIG_REJECTED"] = sum(
        v for k, v in reject_reasons.items() if not k.startswith("stamp_")
    )

    # 8. SUMMARY
    _log_summary(csv_writer, stem, stage_counts, reject_reasons)
    print(f"=== Итог: Подписей: {len(signatures)}, Штампов: {len(stamp_dets)} ===")

    # 9. Сохранение выходного файла
    if save_output:
        draw_detections(
            image_pil, signatures, stamp_dets, DEFAULT_BOTTOM_FRAC,
            output_dir / f"{stem}_processed.jpg",
            all_candidates=fused,
            raw_candidates=fused_wbf,
        )

    return {"signatures": signatures, "stamps": stamp_dets}


# =====================================================================
# ТОЧЕЧНЫЙ ВЫЗОВ
# =====================================================================
def detect_pil_image(image_pil: Image.Image, debug: bool = False, save_output: bool = False) -> dict:
    models = _get_models(debug_mode=debug)
    temp_out = SCRIPT_DIR / "temp_output"
    temp_out.mkdir(exist_ok=True)

    result = process_image(
        image_pil=image_pil,
        stem="memory_image",
        sig_yolo_model=models["yolo_sig"],
        sig_detr_model=models["detr_sig"],
        stamp_model=models["stamp"],
        classifier=models["classifier"],
        output_dir=temp_out,
        csv_writer=None,
        debug=debug,
        save_output=save_output
    )
    return result


# =====================================================================
# CLI
# =====================================================================
def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    if DEBUG_DIR.exists():
        shutil.rmtree(DEBUG_DIR)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in INPUT_DIR.iterdir() if
                    p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
    if not images:
        print(f"[!] В папке '{INPUT_DIR}' нет изображений.")
        return

    models = _get_models(debug_mode=DEBUG)

    csv_file = None
    csv_writer = None
    if CSV_DEBUG:
        csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["image_name", "stage", "idx", "x1", "y1", "x2", "y2", "width", "height", "confidence", "status"])

    total_start = time.perf_counter()
    per_file_times = []

    try:
        for idx, img_path in enumerate(images, start=1):
            file_start = time.perf_counter()
            try:
                with Image.open(img_path) as img:
                    img.load()
                    image_pil = img.convert("RGB")

                process_image(
                    image_pil, img_path.stem, models["yolo_sig"], models["detr_sig"],
                    models["stamp"], models["classifier"], OUTPUT_DIR, csv_writer,
                    debug=DEBUG, save_output=True
                )
            except Exception as e:
                import traceback
                print(f"[!] Ошибка при обработке {img_path.name}: {e}")
                traceback.print_exc()

            file_elapsed = time.perf_counter() - file_start
            per_file_times.append((img_path.name, file_elapsed))
            print(f"[TIME] [{idx}/{len(images)}] {img_path.name}: {file_elapsed:.2f} s")
    finally:
        total_elapsed = time.perf_counter() - total_start
        if csv_file is not None:
            csv_file.close()

        if per_file_times:
            times = [t for _, t in per_file_times]
            total, avg = sum(times), sum(times) / len(times)
            slowest, fastest = max(per_file_times, key=lambda x: x[1]), min(per_file_times, key=lambda x: x[1])
            print("\n" + "=" * 60)
            print("[TIME] ИТОГОВАЯ СТАТИСТИКА")
            print("=" * 60)
            print(f"  Файлов обработано: {len(per_file_times)}")
            print(f"  Сумма по файлам: {total:.2f} s")
            print(f"  Общее время (wall clock): {total_elapsed:.2f} s")
            print(f"  Среднее на файл: {avg:.2f} s")
            print(f"  Самый медленный: {slowest[0]} — {slowest[1]:.2f} s")
            print(f"  Самый быстрый:  {fastest[0]} — {fastest[1]:.2f} s")
            print("=" * 60)


if __name__ == "__main__":
    main()