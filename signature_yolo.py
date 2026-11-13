from __future__ import annotations
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODEL_FILENAME = "detector_yolo_1cls.pt"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
DEFAULT_BOTTOM_FRAC = 0.7

IMG_MAX_SIDE = 1600
IMG_ALIGN = 32
IMG_MIN_SIDE = 64


def _compute_imgsz(image_pil: Image.Image) -> tuple[int, int]:
    """Возвращает (H, W) — так, как ожидает ultralytics."""
    w, h = image_pil.size
    scale = IMG_MAX_SIDE / max(w, h)
    new_h = max(IMG_MIN_SIDE, int(round(h * scale / IMG_ALIGN) * IMG_ALIGN))
    new_w = max(IMG_MIN_SIDE, int(round(w * scale / IMG_ALIGN) * IMG_ALIGN))
    return new_h, new_w


def load_signature_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найдена модель: {MODEL_PATH}")
    print(f"[signature_yolo] Загружаем модель: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def detect_signatures(
        model: YOLO,
        image_pil: Image.Image,
        output_path: str | None = None,
        bottom_frac: float = DEFAULT_BOTTOM_FRAC,
        conf: float = CONF_THRESHOLD,
        use_tta: bool = False,
) -> list[dict]:
    if image_pil is None:
        raise ValueError("Пустое изображение")

    W, H = image_pil.size
    bottom_frac = max(0.1, min(1.0, float(bottom_frac)))
    y_boundary = int(round(H * (1.0 - bottom_frac)))
    imgsz = _compute_imgsz(image_pil)

    results_list = model.predict(
        image_pil,
        max_det=300,
        conf=conf,
        imgsz=imgsz,
        iou=IOU_THRESHOLD,
        augment=use_tta,               # TTA-фолбэк для сложных сканов
        agnostic_nms=True,
        verbose=False,
    )
    results = results_list[0] if results_list else None

    detections = []
    if results is not None and getattr(results, "boxes", None) is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            score = float(box.conf[0])

            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue

            cy = (y1 + y2) / 2.0
            if y_boundary > 0 and cy < y_boundary:
                continue

            detections.append({"confidence": score, "bbox": [x1, y1, x2, y2]})

    return detections