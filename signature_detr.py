from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
HF_MODEL_ID = "tech4humans/conditional-detr-50-signature-detector"
LOCAL_MODEL_DIR = MODELS_DIR / "detr_signature"
DEFAULT_BOTTOM_FRAC = 0.7
CONF_THRESHOLD = 0.1


class DETRSignatureModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, image_pil: Image.Image, conf: float = CONF_THRESHOLD, **kwargs) -> list[dict]:
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")

        # --- ОПТИМИЗАЦИЯ: Уменьшаем картинку для DETR ---
        # DETR квадратично зависит от разрешения. Ограничиваем длинную сторону до 1600px.
        W, H = image_pil.size
        max_side = max(W, H)
        scale = 1.0
        if max_side > 1600:
            scale = 1600.0 / max_side
            new_w = int(round(W * scale))
            new_h = int(round(H * scale))
            image_pil_resized = image_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            image_pil_resized = image_pil

        inputs = self.processor(images=image_pil_resized, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Возвращаем координаты к оригинальному размеру
        target_sizes = torch.tensor([image_pil.size[::-1]]).to(self.device)

        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]

        dets = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                continue
            dets.append({
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return dets


def load_signature_model() -> DETRSignatureModel:
    MODELS_DIR.mkdir(exist_ok=True)
    local_ready = (LOCAL_MODEL_DIR / "config.json").exists()

    if not local_ready:
        raise FileNotFoundError(
            f"Локальная папка DETR не найдена или пуста: {LOCAL_MODEL_DIR}\n"
            f"Скачайте модель и положите в эту папку."
        )

    print(f"[signature_detr] Загружаем модель из локальной папки: {LOCAL_MODEL_DIR}")
    processor = AutoImageProcessor.from_pretrained(str(LOCAL_MODEL_DIR), local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(str(LOCAL_MODEL_DIR), local_files_only=True)

    model.eval()
    return DETRSignatureModel(model, processor)


def detect_signatures(
        model: DETRSignatureModel,
        image_pil: Image.Image,
        output_path: str | None = None,
        bottom_frac: float = DEFAULT_BOTTOM_FRAC,
        conf: float = CONF_THRESHOLD,
) -> list[dict]:
    """
    Инференс по полному изображению. Возвращает ВСЕ детекции DETR,
    чья центральная точка попадает в нижнюю часть листа (bottom_frac).
    """
    if image_pil is None:
        raise ValueError("Пустое изображение")

    W, H = image_pil.size
    bottom_frac = max(0.1, min(1.0, float(bottom_frac)))
    y_boundary = int(round(H * (1.0 - bottom_frac)))

    detections = model.predict(image_pil, conf=conf)

    filtered = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cy = (y1 + y2) / 2.0
        if y_boundary > 0 and cy < y_boundary:
            continue
        filtered.append(d)

    return filtered