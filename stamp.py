from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
WEIGHTS_PATH = MODELS_DIR / "stamps_yolov8.torchscript"

# === DEBUG ===================================================================
DEBUG_ENABLED = False
DEBUG_DIR = SCRIPT_DIR / "debug"
_DEBUG_RAW_CONF_THRES = 0.05
_DEBUG_RAW_MAX_ROWS = 500

_PAPER_COLOR = (238, 243, 248)

# Два recolor-варианта. Один тяжёлый препроцесс (CLAHE + blur) — один раз
# на изображение; recolor через LUT — копейки на вариант.
#   neutral — базовый, ловит чёрные/серые/бледные печати (99%);
#   blue    — синяя «сигнатура» в каналах, ловит синие печати,
#             которые нейтральный recolor не видит.
_STAMP_COLOR_VARIANTS = (
    ("neutral", (55, 50, 45)),
    ("blue",    (130, 70, 40)),
)

# Совместимость со старым кодом (если где-то импортируется).
_INK_COLOR_NEUTRAL = _STAMP_COLOR_VARIANTS[0][1]

# Доля листа снизу, в которой ищем штампы. 0.7 = отбрасываем верхние 30%.
DEFAULT_BOTTOM_FRAC = 0.7

# Подавление длинных линий (росчерки подписей, табличные линии) —
# включено по умолчанию.
DEFAULT_REMOVE_LONG_LINES = True

IMG_SIZE = 640
_FALLBACK_IMG_SIZE = 640

# --- A4-нормализация (препроцесс) ------------------------------------------
SCALE_TO_A4 = True
A4_TARGET_SHORT_SIDE_PX = 1250   #доступно 640/1250/2500
_A4_TARGET_REF = 1250.0

_A4_SCALE_TOLERANCE_FRAC = 0.08
_A4_SCALE_TOLERANCE_MIN = 4

# Финальное центрированное уменьшение bbox.
STAMP_BBOX_SCALE = 0.70

# --- CLAHE -------------------------------------------------------------------
_CLAHE_CLIP_LIMIT = 3.0
_CLAHE_TILE_PX_BASE = 110.0
_CLAHE_TILE_MIN_CELLS = 4
_CLAHE_TILE_MAX_CELLS = 48
_CLAHE_TILE_PX_FLOOR = 16.0

# Лёгкое сглаживание после CLAHE.
_SOFT_KSIZE = 3
_SOFT_SIGMA = 1.0

# --- Слияние TTA между цветовыми проходами ---------------------------------
# dedup шире стандартного (20 vs 15): один и тот же штамп с neutral и blue
# может дать боксы, центры которых в исходном пространстве отличаются
# на 10-15 px.
_TTA_NMS_IOU = 0.30
_TTA_DEDUP_DIST = 20.0
_TTA_CONTAINMENT_RATIO = 0.6

_MODEL_INPUT_SIZE = [IMG_SIZE]


# === DEBUG helpers ===========================================================
def _open_debug_csv(debug_dir: Path) -> tuple:
    try:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = debug_dir / f"stamps_{ts}.csv"
        f = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow([
            "stage", "pass", "idx", "x1", "y1", "x2", "y2", "conf",
            "w", "h", "aspect", "area_frac", "status", "note",
        ])
        f.flush()
        print(f"[stamp] debug CSV: {path}")
        return f, w, path
    except Exception as e:
        print(f"[stamp] не удалось открыть debug CSV: {e}")
        return None, None, None


def _log_bbox(writer, stage, pass_name, idx, x1, y1, x2, y2, conf,
              img_w, img_h, status="", note=""):
    if writer is None:
        return
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    w = x2 - x1
    h = y2 - y1
    aspect = (w / h) if h > 0 else 0.0
    area_frac = (w * h) / (img_w * img_h) if (img_w > 0 and img_h > 0) else 0.0
    writer.writerow([
        stage, pass_name, idx,
        f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
        f"{float(conf):.4f}",
        f"{w:.2f}", f"{h:.2f}",
        f"{aspect:.3f}", f"{area_frac:.6f}",
        status, note,
    ])


def _log_boxes(writer, stage, pass_name, boxes, scores,
               img_w, img_h, status="", note=""):
    if writer is None or boxes is None or len(boxes) == 0:
        return
    for i, (b, s) in enumerate(zip(boxes, scores)):
        _log_bbox(writer, stage, pass_name, i,
                  b[0], b[1], b[2], b[3], s,
                  img_w, img_h, status, note)


# === Model ===================================================================
def load_stamp_model() -> torch.jit.ScriptModule:
    weights = WEIGHTS_PATH
    if not weights.exists():
        alt = MODELS_DIR / "stamps_yolov8.pt"
        if alt.exists():
            alt.rename(weights)
        else:
            raise FileNotFoundError(f"Модель не найдена: {weights}")

    print(f"[stamp] Загружаем модель: {weights}")
    device = torch.device("cpu")
    try:
        model = torch.jit.load(str(weights), map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"Не удалось загрузить TorchScript: {weights}\n"
            f"Причина: {e}"
        ) from e
    model.eval()
    return model


def _as_bgr_numpy(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    try:
        from PIL import Image as _PILImage
    except ImportError:
        _PILImage = None

    if _PILImage is not None and isinstance(image, _PILImage.Image):
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    raise TypeError(
        f"detect_stamps: ожидается np.ndarray или PIL.Image, "
        f"получен {type(image).__name__}"
    )


# === Динамические параметры под текущий A4_TARGET ===========================
def _a4_tolerance_for(target_short: int) -> int:
    return max(
        _A4_SCALE_TOLERANCE_MIN,
        int(round(target_short * _A4_SCALE_TOLERANCE_FRAC)),
    )


def _clahe_tile_px_for(shape_hw: tuple[int, int]) -> float:
    H, W = shape_hw
    short = float(min(H, W))
    tile_px = _CLAHE_TILE_PX_BASE * (short / _A4_TARGET_REF)
    return max(_CLAHE_TILE_PX_FLOOR, tile_px)


def _clahe_grid_for(shape_hw: tuple[int, int]) -> tuple[int, int]:
    H, W = shape_hw
    tile_px = _clahe_tile_px_for(shape_hw)
    gx = int(round(W / tile_px))
    gy = int(round(H / tile_px))
    gx = max(_CLAHE_TILE_MIN_CELLS, min(_CLAHE_TILE_MAX_CELLS, gx))
    gy = max(_CLAHE_TILE_MIN_CELLS, min(_CLAHE_TILE_MAX_CELLS, gy))
    return gx, gy


def _scale_to_a4(
    image_bgr: np.ndarray,
    target_short: int = A4_TARGET_SHORT_SIDE_PX,
    tolerance: int | None = None,
) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return image_bgr, 1.0
    if tolerance is None:
        tolerance = _a4_tolerance_for(target_short)

    short_in = min(h, w)
    if abs(short_in - target_short) <= tolerance:
        return image_bgr, 1.0

    scale = target_short / float(short_in)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=interp), scale


def _suppress_long_lines(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    _, thresh_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h_len = max(200, W // 4)
    v_len = max(200, H // 4)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, h_k)
    v_lines = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, v_k)
    lines = cv2.bitwise_or(h_lines, v_lines)

    result = gray.copy()
    result[lines > 0] = 255
    return result


def _prepare_gray_for_stamps(
    image_bgr: np.ndarray,
    remove_long_lines: bool = False,
) -> np.ndarray:
    """
    Тяжёлый препроцесс, инвариантный к цвету чернил:
      gray -> (опц.) suppress long lines -> CLAHE -> Gaussian.
    Возвращает uint8 (H, W). Из него _recolor_from_soft делает любой
    цветовой вариант за копейки (LUT).
    """
    gray = (
        image_bgr
        if image_bgr.ndim == 2
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    )

    if remove_long_lines:
        gray = _suppress_long_lines(gray)

    tile_grid = _clahe_grid_for(gray.shape[:2])
    clahe = cv2.createCLAHE(
        clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=tile_grid
    )
    gray_adaptive = clahe.apply(gray)

    soft = cv2.GaussianBlur(
        gray_adaptive, (_SOFT_KSIZE, _SOFT_KSIZE), sigmaX=_SOFT_SIGMA
    )
    return soft


_RECOLOR_LUT_CACHE: dict[tuple, np.ndarray] = {}


def _get_recolor_lut(
    ink_bgr: tuple[int, int, int],
    paper_bgr: tuple[int, int, int] = _PAPER_COLOR,
) -> np.ndarray:
    """
    LUT [256, 3] uint8: для каждого значения soft (0..255) даёт
    итоговый BGR paper*soft/255 + ink*(1-soft/255).
    Считается один раз на (ink, paper) и кешируется.
    """
    key = (tuple(ink_bgr), tuple(paper_bgr))
    lut = _RECOLOR_LUT_CACHE.get(key)
    if lut is not None:
        return lut

    paper = np.array(paper_bgr, np.float32)
    ink = np.array(ink_bgr, np.float32)
    v = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(
        paper[None, :] * v[:, None] + ink[None, :] * (1.0 - v[:, None]),
        0, 255,
    ).astype(np.uint8)
    _RECOLOR_LUT_CACHE[key] = lut
    return lut


def _recolor_from_soft(
    soft: np.ndarray,
    ink_bgr: tuple[int, int, int],
) -> np.ndarray:
    return _get_recolor_lut(ink_bgr)[soft]


# Совместимость со старым кодом: если где-то вызывается
# _prepare_adaptive_one_pass(...), работаем как раньше (один recolor).
def _prepare_adaptive_one_pass(
    image_bgr: np.ndarray,
    remove_long_lines: bool = False,
    ink_bgr: tuple[int, int, int] | None = None,
    paper_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    if ink_bgr is None:
        ink_bgr = _INK_COLOR_NEUTRAL
    soft = _prepare_gray_for_stamps(image_bgr, remove_long_lines=remove_long_lines)
    return _recolor_from_soft(soft, ink_bgr)


def _letterbox(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape - new_unpad[0]) / 2
    dh = (new_shape - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (left, top)


def _to_tensor(image_bgr: np.ndarray, img_size: int = IMG_SIZE):
    img, ratio, pad = _letterbox(image_bgr, img_size)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0), ratio, pad


def _nms_numpy(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def _dedup_by_center(boxes, scores, dist_thres=15.0, area_ratio_thres=0.7):
    if len(boxes) <= 1:
        return boxes, scores
    centers = np.stack(
        [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2],
        axis=1,
    )
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    used = np.zeros(len(boxes), dtype=bool)
    keep_idx = []
    for i in order:
        if used[i]:
            continue
        keep_idx.append(i)
        used[i] = True
        dist = np.linalg.norm(centers - centers[i], axis=1)
        area_ratio = np.minimum(areas, areas[i]) / np.maximum(
            areas, areas[i] + 1e-9
        )
        used |= (dist < dist_thres) & (area_ratio > area_ratio_thres)
    keep_idx = np.array(keep_idx, dtype=int)
    return boxes[keep_idx], scores[keep_idx]


def _remove_containing_boxes(boxes, scores, area_ratio_thres=0.6):
    if len(boxes) <= 1:
        return boxes, scores
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    for i in range(len(boxes)):
        is_container = False
        for j in range(len(boxes)):
            if i == j:
                continue
            if (boxes[i, 0] <= boxes[j, 0] and
                    boxes[i, 1] <= boxes[j, 1] and
                    boxes[i, 2] >= boxes[j, 2] and
                    boxes[i, 3] >= boxes[j, 3]):
                if areas[j] < areas[i] * area_ratio_thres:
                    is_container = True
                    break
        if not is_container:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep]


def _decode_output(
    output, ratio, pad, conf_thres=0.25, iou_thres=0.3, class_id=None,
    debug_writer=None, debug_pass_name="", debug_img_hw=None,
):
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        tensor = next(
            (o for o in output if isinstance(o, torch.Tensor) and o.numel() > 0),
            None,
        )
        if tensor is None:
            return []
        output = tensor
    if not isinstance(output, torch.Tensor) or output.numel() == 0:
        return []
    if output.dim() == 4:
        output = output.squeeze(1)
    if output.dim() == 3:
        output = output[0]
    if output.dim() < 2:
        return []
    if output.shape[0] < output.shape[1]:
        output = output.T

    preds = output.detach().cpu().numpy()
    n_det, n_feat = preds.shape
    if n_feat < 5:
        return []

    if n_feat > 5:
        cls_scores = preds[:, 4:].astype(np.float32)
        if cls_scores.size and cls_scores.max() > 1.0 + 1e-6:
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))
        cls = cls_scores.argmax(axis=1)
        conf = cls_scores[np.arange(len(cls)), cls]
    else:
        conf = preds[:, 4].astype(np.float32)
        if conf.size and conf.max() > 1.0 + 1e-6:
            conf = 1.0 / (1.0 + np.exp(-conf))
        cls = np.zeros(len(conf), dtype=np.int64)

    conf = np.clip(conf, 0.0, 1.0)

    xyxy_frac = float(np.mean(
        (preds[:, 2] > preds[:, 0]) & (preds[:, 3] > preds[:, 1])
    ))
    is_raw = (n_det > 1000) or (xyxy_frac < 0.5)
    if is_raw:
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        all_boxes = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
        )
    else:
        all_boxes = preds[:, :4].copy()

    all_boxes = all_boxes.copy()
    all_boxes[:, [0, 2]] -= pad[0]
    all_boxes[:, [1, 3]] -= pad[1]
    all_boxes /= ratio

    if debug_writer is not None and debug_img_hw is not None:
        img_h, img_w = debug_img_hw
        mask_dbg = conf >= _DEBUG_RAW_CONF_THRES
        if int(mask_dbg.sum()) > _DEBUG_RAW_MAX_ROWS:
            idx_sorted = np.argsort(conf[mask_dbg])[::-1][:_DEBUG_RAW_MAX_ROWS]
            idx_full = np.where(mask_dbg)[0][idx_sorted]
            mask_dbg = np.zeros_like(mask_dbg)
            mask_dbg[idx_full] = True
        _log_boxes(
            debug_writer, "raw_preds", debug_pass_name,
            all_boxes[mask_dbg], conf[mask_dbg],
            img_w, img_h, "raw",
        )

    mask = (
        (cls == class_id) & (conf >= conf_thres)
        if class_id is not None
        else conf >= conf_thres
    )
    boxes = all_boxes[mask]
    conf_f = conf[mask]
    if len(boxes) == 0:
        return []

    keep = _nms_numpy(boxes, conf_f, iou_thres)
    boxes, conf_f = boxes[keep], conf_f[keep]
    boxes, conf_f = _dedup_by_center(boxes, conf_f, dist_thres=15.0)

    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
        for b, c in zip(boxes, conf_f)
    ]


def _single_pass_on_image(
    model, image_bgr: np.ndarray, conf_thres: float, iou_thres: float,
    class_id=None, debug_writer=None, debug_pass_name="", debug_img_hw=None,
):
    size = _MODEL_INPUT_SIZE[0]
    tensor, ratio, pad = _to_tensor(image_bgr, size)
    try:
        with torch.inference_mode():
            output = model(tensor)
    except Exception:
        if size != _FALLBACK_IMG_SIZE:
            _MODEL_INPUT_SIZE[0] = _FALLBACK_IMG_SIZE
            tensor, ratio, pad = _to_tensor(image_bgr, _FALLBACK_IMG_SIZE)
            with torch.inference_mode():
                output = model(tensor)
        else:
            raise
    return _decode_output(
        output, ratio, pad,
        conf_thres=conf_thres, iou_thres=iou_thres, class_id=class_id,
        debug_writer=debug_writer, debug_pass_name=debug_pass_name,
        debug_img_hw=debug_img_hw,
    )


def detect_stamps(
    model: torch.jit.ScriptModule,
    image,
    output_path: str | None = None,
    conf_thres: float = 0.5,
    iou_thres: float = 0.45,
    bottom_frac: float = DEFAULT_BOTTOM_FRAC,
    min_aspect: float = 0.3,
    max_aspect: float = 3.0,
    min_area_frac: float = 0.0005,
    max_area_frac: float = 0.20,
    use_cascade_tta: bool = True,
    cascade_stop_strong: float = 0.60,
    cascade_stop_after_merge: float = 0.50,
    remove_long_lines: bool = DEFAULT_REMOVE_LONG_LINES,
    class_id: int | None = None,
    a4_output_coords: bool = False,
    ink_bgr: tuple[int, int, int] | None = None,
    debug: bool = DEBUG_ENABLED,
) -> list[dict]:
    """
    Два recolor-прохода (neutral + blue) по одному общему тяжёлому
    препроцессу.

    Зачем два: модель не color-invariant. Упрямые синие печати ловятся
    только на blue-варианте ((130,70,40) → B-R градиент в каналах);
    чёрные/серые/бледные — на neutral. Слияние через NMS+dedup.

    По времени: 2 forward + 1 CLAHE ≈ 1.6× от одного прохода,
    а не 2× (тяжёлый препроцесс посчитан один раз).

    use_cascade_tta=False → только neutral (отладочный режим).
    ink_bgr задан → только этот цвет (для sweep).
    """
    image_bgr = _as_bgr_numpy(image)
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Пустое изображение")

    debug_f = None
    debug_w = None
    if debug:
        debug_f, debug_w, _ = _open_debug_csv(DEBUG_DIR)

    try:
        h_orig, w_orig = image_bgr.shape[:2]

        # --- 1) Кроп по bottom_frac на исходном ---
        bottom_frac_clamped = max(0.1, min(1.0, float(bottom_frac)))
        y_crop = int(round(h_orig * (1.0 - bottom_frac_clamped)))
        if h_orig - y_crop < 32:
            y_crop = max(0, h_orig - 32)
        image_cropped = image_bgr[y_crop:, :]

        # --- 2) A4-скейл ---
        if SCALE_TO_A4:
            image_scaled, a4_scale = _scale_to_a4(
                image_cropped,
                target_short=A4_TARGET_SHORT_SIDE_PX,
                tolerance=None,
            )
        else:
            image_scaled, a4_scale = image_cropped, 1.0

        H, W = image_scaled.shape[:2]
        tile_grid = _clahe_grid_for((H, W))
        tile_px_eff = _clahe_tile_px_for((H, W))
        tol_eff = _a4_tolerance_for(A4_TARGET_SHORT_SIDE_PX)

        # --- 3) Список вариантов ---
        if ink_bgr is not None:
            active_variants = (("custom", tuple(ink_bgr)),)
        elif use_cascade_tta:
            active_variants = _STAMP_COLOR_VARIANTS
        else:
            active_variants = (_STAMP_COLOR_VARIANTS[0],)

        if debug_w is not None:
            debug_w.writerow([
                "meta", "", "", "", "", "", "", "", "", "", "", "", "",
                f"input_w={w_orig} input_h={h_orig} "
                f"y_crop={y_crop} bottom_frac={bottom_frac_clamped:.3f} "
                f"a4_target={A4_TARGET_SHORT_SIDE_PX} "
                f"a4_tol={tol_eff} a4_scale={a4_scale:.4f} "
                f"a4_w={W} a4_h={H} "
                f"clahe_tile_cells={tile_grid[0]}x{tile_grid[1]} "
                f"clahe_tile_px={tile_px_eff:.1f} "
                f"remove_long_lines={remove_long_lines} "
                f"passes={[n for n, _ in active_variants]} "
                f"mode=two_color_one_prep",
            ])
            debug_f.flush()

        # --- 4) Один тяжёлый препроцесс ---
        soft = _prepare_gray_for_stamps(
            image_scaled, remove_long_lines=remove_long_lines
        )

        local_conf = min(conf_thres, 0.20)
        local_iou = min(iou_thres, 0.35)

        # --- 5) N recolor'ов + N forward'ов ---
        all_dets: list[tuple[float, float, float, float, float]] = []
        for name, ink in active_variants:
            proc = _recolor_from_soft(soft, ink)
            dets = _single_pass_on_image(
                model,
                proc,
                conf_thres=local_conf,
                iou_thres=local_iou,
                class_id=class_id,
                debug_writer=debug_w,
                debug_pass_name=name,
                debug_img_hw=(H, W),
            )
            all_dets.extend(dets)

        if debug_f is not None:
            debug_f.flush()

        if not all_dets:
            return []

        # --- 6) Слияние между проходами ---
        boxes = np.array([d[:4] for d in all_dets], np.float32)
        scores = np.array([d[4] for d in all_dets], np.float32)

        keep_idx = _nms_numpy(boxes, scores, iou_thres=_TTA_NMS_IOU)
        boxes, scores = boxes[keep_idx], scores[keep_idx]
        _log_boxes(debug_w, "merged_nms", "merged", boxes, scores,
                   W, H, "kept", f"iou={_TTA_NMS_IOU}")

        boxes, scores = _dedup_by_center(boxes, scores, dist_thres=_TTA_DEDUP_DIST)
        _log_boxes(debug_w, "merged_dedup", "merged", boxes, scores,
                   W, H, "kept", f"dedup center<{_TTA_DEDUP_DIST}")

        boxes, scores = _remove_containing_boxes(
            boxes, scores, area_ratio_thres=_TTA_CONTAINMENT_RATIO
        )
        _log_boxes(debug_w, "merged_containment", "merged", boxes, scores,
                   W, H, "kept",
                   f"remove_containing area<{_TTA_CONTAINMENT_RATIO}")

        # --- 7) Финальные фильтры ---
        keep = []
        for b, conf in zip(boxes, scores):
            x1, y1, x2, y2 = b
            w_box, h_box = x2 - x1, y2 - y1
            ar = (w_box / h_box) if h_box > 0 else 0.0
            area_frac = (w_box * h_box) / (W * H) if (W * H) > 0 else 0.0

            if conf < conf_thres:
                _log_bbox(debug_w, "filter_conf", "merged", -1,
                          x1, y1, x2, y2, conf, W, H, "rejected",
                          f"conf={conf:.3f} < conf_thres={conf_thres}")
                continue
            if w_box <= 1 or h_box <= 1:
                continue
            if not (min_aspect <= ar <= max_aspect):
                continue
            if not (min_area_frac <= area_frac <= max_area_frac):
                continue
            keep.append((x1, y1, x2, y2, conf))

        # --- 8) Центрированное уменьшение bbox ---
        if STAMP_BBOX_SCALE != 1.0 and keep:
            s = float(STAMP_BBOX_SCALE)
            shrunk = []
            for x1, y1, x2, y2, conf in keep:
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                nw, nh = (x2 - x1) * s, (y2 - y1) * s
                shrunk.append((
                    cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2, conf,
                ))
            keep = shrunk

        # --- 9) Координаты ---
        if a4_output_coords:
            inv_scale = 1.0
            y_shift = 0.0
        else:
            inv_scale = (1.0 / a4_scale) if a4_scale > 0 else 1.0
            y_shift = float(y_crop)

        result = [
            {
                "confidence": float(c),
                "bbox": [
                    float(x1 * inv_scale),
                    float(y1 * inv_scale + y_shift),
                    float(x2 * inv_scale),
                    float(y2 * inv_scale + y_shift),
                ],
            }
            for x1, y1, x2, y2, c in keep
        ]
        result.sort(key=lambda r: r["confidence"], reverse=True)

        # --- 10) Debug: финальные строки ---
        if debug_w is not None:
            for i, item in enumerate(result):
                b = item["bbox"]
                debug_w.writerow([
                    "final", "merged", i,
                    f"{b[0]:.2f}", f"{b[1]:.2f}", f"{b[2]:.2f}", f"{b[3]:.2f}",
                    f"{item['confidence']:.4f}",
                    f"{b[2] - b[0]:.2f}", f"{b[3] - b[1]:.2f}",
                    f"{(b[2] - b[0]) / max(b[3] - b[1], 1e-9):.3f}",
                    "",
                    "kept",
                    f"coords={'A4_crop' if a4_output_coords else 'original'}",
                ])

        return result

    finally:
        if debug_f is not None:
            try:
                debug_f.flush()
                debug_f.close()
            except Exception:
                pass