from __future__ import annotations
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = "google/siglip-base-patch16-256-multilingual"
DEBUG_ENABLED = False
DEBUG_DIR = SCRIPT_DIR / "debug"

# --- Пороги подписей ---
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_DATE_MARGIN = 0.10
DEFAULT_TRASH_MARGIN = 0.05
DEFAULT_DATE_HARD_REJECT_PROB = 0.25
DEFAULT_TRASH_HARD_REJECT_PROB = 0.40
DEFAULT_STAMP_HARD_REJECT_PROB = 0.55
DEFAULT_DATE_HARD_REJECT_MATCH = 0.60
DEFAULT_TRASH_HARD_REJECT_MATCH = 0.90
DEFAULT_TRASH_HARD_OVERRIDE_MARGIN = 0.10
DEFAULT_SIG_ON_STAMP_PROB = 0.30
DEFAULT_SIG_ON_STAMP_MATCH = 0.55

STAMP_MIN_CONF_FOR_IOU = 0.40
STAMP_HEAVY_OVERLAP_IOU = 0.60
STAMP_MEDIUM_OVERLAP_IOU = 0.50
STAMP_GEOMETRIC_SIG_RELAX = 0.80
STAMP_HARD_REJECT_P_SIG = 0.15

SIG_ON_STAMP_SIGMOID_MIN = 0.90
SIG_ON_STAMP_SIGMOID_CONF = 0.05
SIG_ON_STAMP_SIGMOID_MARGIN = 0.15
SIG_ON_STAMP_SIGMOID_TRASH_GAP = 0.15
SIG_ON_STAMP_SIGMOID_STAMP_CONF = 0.70

TRASH_HARD_SIGMOID_MAX_SIG = 0.75

SIGMOID_OVERRIDE_MIN_SIG = 0.90
SIGMOID_OVERRIDE_MAX_DATE = 0.02
SIGMOID_OVERRIDE_MAX_STAMP = 0.15
SIGMOID_OVERRIDE_MIN_SIG_STAMP_GAP = 0.05

SOFTMAX_ARTIFACT_MIN_PSIG = 0.70
SOFTMAX_ARTIFACT_MAX_PSIG = 0.90
SOFTMAX_ARTIFACT_MAX_MSIG = 0.95
SOFTMAX_ARTIFACT_MIN_TRASH = 0.40
SOFTMAX_ARTIFACT_MIN_DATE = 0.20

LOW_MSIG_THRESHOLD = 0.50
LOW_MSIG_MAX_PSIG = 0.90
LOW_MSIG_MIN_PTRASH = 0.25

# --- Пороги штампов ---
STAMP_VERIFY_MIN_P_STAMP = 0.05

TOP_K_PROMPTS = 3
PAD_W_MIN = 15
PAD_H_MIN = 10
PAD_W_FRAC = 0.008
PAD_H_FRAC = 0.004


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-9)


class SigLIPMultilingualClassifier:
    def __init__(self, debug: bool = DEBUG_ENABLED):
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[siglip_multilingual] Загружаем модель '{MODEL_ID}' на устройство '{self.device}'...")

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            MODEL_ID, attn_implementation="sdpa"
        ).to(self.device).eval()

        self.prompts = {
            "signature": [
                "рукописная подпись человека", "личный автограф под текстом документа",
                "подпись в виде сокращенной фамилии",
                "размашистый росчерк шариковой ручкой", "слитная волнистая линия от руки",
                "сложная каллиграфическая подпись",
                "подпись перьевой ручкой на документе", "подпись с петлями и завитками",
                "непрерывные кривые линии образующие подпись",
                "замысловатая рукописная подпись", "несколько связанных штрихов образующих автограф",
                "плавная непрерывная линия без печатных букв",
                "подпись черной ручкой поверх печати", "росчерк ручкой на фоне штампа",
                "автограф поперек круглой печати",
                "рукописная подпись, перекрывающая оттиск печати", "крупный автограф поперек синей печати организации",
                "росчерк рукой поперек черно-белого оттиска печати",
                "подпись, нанесенная поверх штампа организации", "черно-белый скан рукописной подписи",
                "монохромное изображение подписи от руки",
                "бледная подпись ручкой на документе", "тонкая прерывистая подпись от руки",
                "слабая подпись на светлом скане",
                "еле заметный росчерк ручкой", "подпись, нарисованная отдельными штрихами",
                "стилизованная подпись из коротких элементов",
                "росчерк с пробелами между штрихами", "буквы подписи нарисованные по отдельности",
                "автограф из раздельных кривых линий",
            ],
            "date": [
                "рукописные цифры ручкой на бумаге", "число месяца цифрами написанное от руки", "две цифры через точку",
                "цифры разделенные точками", "дата в формате число точка месяц",
                "дата в формате число точка месяц точка год",
                "рукописная дата под текстом документа", "несколько цифр написанных слитно ручкой",
                "одна рукописная цифра",
                "две рукописные цифры рядом", "короткая числовая запись от руки", "черно-белый скан рукописных цифр",
                "размашисто написанные цифры от руки", "число написанное с петлей", "рукописная дата с длинным хвостом",
                "пара цифр с завитком",
            ],
            "stamp": [
                "оттиск круглой печати", "черно-белый оттиск печати", "синий оттиск штампа",
                "печать организации с гербом",
                "текст по окружности печати", "зубчатый край печати", "фрагмент круглой печати без подписи",
                "чистая печать без рукописных элементов", "оттиск печати без подписи", "штамп организации на документе",
                "бледный ч/б оттиск печати",
                # --- Добавляем для бледных печатей ---
                "очень бледный оттиск печати", "слабый след от печати на бумаге", "еле заметная круглая печать",
                "выцветший штамп организации",
            ],
            "trash": [
                "блок печатного текста", "машинописный текст", "прямая строка печатных букв",
                "одиночная печатная буква",
                "одно печатное слово", "печатные буквы М.П.", "текст место печати без рамки",
                "вертикальная черная линия",
                "горизонтальная черная линия", "короткий штрих", "одинокие мелкие символы, точки, запятые",
                "подчеркивание",
                "галочка", "квадратная рамка", "прямоугольник с печатным текстом", "печатные цифры в прямоугольнике",
                "одиночные печатные цифры", "логотип компании", "логотип и текст рядом", "штрих-код", "QR-код",
                "чистый лист бумаги", "пустое белое поле", "случайные пятна и шум", "черно-белый шум сканирования",
                "разбросанные черные пятна и фрагменты букв", "фрагмент печатного документа",
                "текст М.П. напечатанный точками",
                "матричный текст", "точечный принтер", "пунктирный шрифт", "точечные буквы", "размытый мелкий мусор",
                "артефакт сканирования", "обрывок печатной строки", "одиночный элемент печатного текста",
            ],
        }

        self._class_names = list(self.prompts.keys())
        self._sig_idx = self._class_names.index("signature")
        self._date_idx = self._class_names.index("date")
        self._stamp_idx = self._class_names.index("stamp")
        self._trash_idx = self._class_names.index("trash")

        self._flat_prompts = []
        self._prompt_class_idx = []
        for cls_idx, (_, plist) in enumerate(self.prompts.items()):
            for p in plist:
                self._flat_prompts.append(p)
                self._prompt_class_idx.append(cls_idx)

        self._prompt_idx_per_class: list[list[int]] = [[] for _ in self._class_names]
        for p_idx, cls_idx in enumerate(self._prompt_class_idx):
            self._prompt_idx_per_class[cls_idx].append(p_idx)

        print(f"[siglip_multilingual] Классов: {len(self._class_names)}, промптов: {len(self._flat_prompts)}")
        print("[siglip_multilingual] Предвычисляем эмбеддинги текстовых промптов...")
        text_inputs = self.processor(text=self._flat_prompts, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            self.text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            self.logit_scale = self.model.logit_scale.exp()
            self.logit_bias = self.model.logit_bias
        if self.debug:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    def _crop(self, image_pil, bbox):
        W, H = image_pil.size
        x1, y1, x2, y2 = bbox
        pad_w = max(PAD_W_MIN, int(W * PAD_W_FRAC))
        pad_h = max(PAD_H_MIN, int(H * PAD_H_FRAC))
        cx1 = max(0, int(x1) - pad_w)
        cy1 = max(0, int(y1) - pad_h)
        cx2 = min(W, int(x2) + pad_w)
        cy2 = min(H, int(y2) + pad_h)
        return image_pil.crop((cx1, cy1, cx2, cy2)).convert("RGB")

    def _score_crops(self, crops: list[Image.Image]):
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            logits_per_image = torch.matmul(image_features, self.text_features.T) * self.logit_scale + self.logit_bias

        n_crops = logits_per_image.shape[0]
        n_classes = len(self._class_names)
        class_logits = torch.full((n_crops, n_classes), -float("inf"), device=self.device)

        for cls_idx, pidx in enumerate(self._prompt_idx_per_class):
            cls_logits = logits_per_image[:, pidx]
            k = min(TOP_K_PROMPTS, cls_logits.shape[1])
            topk_vals, _ = torch.topk(cls_logits, k, dim=-1)
            class_logits[:, cls_idx] = topk_vals.mean(dim=-1)

        softmax_probs = torch.softmax(class_logits, dim=-1)
        sigmoid_probs = torch.sigmoid(class_logits)
        return softmax_probs.cpu().numpy(), sigmoid_probs.cpu().numpy()

    def predict_all(
            self, image_pil: Image.Image, sig_bboxes: list[list[float]], stamp_dets: list[dict],
            stem: str = "", crop_indices: list[int] | None = None,
    ) -> tuple[list[tuple[str | None, float, str]], list[dict]]:

        dbg_dir = DEBUG_DIR / stem if stem else DEBUG_DIR
        if self.debug:
            dbg_dir.mkdir(parents=True, exist_ok=True)

        # 1. Готовим кропы подписей
        sig_crops, sig_indices = [], []
        for i, bbox in enumerate(sig_bboxes):
            x1, x2 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
            y1, y2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue
            crop_img = self._crop(image_pil, (x1, y1, x2, y2))
            sig_crops.append(crop_img)
            sig_indices.append(i)
            real_idx = crop_indices[i] if crop_indices else i
            if self.debug and stem:
                crop_img.save(dbg_dir / f"siglip_raw_{real_idx}.jpg")

        # 2. Готовим кропы штампов (с CLAHE)
        stamp_crops, stamp_indices = [], []
        for i, det in enumerate(stamp_dets):
            bbox = det.get("bbox")
            if bbox is None: continue
            x1, x2 = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
            y1, y2 = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue
            crop_img = self._crop(image_pil, (x1, y1, x2, y2))

            img_np = np.array(crop_img)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            crop_img_enhanced = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

            stamp_crops.append(crop_img_enhanced)
            stamp_indices.append(i)
            if self.debug and stem:
                crop_img.save(dbg_dir / f"stamp_verify_{i}.jpg")
                crop_img_enhanced.save(dbg_dir / f"stamp_verify_{i}_enhanced.jpg")

        # 3. Один инференс для всего
        all_crops = sig_crops + stamp_crops
        if not all_crops:
            return [(None, 0.0, "error")] * len(sig_bboxes), [{"is_stamp": False, "reason": "empty"} for _ in
                                                              stamp_dets]

        try:
            sm_all, sigm_all = self._score_crops(all_crops)
        except Exception as e:
            print(f"      [SIGLIP-MULTI-DEBUG] Ошибка батча ({len(all_crops)} кропов): {e}")
            return [(None, 0.0, "error")] * len(sig_bboxes), [{"is_stamp": False, "reason": "error"} for _ in
                                                              stamp_dets]

        # 4. Разделяем результаты
        sm_sig = sm_all[:len(sig_crops)]
        sigm_sig = sigm_all[:len(sig_crops)]
        sm_stamp = sm_all[len(sig_crops):]

        # 5. Обработка результатов ПОДПИСЕЙ (старая логика)
        sig_results: list[tuple[str | None, float, str]] = [(None, 0.0, "error")] * len(sig_bboxes)
        for k, idx in enumerate(sig_indices):
            s = sm_sig[k]
            sigm = sigm_sig[k]
            p_sig, p_date, p_stamp, p_trash = float(s[self._sig_idx]), float(s[self._date_idx]), float(
                s[self._stamp_idx]), float(s[self._trash_idx])
            m_sig, m_date, m_stamp, m_trash = float(sigm[self._sig_idx]), float(sigm[self._date_idx]), float(
                sigm[self._stamp_idx]), float(sigm[self._trash_idx])
            best = int(s.argmax())
            best_score = float(s[best])
            best_name = self._class_names[best]

            stamp_iou = 0.0
            bbox = sig_bboxes[idx]
            if stamp_dets:
                for st in stamp_dets:
                    if st.get("confidence", 1.0) < STAMP_MIN_CONF_FOR_IOU: continue
                    stamp_iou = max(stamp_iou, _iou(bbox, st["bbox"]))

            sig_is_best = (best_name == "signature")
            sig_on_stamp_visual = ((best_name == "stamp") and (p_sig >= DEFAULT_SIG_ON_STAMP_PROB) and (
                        m_sig >= DEFAULT_SIG_ON_STAMP_MATCH) and (p_sig >= p_trash))
            sig_on_stamp_geometric = ((stamp_iou >= STAMP_MEDIUM_OVERLAP_IOU) and (
                        p_sig >= DEFAULT_CONF_THRESHOLD * STAMP_GEOMETRIC_SIG_RELAX) and (p_sig >= p_trash))
            sig_on_stamp_sigmoid = ((best_name == "stamp") and (m_sig >= SIG_ON_STAMP_SIGMOID_MIN) and (
                        p_sig >= SIG_ON_STAMP_SIGMOID_CONF) and (p_sig >= p_trash) and (
                                                p_stamp >= SIG_ON_STAMP_SIGMOID_STAMP_CONF) and (
                                                m_sig - m_trash >= SIG_ON_STAMP_SIGMOID_TRASH_GAP))
            sig_on_stamp_high_sigmoid = (
                        (best_name == "stamp") and (m_sig >= 0.95) and (p_stamp >= 0.70) and (p_sig >= p_trash))
            sig_on_stamp = sig_on_stamp_visual or sig_on_stamp_geometric or sig_on_stamp_sigmoid or sig_on_stamp_high_sigmoid

            sig_sigmoid_override = (best_name == "trash" and m_sig >= SIGMOID_OVERRIDE_MIN_SIG and (
                        m_sig - m_stamp) >= SIGMOID_OVERRIDE_MIN_SIG_STAMP_GAP and p_date < SIGMOID_OVERRIDE_MAX_DATE and p_stamp < SIGMOID_OVERRIDE_MAX_STAMP)
            reject_date = ((p_date >= DEFAULT_DATE_HARD_REJECT_PROB) or (
                        m_date >= DEFAULT_DATE_HARD_REJECT_MATCH and m_date > m_sig and not sig_is_best and not sig_sigmoid_override))
            sig_beats_trash = (best_name == "signature" and (p_sig - p_trash) >= DEFAULT_TRASH_HARD_OVERRIDE_MARGIN)
            reject_trash = (((
                                         p_trash >= DEFAULT_TRASH_HARD_REJECT_PROB) and not sig_beats_trash and not sig_sigmoid_override) or (
                                        m_trash >= DEFAULT_TRASH_HARD_REJECT_MATCH and m_trash > m_sig and m_sig <= TRASH_HARD_SIGMOID_MAX_SIG and not sig_is_best and not sig_sigmoid_override))
            stamp_hard_reject = ((best_name == "stamp") and (p_sig < STAMP_HARD_REJECT_P_SIG) and (
                        p_stamp >= DEFAULT_STAMP_HARD_REJECT_PROB) and (m_sig < SIG_ON_STAMP_SIGMOID_MIN))
            if (stamp_iou >= STAMP_HEAVY_OVERLAP_IOU) and (p_sig < STAMP_HARD_REJECT_P_SIG) and not sig_is_best:
                stamp_hard_reject = True
            reject_softmax_artifact = (
                        sig_is_best and SOFTMAX_ARTIFACT_MIN_PSIG <= p_sig < SOFTMAX_ARTIFACT_MAX_PSIG and m_sig < SOFTMAX_ARTIFACT_MAX_MSIG and m_trash >= SOFTMAX_ARTIFACT_MIN_TRASH and m_date >= SOFTMAX_ARTIFACT_MIN_DATE)
            reject_low_msig = (
                        sig_is_best and m_sig < LOW_MSIG_THRESHOLD and p_sig < LOW_MSIG_MAX_PSIG and p_trash >= LOW_MSIG_MIN_PTRASH)
            hard_reject = reject_date or reject_trash or stamp_hard_reject or reject_softmax_artifact or reject_low_msig

            effective_conf = DEFAULT_CONF_THRESHOLD
            if sig_on_stamp_geometric and not sig_is_best: effective_conf = DEFAULT_CONF_THRESHOLD * STAMP_GEOMETRIC_SIG_RELAX
            if sig_on_stamp_sigmoid and not sig_is_best: effective_conf = min(effective_conf, SIG_ON_STAMP_SIGMOID_CONF)
            date_margin_ok = (p_sig - p_date) >= DEFAULT_DATE_MARGIN
            if sig_on_stamp_sigmoid: date_margin_ok = date_margin_ok or (
                        (m_sig - m_date) >= SIG_ON_STAMP_SIGMOID_MARGIN)
            trash_margin_ok = (p_sig - p_trash) >= DEFAULT_TRASH_MARGIN
            if sig_on_stamp_sigmoid: trash_margin_ok = trash_margin_ok or ((m_sig - m_trash) >= 0.0)

            sig_accept = sig_is_best or sig_on_stamp or sig_sigmoid_override
            conf_ok = (p_sig >= effective_conf) or sig_sigmoid_override
            date_ok = date_margin_ok or sig_sigmoid_override
            trash_ok = trash_margin_ok or sig_sigmoid_override
            accept = ((not hard_reject) and sig_accept and conf_ok and date_ok and trash_ok)

            if accept:
                reason = "sig_best" if sig_is_best else ("sig_on_stamp_geom" if sig_on_stamp_geometric else (
                    "sig_on_stamp_sigmoid" if sig_on_stamp_sigmoid else (
                        "sig_on_stamp_high_sig" if sig_on_stamp_high_sigmoid else "sig_on_stamp")))
                sig_results[idx] = ("signature", max(p_sig, best_score), reason)
            else:
                reason = "low_msig" if reject_low_msig else ("softmax_artifact" if reject_softmax_artifact else (
                    "stamp_hard" if stamp_hard_reject else ("date_hard" if reject_date else (
                        "trash_hard" if reject_trash else (
                            "stamp_dominates" if best_name == "stamp" else f"best_{best_name}")))))
                sig_results[idx] = (None, p_sig, reason)

            if self.debug:
                print(
                    f"      [SIGLIP-MULTI-DEBUG] (crop_{crop_indices[idx] if crop_indices else idx}.jpg) {'ACCEPT' if accept else 'REJECT'} [{reason}] | best={best_name} | stamp_iou={stamp_iou:.2f}")

        # 6. Обработка результатов ШТАМПОВ
        stamp_results: list[dict] = [{"is_stamp": False, "p_stamp": 0.0, "reason": "empty"} for _ in stamp_dets]
        for k, idx in enumerate(stamp_indices):
            s = sm_stamp[k]
            p_sig, p_date, p_stamp, p_trash = float(s[self._sig_idx]), float(s[self._date_idx]), float(
                s[self._stamp_idx]), float(s[self._trash_idx])
            best = int(s.argmax())
            best_name = self._class_names[best]

            is_stamp = False
            if p_stamp >= STAMP_VERIFY_MIN_P_STAMP:
                is_stamp = True
                reason = "stamp_present"
            elif p_sig > p_stamp and p_stamp < 0.10:
                reason = "pure_signature"
            elif p_trash > p_stamp and p_stamp < 0.10:
                reason = "pure_trash"
            else:
                reason = "low_confidence_drop"

            stamp_results[idx] = {"is_stamp": is_stamp, "p_stamp": p_stamp, "p_sig": p_sig, "p_trash": p_trash,
                                  "p_date": p_date, "best": best_name, "reason": reason}

            if self.debug:
                print(
                    f"      [STAMP-VERIFY-DEBUG] (stamp_{idx}) {'KEEP' if is_stamp else 'DROP'} [{reason}] | best={best_name} | p_sig={p_sig:.3f} p_stamp={p_stamp:.3f} p_trash={p_trash:.3f}")

        return sig_results, stamp_results


def load_signature_classifier(debug: bool = DEBUG_ENABLED) -> SigLIPMultilingualClassifier:
    return SigLIPMultilingualClassifier(debug=debug)