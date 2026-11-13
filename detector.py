import sys
from pathlib import Path
from PIL import Image

# =====================================================================
# 1. НАСТРОЙКА ПУТИ К ДЕТЕКТОРУ
# =====================================================================
# АБСОЛЮТНЫЙ путь к папке, где лежат detector_main.py и models/
DETECTOR_DIR = Path(r"C:\Users\rkokov\Desktop\yolo11\test\SIGN_STAMP_detector")  # <-- Замените на свой путь

# Добавляем папку в sys.path, чтобы Python нашел detector_main и его зависимости
if str(DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTOR_DIR))

# =====================================================================
# 2. ИМПОРТ ДЕТЕКТОРА
# =====================================================================
import detector_main

# (Опционально) Прогрев моделей при старте сервера.
# Если этого не сделать, первый вызов detect_pil_image займет на 2-3 секунды больше.
print("[DETECTOR] Загружаем модели в память (прогрев)...")
detector_main._get_models(debug_mode=False)
print("[DETECTOR] Модели готовы к работе.")


# =====================================================================
# 3. ФУНКЦИЯ ДЛЯ ВЫЗОВА
# =====================================================================
def detect_document(image_pil: Image.Image) -> dict:
    """
    Принимает PIL.Image, возвращает словарь с подписями и штампами.
    """
    # debug=False, save_output=False — тихий режим, ничего не пишет на диск
    result = detector_main.detect_pil_image(
        image_pil,
        debug=False,
        save_output=False
    )
    return result


# =====================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (если запускать файл напрямую для теста)
# =====================================================================
if __name__ == "__main__":
    # Создадим фейковую картинку для проверки (или загрузите реальную)
    test_img = Image.new("RGB", (1000, 1500), color="white")

    print("\nОтправляем картинку в детектор...")
    res = detect_document(test_img)

    print("\n--- РЕЗУЛЬТАТ ---")
    print(f"Подписей: {len(res['signatures'])}")
    print(f"Штампов:  {len(res['stamps'])}")
    for s in res["stamps"]:
        print(f"  Штамп: bbox={s['bbox']}, conf={s['confidence']:.3f}")