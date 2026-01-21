# keypoint_inference_fixed.py
import os
import json
from PIL import Image
from rex_omni import RexOmniVisualize, RexOmniWrapper
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------
# 1. Параметры
# ----------------------------------------------------------------------
MODEL_PATH = "IDEA-Research/Rex-Omni"
COCO_ANN = "coco-animal-1/test/_annotations.coco.json"
COCO_IMG_DIR = "coco-animal-1/test/"
OUTPUT_DIR = "/home/efimenko.aleksey7/rex/Rex-Omni/keypoint_results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Имена ключевых точек в МОДЕЛИ
MODEL_KEYPOINT_NAMES = [
        "left eye",
        "right eye",
        "nose",
        "neck",
        "root of tail",
        "left shoulder",
        "left elbow",
        "left front paw",
        "right shoulder",
        "right elbow",
        "right front paw",
        "left hip",
        "left knee",
        "left back paw",
        "right hip",
        "right knee",
        "right back paw",
    ]

# Имена ключевых точек в ДАТАСЕТЕ (из вашего анализа)
DATASET_KEYPOINT_NAMES = [
    "L_Eye", "R_Eye", "Nose", "Neck", "Root of tail", "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow","R_F_Paw", "L_Hip", "L_Knee",
    "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw"
]

# Маппинг из модели в датасет
KEYPOINT_MAPPING = {
        "left eye":"L_Eye",
        "right eye":"R_Eye",
        "nose":"Nose",
        "neck":"Neck",
        "root of tail":"Root of tail",
        "left shoulder":"L_Shoulder",
        "left elbow":"L_Elbow",
        "left front paw":"L_F_Paw",
        "right shoulder":"R_Shoulder",
        "right elbow":"R_Elbow",
        "right front paw":"R_F_Paw",
        "left hip":"L_Hip",
        "left knee":"L_Knee",
        "left back paw":"L_B_Paw",
        "right hip":"R_Hip",
        "right knee":"R_Knee",
        "right back paw":"R_B_Paw",
}

# Обратный маппинг (из датасета в модель)
REVERSE_MAPPING = {v: k for k, v in KEYPOINT_MAPPING.items()}

# ----------------------------------------------------------------------
# 2. Функции для преобразования ключевых точек
# ----------------------------------------------------------------------
def convert_model_to_dataset_keypoints(model_keypoints_dict):
    """Преобразует ключевые точки из формата модели в формат датасета"""
    dataset_keypoints_dict = {}
    
    for model_name, dataset_name in KEYPOINT_MAPPING.items():
        if model_name in model_keypoints_dict:
            dataset_keypoints_dict[dataset_name] = model_keypoints_dict[model_name]
    
    return dataset_keypoints_dict

def convert_dataset_to_model_keypoints(dataset_keypoints_list, dataset_keypoint_names):
    """Преобразует плоский список ключевых точек датасета в словарь формата модели"""
    # Преобразуем плоский список в словарь с именами датасета
    dataset_kp_dict = {}
    for i, name in enumerate(dataset_keypoint_names):
        x = dataset_keypoints_list[i * 3]
        y = dataset_keypoints_list[i * 3 + 1]
        visibility = dataset_keypoints_list[i * 3 + 2]
        dataset_kp_dict[name] = [x, y, visibility]
    
    # Конвертируем в формат модели
    model_kp_dict = {}
    for dataset_name, model_name in REVERSE_MAPPING.items():
        if dataset_name in dataset_kp_dict:
            model_kp_dict[model_name] = dataset_kp_dict[dataset_name]
    
    return model_kp_dict

def create_coco_format_keypoints(model_keypoints_dict, keypoint_names):
    """Создает ключевые точки в COCO формате из словаря"""
    kpts_flat = []
    for name in keypoint_names:
        if name in model_keypoints_dict:
            x, y, vis = model_keypoints_dict[name]
            kpts_flat.extend([x, y, vis])
        else:
            kpts_flat.extend([0, 0, 0])  # Если точка отсутствует
    
    return kpts_flat

# ----------------------------------------------------------------------
# 3. Обновленная функция вычисления OKS с учетом маппинга
# ----------------------------------------------------------------------
def compute_oks_with_mapping(dt_kpts, gt_kpts, area, dt_keypoint_names, gt_keypoint_names):
    """
    Compute Object Keypoint Similarity с учетом разных форматов ключевых точек
    """
    sigmas = COCO_SIGMAS
    vars = (sigmas * 2) ** 2
    k = len(sigmas)
    
    # Конвертируем GT в формат модели
    gt_kp_dict = convert_dataset_to_model_keypoints(gt_kpts, gt_keypoint_names)
    
    # Преобразуем предсказание в словарь
    dt_array = np.array(dt_kpts).reshape(k, 3)
    dt_kp_dict = {}
    for i, name in enumerate(MODEL_KEYPOINT_NAMES):
        dt_kp_dict[name] = [dt_array[i, 0], dt_array[i, 1], dt_array[i, 2]]
    
    # Считаем расстояния для соответствующих точек
    errors = []
    vis_counts = 0
    
    for i, model_name in enumerate(MODEL_KEYPOINT_NAMES):
        if model_name in dt_kp_dict and model_name in gt_kp_dict:
            dt_x, dt_y, dt_vis = dt_kp_dict[model_name]
            gt_x, gt_y, gt_vis = gt_kp_dict[model_name]
            
            # Учитываем только видимые точки в GT
            if gt_vis > 0:
                dx = dt_x - gt_x
                dy = dt_y - gt_y
                e = (dx ** 2 + dy ** 2) / vars[i] / (area + np.spacing(1)) / 2
                errors.append(np.exp(-e))
                vis_counts += 1
    
    if vis_counts == 0:
        return 0.0
    
    oks = np.sum(errors) / vis_counts
    return oks

# ----------------------------------------------------------------------
# 4. Обновленная функция расчета метрик
# ----------------------------------------------------------------------
def calculate_metrics_with_mapping(predictions, coco_annotations, processed_img_ids):
    """
    Правильный расчет метрик с Hungarian matching для сопоставления предсказаний и GT
    """
    # Группируем GT по image_id
    gt_by_image = {}
    for ann in coco_annotations:
        if ann["category_id"] == 1 and ann["image_id"] in processed_img_ids:
            img_id = ann["image_id"]
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append(ann)
    
    # Пороги для OKS
    oks_thresholds = np.linspace(0.5, 0.95, 10)
    
    # Для каждого порога будем считать TP, FP, FN
    all_tp = {thresh: 0 for thresh in oks_thresholds}
    all_fp = {thresh: 0 for thresh in oks_thresholds} 
    all_fn = {thresh: 0 for thresh in oks_thresholds}
    
    matched_predictions = []
    
    # Обрабатываем каждое изображение отдельно
    for img_id in processed_img_ids:
        if img_id not in gt_by_image:
            continue
            
        img_gts = gt_by_image[img_id]
        img_preds = [p for p in predictions if p["image_id"] == img_id]
        
        if not img_preds or not img_gts:
            continue
        
        # Строим матрицу стоимостей (1 - OKS) для Hungarian matching
        cost_matrix = np.zeros((len(img_preds), len(img_gts)))
        
        for i, pred in enumerate(img_preds):
            for j, gt in enumerate(img_gts):
                bbox = gt["bbox"]
                area = bbox[2] * bbox[3]
                oks = compute_oks_with_mapping(
                    pred["keypoints"], gt["keypoints"], area,
                    MODEL_KEYPOINT_NAMES, DATASET_KEYPOINT_NAMES
                )
                cost_matrix[i, j] = 1 - oks  # преобразуем в стоимость
        
        # Hungarian assignment
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Для каждого порога считаем статистику
        for threshold in oks_thresholds:
            tp = 0
            matched_gt = set()
            matched_pred = set()
            
            # Считаем true positives
            for i, j in zip(pred_indices, gt_indices):
                oks_score = 1 - cost_matrix[i, j]
                if oks_score >= threshold:
                    tp += 1
                    matched_gt.add(j)
                    matched_pred.add(i)
                    
                    # Сохраняем информацию о матчинге для первого порога
                    if threshold == oks_thresholds[0]:
                        matched_predictions.append({
                            "pred": img_preds[i],
                            "gt": img_gts[j], 
                            "oks": oks_score,
                            "image_id": img_id
                        })
            
            fp = len(img_preds) - len(matched_pred)  # несовпавшие предсказания
            fn = len(img_gts) - len(matched_gt)      # необнаруженные GT
            
            all_tp[threshold] += tp
            all_fp[threshold] += fp  
            all_fn[threshold] += fn
    
    # Вычисляем метрики
    metrics = {}
    ap_scores = []
    ar_scores = []
    oks_scores = [m["oks"] for m in matched_predictions] if matched_predictions else [0]
    
    for threshold in oks_thresholds:
        tp = all_tp[threshold]
        fp = all_fp[threshold]
        fn = all_fn[threshold]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ap_scores.append(precision)
        ar_scores.append(recall)
    
    if matched_predictions:
        metrics["AP"] = np.mean(ap_scores)
        metrics["AP_50"] = ap_scores[0]
        metrics["AP_75"] = ap_scores[5] 
        metrics["AR"] = np.mean(ar_scores)
        metrics["mOKS"] = np.mean(oks_scores)
        metrics["OKS_std"] = np.std(oks_scores)
        metrics["total_matches"] = len(matched_predictions)
        metrics["total_gt"] = sum(len(gts) for gts in gt_by_image.values())
        metrics["total_preds"] = len(predictions)
    else:
        metrics = {
            "AP": 0, "AP_50": 0, "AP_75": 0, "AR": 0, 
            "mOKS": 0, "OKS_std": 0, "total_matches": 0,
            "total_gt": sum(len(gts) for gts in gt_by_image.values()),
            "total_preds": len(predictions)
        }
    
    return metrics, matched_predictions

# ----------------------------------------------------------------------
# 5. Обновленный основной цикл
# ----------------------------------------------------------------------
print("Initializing Rex Omni model...")
rex = RexOmniWrapper(
    model_path=MODEL_PATH,
    backend="transformers",
    max_tokens=4096,
    temperature=0.75,
    top_p=0.50,
    top_k=10,
    repetition_penalty=1,
)

print("Loading COCO annotations...")
with open(COCO_ANN, "r") as f:
    coco = json.load(f)

img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
person_img_ids = {ann["image_id"] for ann in coco["annotations"] if ann["category_id"] == 1}
person_img_ids = list(person_img_ids)[:101]
print(f"Processing {len(person_img_ids)} images")

# Сигмы для OKS (Object Keypoint Similarity) из COCO
COCO_SIGMAS = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035,
    0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107,
    0.087, 0.087, 0.089, 0.089
])

predictions = []
gt_by_img = {}

print("Starting inference...")
for img_id in tqdm(person_img_ids):
    file_name = img_id_to_file[img_id]
    img_path = os.path.join(COCO_IMG_DIR, file_name)
    if not os.path.isfile(img_path): 
        continue

    img = Image.open(img_path).convert("RGB")

    # --- Инференс ---
    res = rex.inference(images=img, task="keypoint", keypoint_type="animal", categories=["animal"])
    if not res[0]["success"]: 
        continue

    extracted = res[0]["extracted_predictions"]
    person_preds = extracted.get("animal", [])
    if not person_preds: 
        continue

    # --- Масштабирование ---
    orig_w, orig_h = res[0]["image_size"]
    resized_w, resized_h = res[0]["resized_size"]
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    # --- COCO-формат с правильным порядком ключевых точек ---
    for p in person_preds:
        kp_dict = p.get("keypoints", {})
        
        # Масштабируем координаты
        scaled_kp_dict = {}
        for name, coords in kp_dict.items():
            if len(coords) >= 2:
                x = coords[0] * scale_x
                y = coords[1] * scale_y
                scaled_kp_dict[name] = [x, y, 2.0]  # visibility = 2.0
        
        # Создаем ключевые точки в правильном порядке (формат модели)
        kpts_flat = create_coco_format_keypoints(scaled_kp_dict, MODEL_KEYPOINT_NAMES)
        
        if len(kpts_flat) == 51:  # 17 points * 3
            predictions.append({
                "image_id": img_id,
                "category_id": 1,
                "keypoints": kpts_flat,
                "score": 1.0
            })

    # --- Визуализация ---
    vis = RexOmniVisualize(
        image=img,
        predictions=extracted,
        font_size=8,
        draw_width=3,
        show_labels=True,
    )
    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}_kpts.jpg")
    vis.save(out_path)

    # --- GT ---
    gt_by_img[img_id] = [ann for ann in coco["annotations"] if ann["image_id"] == img_id and ann["category_id"] == 1]

# ----------------------------------------------------------------------
# 6. Расчет и вывод метрик с правильным маппингом
# ----------------------------------------------------------------------
print("\n" + "="*50)
print("РАСЧЕТ МЕТРИК С ПРАВИЛЬНЫМ МАППИНГОМ")
print("="*50)

# Получаем все аннотации для людей
all_gt_annotations = [ann for ann in coco["annotations"] if ann["category_id"] == 1]

# Вычисляем метрики с учетом маппинга
metrics, matches = calculate_metrics_with_mapping(predictions, all_gt_annotations, person_img_ids)

print(f"Обработано изображений: {len(person_img_ids)}")
print(f"Найдено предсказаний: {len(predictions)}")
print(f"Всего GT аннотаций: {len(all_gt_annotations)}")

if metrics:
    print("\nОСНОВНЫЕ МЕТРИКИ:")
    print(f"AP (Average Precision): {metrics['AP']:.4f}")
    print(f"AP@0.5: {metrics['AP_50']:.4f}")
    print(f"AP@0.75: {metrics['AP_75']:.4f}")
    print(f"AR (Average Recall): {metrics['AR']:.4f}")
    print(f"mOKS (mean OKS): {metrics['mOKS']:.4f}")
    print(f"OKS std: {metrics['OKS_std']:.4f}")
    print(f"Успешных сопоставлений: {metrics['total_matches']}/{metrics['total_gt']}")
    
    # Детальная статистика по OKS
    if matches:
        oks_scores = [m["oks"] for m in matches]
        print(f"\nДЕТАЛЬНАЯ СТАТИСТИКА OKS:")
        print(f"Min OKS: {min(oks_scores):.4f}")
        print(f"Max OKS: {max(oks_scores):.4f}")
        print(f"Median OKS: {np.median(oks_scores):.4f}")
        
        # Распределение по качеству
        excellent = sum(1 for oks in oks_scores if oks >= 0.8)
        good = sum(1 for oks in oks_scores if 0.5 <= oks < 0.8)
        poor = sum(1 for oks in oks_scores if oks < 0.5)
        
        print(f"\nКАЧЕСТВО ДЕТЕКЦИИ:")
        print(f"Отличные (OKS ≥ 0.8): {excellent} ({excellent/len(oks_scores)*100:.1f}%)")
        print(f"Хорошие (0.5 ≤ OKS < 0.8): {good} ({good/len(oks_scores)*100:.1f}%)")
        print(f"Плохие (OKS < 0.5): {poor} ({poor/len(oks_scores)*100:.1f}%)")
else:
    print("Не удалось вычислить метрики - нет сопоставленных предсказаний")

print(f"\nВизуализации сохранены в: {os.path.abspath(OUTPUT_DIR)}")