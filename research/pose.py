import mediapipe as mp
import cv2
import numpy as np
import json
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------
# 1. Конфигурация и константы
# ----------------------------------------------------------------------

DATASET_KEYPOINT_NAMES = [
    "ear-left", "eye-left", "nose", "left-sh", "right-sh",
    "right-elb", "right-twist", "left-elb", "left-twist",
    "right-kan", "right-knee", "right-ankle", "left-kan",
    "left-knee", "left-ankle", "eye-right", "ear-right"
]

DATASET_SIGMAS = np.array([
    0.035, 0.025, 0.026, 0.079, 0.079,
    0.072, 0.062, 0.072, 0.062, 0.107,
    0.087, 0.089, 0.107, 0.087, 0.089,
    0.025, 0.035
])

MEDIAPIPE_TO_DATASET_MAPPING = {
    0: "nose", 2: "eye-left", 5: "eye-right", 7: "ear-left", 8: "ear-right",
    11: "left-sh", 12: "right-sh", 13: "left-elb", 14: "right-elb",
    15: "left-twist", 16: "right-twist", 23: "left-kan", 24: "right-kan",
    25: "left-knee", 26: "right-knee", 27: "left-ankle", 28: "right-ankle"
}

COCO_ANN = "human-pose-1/train/_annotations.coco.json"
COCO_IMG_DIR = "human-pose-1/train/"
MAX_IMAGES = 200

# ----------------------------------------------------------------------
# 2. MediaPipe Pose реализация
# ----------------------------------------------------------------------

class MediaPipePoseEvaluator:
    def __init__(self, min_detection_confidence=0.3, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence
        )
    
    def mediapipe_inference(self, image_path):
        """Инференс MediaPipe Pose на одном изображении"""
        image = cv2.imread(image_path)
        if image is None:
            return [], None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.pose.process(image_rgb)
        predictions = []
        
        if results.pose_landmarks:
            h, w = image.shape[:2]
            dataset_keypoints = self.convert_mediapipe_to_dataset(results.pose_landmarks.landmark, w, h)
            
            if dataset_keypoints:
                visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
                confidence = np.mean(visibilities) if visibilities else 0.5
                
                predictions.append({
                    "category_id": 1,
                    "keypoints": dataset_keypoints,
                    "score": confidence
                })
        
        return predictions, image
    
    def convert_mediapipe_to_dataset(self, mediapipe_landmarks, img_w, img_h):
        """Конвертирует MediaPipe landmarks в формат датасета"""
        dataset_keypoints = []
        
        mediapipe_kpts = []
        for landmark in mediapipe_landmarks:
            x = landmark.x * img_w
            y = landmark.y * img_h
            v = 2.0 if landmark.visibility > 0.5 else 0.0
            mediapipe_kpts.append([x, y, v])
        
        for dataset_point in DATASET_KEYPOINT_NAMES:
            found = False
            for mp_idx, dataset_name in MEDIAPIPE_TO_DATASET_MAPPING.items():
                if dataset_name == dataset_point and mp_idx < len(mediapipe_kpts):
                    x, y, v = mediapipe_kpts[mp_idx]
                    dataset_keypoints.extend([x, y, v])
                    found = True
                    break
            
            if not found:
                dataset_keypoints.extend([0, 0, 0])
        
        return dataset_keypoints
    
    def close(self):
        self.pose.close()

# ----------------------------------------------------------------------
# 3. Функции для расчета метрик ключевых точек
# ----------------------------------------------------------------------

def compute_oks(dt_kpts, gt_kpts, area):
    """Вычисление OKS между предсказанием и GT"""
    if area <= 0:
        return 0.0
        
    vars = (DATASET_SIGMAS * 2) ** 2
    k = len(DATASET_SIGMAS)
    
    dt = np.array(dt_kpts).reshape(k, 3)
    gt = np.array(gt_kpts).reshape(k, 3)
    
    dx = dt[:, 0] - gt[:, 0]
    dy = dt[:, 1] - gt[:, 1]
    
    vis_flag = gt[:, 2] > 0
    if np.sum(vis_flag) == 0:
        return 0.0
    
    e = (dx ** 2 + dy ** 2) / vars / (area + np.spacing(1)) / 2
    e = e[vis_flag]
    
    oks = np.sum(np.exp(-e)) / len(e)
    return oks

def validate_gt_annotations(coco_annotations):
    """Валидация GT аннотаций - проверка наличия ключевых точек"""
    valid_annotations = []
    
    for ann in coco_annotations:
        if ann["category_id"] != 1:
            continue
            
        # Проверяем наличие ключевых точек в аннотации
        if "keypoints" not in ann:
            print(f"⚠️ В аннотации {ann.get('id', 'unknown')} отсутствуют ключевые точки")
            continue
            
        keypoints = ann["keypoints"]
        if len(keypoints) != len(DATASET_KEYPOINT_NAMES) * 3:
            print(f"⚠️ Неправильное количество ключевых точек в аннотации {ann.get('id', 'unknown')}: "
                  f"ожидалось {len(DATASET_KEYPOINT_NAMES) * 3}, получено {len(keypoints)}")
            continue
            
        # Проверяем, что есть хотя бы одна видимая ключевая точка
        kpts_array = np.array(keypoints).reshape(-1, 3)
        visible_points = kpts_array[kpts_array[:, 2] > 0]
        
        if len(visible_points) == 0:
            print(f"⚠️ В аннотации {ann.get('id', 'unknown')} нет видимых ключевых точек")
            continue
            
        valid_annotations.append(ann)
    
    print(f"✅ Валидных GT аннотаций: {len(valid_annotations)}/{len(coco_annotations)}")
    return valid_annotations

def calculate_keypoint_metrics(predictions, coco_annotations, img_id_to_file):
    """
    Расчет метрик для ключевых точек с улучшенным сопоставлением
    """
    # Валидация GT аннотаций
    valid_gt_annotations = validate_gt_annotations(coco_annotations)
    
    # Группируем GT по image_id
    gt_by_image = {}
    for ann in valid_gt_annotations:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)
    
    # Пороги для OKS
    oks_thresholds = np.linspace(0.5, 0.95, 10)
    
    all_tp = {thresh: 0 for thresh in oks_thresholds}
    all_fp = {thresh: 0 for thresh in oks_thresholds}
    all_fn = {thresh: 0 for thresh in oks_thresholds}
    
    matched_predictions = []
    detailed_oks_scores = []
    
    # Обрабатываем каждое изображение
    for img_id, file_name in img_id_to_file.items():
        if img_id not in gt_by_image:
            continue
            
        img_gts = gt_by_image[img_id]
        img_preds = [p for p in predictions if p["image_id"] == img_id]
        
        if not img_gts:
            continue
            
        if not img_preds:
            # Если есть GT но нет предсказаний - это FN
            for threshold in oks_thresholds:
                all_fn[threshold] += len(img_gts)
            continue
        
        # Матрица OKS для всех пар pred-gt
        oks_matrix = np.zeros((len(img_preds), len(img_gts)))
        
        for i, pred in enumerate(img_preds):
            for j, gt in enumerate(img_gts):
                bbox = gt["bbox"]
                area = bbox[2] * bbox[3]
                oks = compute_oks(pred["keypoints"], gt["keypoints"], area)
                oks_matrix[i, j] = oks
        
        # Hungarian matching на основе OKS
        cost_matrix = 1 - oks_matrix
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Для каждого порога считаем статистику
        for threshold in oks_thresholds:
            tp = 0
            matched_gt = set()
            matched_pred = set()
            
            # True positives
            for i, j in zip(pred_indices, gt_indices):
                oks_score = oks_matrix[i, j]
                if oks_score >= threshold:
                    tp += 1
                    matched_gt.add(j)
                    matched_pred.add(i)
                    
                    if threshold == 0.5:  # Сохраняем matches для порога 0.5
                        matched_predictions.append({
                            "pred": img_preds[i],
                            "gt": img_gts[j],
                            "oks": oks_score,
                            "image_id": img_id
                        })
                        detailed_oks_scores.append(oks_score)
            
            fp = len(img_preds) - len(matched_pred)
            fn = len(img_gts) - len(matched_gt)
            
            all_tp[threshold] += tp
            all_fp[threshold] += fp
            all_fn[threshold] += fn
    
    # Вычисляем финальные метрики
    ap_scores = []
    ar_scores = []
    
    for threshold in oks_thresholds:
        tp = all_tp[threshold]
        fp = all_fp[threshold]
        fn = all_fn[threshold]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ap_scores.append(precision)
        ar_scores.append(recall)
    
    if detailed_oks_scores:
        metrics = {
            "AP": np.mean(ap_scores),
            "AP_50": ap_scores[0],
            "AP_75": ap_scores[5],
            "AR": np.mean(ar_scores),
            "mOKS": np.mean(detailed_oks_scores),
            "OKS_std": np.std(detailed_oks_scores),
            "total_matches": len(matched_predictions),
            "total_gt": sum(len(gts) for gts in gt_by_image.values()),
            "total_preds": len(predictions),
            "match_ratio": len(matched_predictions) / len(predictions) if len(predictions) > 0 else 0
        }
    else:
        metrics = {
            "AP": 0, "AP_50": 0, "AP_75": 0, "AR": 0,
            "mOKS": 0, "OKS_std": 0, "total_matches": 0,
            "total_gt": sum(len(gts) for gts in gt_by_image.values()),
            "total_preds": len(predictions),
            "match_ratio": 0
        }
    
    return metrics, matched_predictions

def analyze_keypoint_performance(matched_predictions, metrics):
    """Анализ производительности по ключевым точкам"""
    print(f"\n🔍 АНАЛИЗ КЛЮЧЕВЫХ ТОЧЕК:")
    print(f"   Всего предсказаний: {metrics['total_preds']}")
    print(f"   Всего GT аннотаций: {metrics['total_gt']}")
    print(f"   Успешных сопоставлений: {metrics['total_matches']}")
    print(f"   Эффективность сопоставления: {metrics['match_ratio']:.1%}")
    
    if not matched_predictions:
        return
    
    # Анализ качества OKS
    oks_scores = [m["oks"] for m in matched_predictions]
    good_matches = len([oks for oks in oks_scores if oks >= 0.5])
    excellent_matches = len([oks for oks in oks_scores if oks >= 0.75])
    
    print(f"\n📊 КАЧЕСТВО СОПОСТАВЛЕНИЯ:")
    print(f"   Средний OKS: {metrics['mOKS']:.3f} ± {metrics['OKS_std']:.3f}")
    print(f"   Хорошие сопоставления (OKS ≥ 0.5): {good_matches}/{len(oks_scores)} ({good_matches/len(oks_scores):.1%})")
    print(f"   Отличные сопоставления (OKS ≥ 0.75): {excellent_matches}/{len(oks_scores)} ({excellent_matches/len(oks_scores):.1%})")
    
    # Анализ по отдельным ключевым точкам
    if matched_predictions:
        sample_match = matched_predictions[0]
        pred_kpts = np.array(sample_match["pred"]["keypoints"]).reshape(-1, 3)
        gt_kpts = np.array(sample_match["gt"]["keypoints"]).reshape(-1, 3)
        
        print(f"\n📍 ПРИМЕР КЛЮЧЕВЫХ ТОЧЕК (первое сопоставление):")
        print(f"   OKS: {sample_match['oks']:.3f}")
        print(f"   Видимых точек в GT: {np.sum(gt_kpts[:, 2] > 0)}")
        print(f"   Обнаружено точек: {np.sum(pred_kpts[:, 2] > 0)}")

# ----------------------------------------------------------------------
# 4. Основной пайплайн оценки
# ----------------------------------------------------------------------

def main():
    print("🚀 ЗАПУСК ОЦЕНКИ MEDIAPIPE POSE ПО КЛЮЧЕВЫМ ТОЧКАМ")
    print("="*50)
    
    # Загрузка COCO аннотаций
    print("📁 Загрузка COCO аннотаций...")
    try:
        with open(COCO_ANN, "r") as f:
            coco = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл аннотаций не найден: {COCO_ANN}")
        return
    except json.JSONDecodeError:
        print(f"❌ Ошибка чтения JSON файла: {COCO_ANN}")
        return
    
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    
    # Фильтруем изображения с аннотациями людей
    person_annotations = [ann for ann in coco["annotations"] if ann["category_id"] == 1]
    person_img_ids = {ann["image_id"] for ann in person_annotations}
    
    if MAX_IMAGES:
        person_img_ids = list(person_img_ids)[:MAX_IMAGES]
    else:
        person_img_ids = list(person_img_ids)
    
    print(f"📊 Будет обработано {len(person_img_ids)} изображений с людьми")
    print(f"📊 Всего аннотаций людей: {len(person_annotations)}")
    
    # Инициализация и оценка MediaPipe
    print(f"\n🧪 Инициализация MediaPipe Pose...")
    mediapipe_evaluator = MediaPipePoseEvaluator(
        min_detection_confidence=0.3,
        model_complexity=1
    )
    
    mediapipe_predictions = []
    processed_count = 0
    
    for img_id in tqdm(person_img_ids, desc="MediaPipe Inference"):
        file_name = img_id_to_file[img_id]
        img_path = os.path.join(COCO_IMG_DIR, file_name)
        
        if not os.path.isfile(img_path):
            print(f"⚠️ Изображение не найдено: {img_path}")
            continue
        
        preds, image = mediapipe_evaluator.mediapipe_inference(img_path)
        
        for pred in preds:
            pred["image_id"] = img_id
            mediapipe_predictions.append(pred)
        
        processed_count += 1
    
    mediapipe_evaluator.close()
    
    print(f"✅ Обработано изображений: {processed_count}/{len(person_img_ids)}")
    print(f"✅ Найдено предсказаний: {len(mediapipe_predictions)}")
    
    # Расчет метрик ключевых точек
    print(f"\n📈 Расчет метрик ключевых точек...")
    all_gt_annotations = [ann for ann in coco["annotations"] if ann["category_id"] == 1]
    mediapipe_metrics, mediapipe_matches = calculate_keypoint_metrics(
        mediapipe_predictions, all_gt_annotations, img_id_to_file
    )
    
    # Вывод результатов
    print(f"\n✅ РЕЗУЛЬТАТЫ ОЦЕНКИ:")
    print(f"   Обработано изображений: {processed_count}")
    print(f"   Найдено предсказаний: {mediapipe_metrics['total_preds']}")
    print(f"   Всего GT аннотаций: {mediapipe_metrics['total_gt']}")
    print(f"   Успешных сопоставлений: {mediapipe_metrics['total_matches']}")
    
    print(f"\n📈 МЕТРИКИ КЛЮЧЕВЫХ ТОЧЕК:")
    print(f"   AP: {mediapipe_metrics['AP']:.4f}")
    print(f"   AP@0.5: {mediapipe_metrics['AP_50']:.4f}")
    print(f"   AP@0.75: {mediapipe_metrics['AP_75']:.4f}")
    print(f"   AR: {mediapipe_metrics['AR']:.4f}")
    print(f"   Средний OKS: {mediapipe_metrics['mOKS']:.4f}")
    print(f"   Std OKS: {mediapipe_metrics['OKS_std']:.4f}")
    print(f"   Эффективность сопоставления: {mediapipe_metrics['match_ratio']:.1%}")
    
    # Детальный анализ
    analyze_keypoint_performance(mediapipe_matches, mediapipe_metrics)
    
    # Общая эффективность
    if mediapipe_metrics['total_gt'] > 0:
        detection_efficiency = mediapipe_metrics['total_matches'] / mediapipe_metrics['total_gt'] * 100
        print(f"\n🎯 ОБЩАЯ ЭФФЕКТИВНОСТЬ:")
        print(f"   Эффективность обнаружения: {detection_efficiency:.1f}%")
        print(f"   Пропущено людей: {mediapipe_metrics['total_gt'] - mediapipe_metrics['total_matches']}")
        print(f"   Ложные срабатывания: {mediapipe_metrics['total_preds'] - mediapipe_metrics['total_matches']}")

if __name__ == "__main__":
    main()