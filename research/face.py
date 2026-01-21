
import mediapipe as mp
import cv2
import numpy as np
import json
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment

# ----------------------------------------------------------------------
# 1. Конфигурация и константы для лица
# ----------------------------------------------------------------------

DATASET_KEYPOINT_NAMES = [
    "left_eye", "right_eye", "left_ear", "right_ear", 
    "mouth", "nose", "left_eyebrow", "right_eyebrow"
]

# Уточненные сигмы для лица
DATASET_SIGMAS = np.array([
    0.025, 0.025, 0.035, 0.035, 
    0.035, 0.026, 0.025, 0.025
])

# Упрощенный маппинг для лучшей совместимости
MEDIAPIPE_TO_DATASET_MAPPING = {
    # Глаза (используем центральные точки)
    468: "left_eye",    # Центр левого глаза
    473: "right_eye",   # Центр правого глаза
    
    # Уши (используем точки сбоку головы)
    454: "left_ear",    # Левое ухо
    234: "right_ear",   # Правое ухо
    
    # Рот
    13: "mouth",        # Центр рта
    
    # Нос
    1: "nose",          # Кончик носа
    
    # Брови (центральные точки)
    70: "left_eyebrow",   # Центр левой брови
    300: "right_eyebrow"  # Центр правой брови
}

COCO_ANN = "Human-Face-Pose-1/train/_annotations.coco.json"
COCO_IMG_DIR = "Human-Face-Pose-1/train/"
MAX_IMAGES = 400
OUTPUT_DIR = "output_predictions_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 2. MediaPipe Face реализация с улучшенной диагностикой
# ----------------------------------------------------------------------

class MediaPipeFaceEvaluator:
    def __init__(self, min_detection_confidence=0.3):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=False,  # Упрощаем для стабильности
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.3
        )
        self.detection_stats = {
            "total_images": 0,
            "faces_detected": 0,
            "no_faces": 0,
            "landmarks_processed": 0
        }
    
    def mediapipe_inference(self, image_path):
        """Инференс MediaPipe Face Mesh с диагностикой"""
        image = cv2.imread(image_path)
        if image is None:
            return [], None, None
        
        self.detection_stats["total_images"] += 1
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.face_mesh.process(image_rgb)
        predictions = []
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            self.detection_stats["faces_detected"] += len(results.multi_face_landmarks)
            
            for face_landmarks in results.multi_face_landmarks:
                dataset_keypoints = self.convert_mediapipe_to_dataset(face_landmarks.landmark, w, h)
                
                if dataset_keypoints and self.has_sufficient_face_points(dataset_keypoints):
                    confidence = self.calculate_face_confidence(face_landmarks.landmark)
                    
                    predictions.append({
                        "category_id": 1,
                        "keypoints": dataset_keypoints,
                        "score": confidence,
                        "landmarks": face_landmarks
                    })
                    self.detection_stats["landmarks_processed"] += 1
        else:
            self.detection_stats["no_faces"] += 1
        
        return predictions, image, results.multi_face_landmarks
    
    def convert_mediapipe_to_dataset(self, mediapipe_landmarks, img_w, img_h):
        """Конвертирует MediaPipe Face landmarks в формат датасета"""
        dataset_keypoints = []
        
        mediapipe_kpts = []
        for landmark in mediapipe_landmarks:
            x = landmark.x * img_w
            y = landmark.y * img_h
            v = 2.0  # MediaPipe всегда возвращает видимые точки
            mediapipe_kpts.append([x, y, v])
        
        # Для каждой точки датасета ищем соответствие
        points_found = 0
        for dataset_point in DATASET_KEYPOINT_NAMES:
            found = False
            for mp_idx, dataset_name in MEDIAPIPE_TO_DATASET_MAPPING.items():
                if dataset_name == dataset_point and mp_idx < len(mediapipe_kpts):
                    x, y, v = mediapipe_kpts[mp_idx]
                    dataset_keypoints.extend([x, y, v])
                    found = True
                    points_found += 1
                    break
            
            if not found:
                dataset_keypoints.extend([0, 0, 0])  # Точка не найдена
        
        return dataset_keypoints
    
    def has_sufficient_face_points(self, keypoints):
        """Проверяет, достаточно ли точек лица обнаружено"""
        kpts_array = np.array(keypoints).reshape(-1, 3)
        visible_points = kpts_array[kpts_array[:, 2] > 0]
        
        # Требуем хотя бы 5 из 8 ключевых точек для валидного лица
        return len(visible_points) >= 5
    
    def calculate_face_confidence(self, landmarks):
        """Вычисляет confidence для лица"""
        if not landmarks:
            return 0.3
        # Простая эвристика на основе количества точек
        return min(0.5 + (len(landmarks) / 478) * 0.5, 1.0)
    
    def print_detection_stats(self):
        """Выводит статистику детекции"""
        print(f"\n📊 СТАТИСТИКА ДЕТЕКЦИИ MEDIAPIPE:")
        print(f"   Обработано изображений: {self.detection_stats['total_images']}")
        print(f"   Найдено лиц: {self.detection_stats['faces_detected']}")
        print(f"   Изображений без лиц: {self.detection_stats['no_faces']}")
        print(f"   Обработано landmarks: {self.detection_stats['landmarks_processed']}")
        
        if self.detection_stats['total_images'] > 0:
            detection_rate = (self.detection_stats['faces_detected'] / self.detection_stats['total_images']) * 100
            print(f"   Процент детекции: {detection_rate:.1f}%")
    
    def close(self):
        self.face_mesh.close()

# ----------------------------------------------------------------------
# 3. Функции для расчета метрик с улучшенной диагностикой
# ----------------------------------------------------------------------

def compute_oks(dt_kpts, gt_kpts, area):
    """Вычисление OKS между предсказанием и GT для лица"""
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

def validate_and_analyze_gt(coco_annotations, img_id_to_file):
    """Валидация и анализ GT аннотаций"""
    valid_annotations = []
    analysis = {
        "total_annotations": 0,
        "valid_annotations": 0,
        "invalid_keypoints": 0,
        "insufficient_points": 0,
        "bbox_issues": 0
    }
    
    for ann in coco_annotations:
        if ann["category_id"] != 1:
            continue
            
        analysis["total_annotations"] += 1
        
        if "keypoints" not in ann:
            analysis["invalid_keypoints"] += 1
            continue
            
        keypoints = ann["keypoints"]
        if len(keypoints) != len(DATASET_KEYPOINT_NAMES) * 3:
            analysis["invalid_keypoints"] += 1
            continue
            
        # Проверяем видимые точки
        kpts_array = np.array(keypoints).reshape(-1, 3)
        visible_points = kpts_array[kpts_array[:, 2] > 0]
        
        if len(visible_points) < 3:
            analysis["insufficient_points"] += 1
            continue
            
        # Проверяем bbox
        if "bbox" not in ann or ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
            analysis["bbox_issues"] += 1
            continue
            
        valid_annotations.append(ann)
        analysis["valid_annotations"] += 1
    
    print(f"\n🔍 АНАЛИЗ GT АННОТАЦИЙ:")
    print(f"   Всего аннотаций: {analysis['total_annotations']}")
    print(f"   Валидных: {analysis['valid_annotations']}")
    print(f"   Проблемы с ключевыми точками: {analysis['invalid_keypoints']}")
    print(f"   Недостаточно точек: {analysis['insufficient_points']}")
    print(f"   Проблемы с bbox: {analysis['bbox_issues']}")
    
    return valid_annotations

def analyze_prediction_quality(predictions, matched_predictions):
    """Анализ качества предсказаний"""
    print(f"\n🔍 АНАЛИЗ ПРЕДСКАЗАНИЙ:")
    print(f"   Всего предсказаний: {len(predictions)}")
    print(f"   Успешных сопоставлений: {len(matched_predictions)}")
    
    if not predictions:
        return
    
    # Анализ confidence
    confidences = [p["score"] for p in predictions]
    print(f"   Средний confidence: {np.mean(confidences):.3f}")
    print(f"   Min confidence: {np.min(confidences):.3f}")
    print(f"   Max confidence: {np.max(confidences):.3f}")
    
    # Анализ точек в предсказаниях
    points_per_pred = []
    for pred in predictions:
        kpts = np.array(pred["keypoints"]).reshape(-1, 3)
        visible_points = len(kpts[kpts[:, 2] > 0])
        points_per_pred.append(visible_points)
    
    print(f"   Среднее точек на лицо: {np.mean(points_per_pred):.1f}")
    print(f"   Min точек: {np.min(points_per_pred)}")
    print(f"   Max точек: {np.max(points_per_pred)}")

def calculate_keypoint_metrics(predictions, coco_annotations, img_id_to_file):
    """Расчет метрик с улучшенной диагностикой"""
    valid_gt_annotations = validate_and_analyze_gt(coco_annotations, img_id_to_file)
    
    if not valid_gt_annotations:
        print("❌ Нет валидных GT аннотаций для расчета метрик!")
        return {
            "AP": 0, "AP_50": 0, "AP_75": 0, "AR": 0,
            "mOKS": 0, "OKS_std": 0, "total_matches": 0,
            "total_gt": 0, "total_preds": len(predictions),
            "match_ratio": 0
        }, []
    
    gt_by_image = {}
    for ann in valid_gt_annotations:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            gt_by_image[img_id] = []
        gt_by_image[img_id].append(ann)
    
    # Правильные пороги для OKS
    oks_thresholds = np.linspace(0.05, 0.95, 19)
    
    all_tp = {thresh: 0 for thresh in oks_thresholds}
    all_fp = {thresh: 0 for thresh in oks_thresholds}
    all_fn = {thresh: 0 for thresh in oks_thresholds}
    
    matched_predictions = []
    detailed_oks_scores = []
    
    # Анализ сопоставлений
    matching_analysis = {
        "images_processed": 0,
        "images_with_gt": 0,
        "images_with_preds": 0,
        "total_matching_attempts": 0,
        "successful_matches": 0
    }
    
    for img_id, file_name in img_id_to_file.items():
        matching_analysis["images_processed"] += 1
        
        if img_id not in gt_by_image:
            continue
            
        img_gts = gt_by_image[img_id]
        img_preds = [p for p in predictions if p["image_id"] == img_id]
        
        matching_analysis["images_with_gt"] += 1
        
        if not img_gts:
            continue
            
        if not img_preds:
            for threshold in oks_thresholds:
                all_fn[threshold] += len(img_gts)
            continue
        
        matching_analysis["images_with_preds"] += 1
        
        # Матрица OKS
        oks_matrix = np.zeros((len(img_preds), len(img_gts)))
        
        for i, pred in enumerate(img_preds):
            for j, gt in enumerate(img_gts):
                bbox = gt["bbox"]
                area = bbox[2] * bbox[3]
                oks = compute_oks(pred["keypoints"], gt["keypoints"], area)
                oks_matrix[i, j] = oks
                matching_analysis["total_matching_attempts"] += 1
        
        # Hungarian matching
        if oks_matrix.size > 0:
            cost_matrix = 1 - oks_matrix
            pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
            
            for threshold in oks_thresholds:
                tp = 0
                matched_gt = set()
                matched_pred = set()
                
                for i, j in zip(pred_indices, gt_indices):
                    oks_score = oks_matrix[i, j]
                    if oks_score >= threshold:
                        tp += 1
                        matched_gt.add(j)
                        matched_pred.add(i)
                        
                        if abs(threshold - 0.50) < 0.01:
                            matched_predictions.append({
                                "pred": img_preds[i],
                                "gt": img_gts[j],
                                "oks": oks_score,
                                "image_id": img_id
                            })
                            detailed_oks_scores.append(oks_score)
                            matching_analysis["successful_matches"] += 1
                
                fp = len(img_preds) - len(matched_pred)
                fn = len(img_gts) - len(matched_gt)
                
                all_tp[threshold] += tp
                all_fp[threshold] += fp
                all_fn[threshold] += fn
    
    print(f"\n🔍 АНАЛИЗ СОПОСТАВЛЕНИЙ:")
    print(f"   Обработано изображений: {matching_analysis['images_processed']}")
    print(f"   Изображений с GT: {matching_analysis['images_with_gt']}")
    print(f"   Изображений с предсказаниями: {matching_analysis['images_with_preds']}")
    print(f"   Попыток сопоставления: {matching_analysis['total_matching_attempts']}")
    print(f"   Успешных сопоставлений: {matching_analysis['successful_matches']}")
    
    # Расчет метрик
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
    
    # Находим индексы для AP50 и AP75
    threshold_50_idx = np.argmin(np.abs(oks_thresholds - 0.50))
    threshold_75_idx = np.argmin(np.abs(oks_thresholds - 0.75))
    
    if detailed_oks_scores:
        metrics = {
            "AP": np.mean(ap_scores),
            "AP_50": ap_scores[threshold_50_idx],
            "AP_75": ap_scores[threshold_75_idx],
            "AR": np.mean(ar_scores),
            "mOKS": np.mean(detailed_oks_scores),
            "OKS_std": np.std(detailed_oks_scores),
            "total_matches": len(matched_predictions),
            "total_gt": len(valid_gt_annotations),
            "total_preds": len(predictions),
            "match_ratio": len(matched_predictions) / len(predictions) if len(predictions) > 0 else 0
        }
    else:
        metrics = {
            "AP": 0, "AP_50": 0, "AP_75": 0, "AR": 0,
            "mOKS": 0, "OKS_std": 0, "total_matches": 0,
            "total_gt": len(valid_gt_annotations),
            "total_preds": len(predictions),
            "match_ratio": 0
        }
    
    return metrics, matched_predictions

# ----------------------------------------------------------------------
# 4. Основной пайплайн оценки
# ----------------------------------------------------------------------

def main():
    print("🚀 ЗАПУСК ОЦЕНКИ MEDIAPIPE FACE С ДИАГНОСТИКОЙ")
    print("="*50)
    
    # Загрузка COCO аннотаций
    print("📁 Загрузка COCO аннотаций...")
    try:
        with open(COCO_ANN, "r") as f:
            coco = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл аннотаций не найден: {COCO_ANN}")
        return
    
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    
    face_annotations = [ann for ann in coco["annotations"] if ann["category_id"] == 1]
    face_img_ids = {ann["image_id"] for ann in face_annotations}
    
    if MAX_IMAGES:
        face_img_ids = list(face_img_ids)[:MAX_IMAGES]
    else:
        face_img_ids = list(face_img_ids)
    
    print(f"📊 Всего изображений с лицами: {len(face_img_ids)}")
    print(f"📊 Всего аннотаций лиц: {len(face_annotations)}")
    
    # Инициализация и оценка MediaPipe Face
    print(f"\n🧪 Инициализация MediaPipe Face Mesh...")
    mediapipe_evaluator = MediaPipeFaceEvaluator(min_detection_confidence=0.3)
    
    mediapipe_predictions = []
    
    for img_id in tqdm(face_img_ids, desc="MediaPipe Face Inference"):
        file_name = img_id_to_file[img_id]
        img_path = os.path.join(COCO_IMG_DIR, file_name)
        
        if not os.path.isfile(img_path):
            continue
        
        preds, image, landmarks = mediapipe_evaluator.mediapipe_inference(img_path)
        
        for pred in preds:
            pred["image_id"] = img_id
            mediapipe_predictions.append(pred)
    
    mediapipe_evaluator.close()
    
    # Статистика детекции
    mediapipe_evaluator.print_detection_stats()
    
    # Расчет метрик
    print(f"\n📈 Расчет метрик ключевых точек лица...")
    all_gt_annotations = [ann for ann in coco["annotations"] if ann["category_id"] == 1]
    mediapipe_metrics, mediapipe_matches = calculate_keypoint_metrics(
        mediapipe_predictions, all_gt_annotations, img_id_to_file
    )
    
    # Анализ качества предсказаний
    analyze_prediction_quality(mediapipe_predictions, mediapipe_matches)
    
    # Вывод результатов
    print(f"\n✅ РЕЗУЛЬТАТЫ ОЦЕНКИ ЛИЦ:")
    print(f"   AP: {mediapipe_metrics['AP']:.4f}")
    print(f"   AP@0.5: {mediapipe_metrics['AP_50']:.4f}")
    print(f"   AP@0.75: {mediapipe_metrics['AP_75']:.4f}")
    print(f"   AR: {mediapipe_metrics['AR']:.4f}")
    print(f"   Средний OKS: {mediapipe_metrics['mOKS']:.4f}")
    
    if mediapipe_metrics['total_gt'] > 0:
        detection_efficiency = mediapipe_metrics['total_matches'] / mediapipe_metrics['total_gt'] * 100
        print(f"\n🎯 ОБЩАЯ ЭФФЕКТИВНОСТЬ:")
        print(f"   Эффективность обнаружения: {detection_efficiency:.1f}%")
        print(f"   Пропущено лиц: {mediapipe_metrics['total_gt'] - mediapipe_metrics['total_matches']}")
        print(f"   Ложные срабатывания: {mediapipe_metrics['total_preds'] - mediapipe_metrics['total_matches']}")

if __name__ == "__main__":
    main()