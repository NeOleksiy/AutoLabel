import torch
from transformers import Sam2Model, Sam2Processor
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

class SAM2InferenceWithMetrics:
    """
    Класс для инференса и расчета метрик с использованием модели SAM2 из transformers.
    """
    
    def __init__(self, model_name="facebook/sam2.1-hiera-large"):
        """
        Инициализация модели и процессора SAM2.
        
        Args:
            model_name (str): Название предобученной модели SAM2 на Hugging Face Hub.
        """
        print("Загрузка модели SAM2...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Sam2Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam2Processor.from_pretrained(model_name)
        print(f"Модель загружена на устройство: {self.device}")

    def get_bounding_boxes_from_masks(self, masks, original_size, confidence_threshold=0.5):
        """
        Извлекает bounding boxes из масок, сгенерированных SAM2.
        
        Args:
            masks: Выходные маски от модели
            original_size: Оригинальный размер изображения
            confidence_threshold: Порог для фильтрации масок по уверенности
            
        Returns:
            bounding_boxes: Список bounding boxes в формате [x1, y1, x2, y2, score, class_id]
        """
        # Для Sam2Model из transformers нужно обработать выходы
        if isinstance(masks, torch.Tensor):
            # Берем первую маску с наибольшей уверенностью
            if masks.shape[1] > 0:
                mask = masks[0, 0].cpu().numpy() > 0  # Бинаризуем маску
                score = 0.9  # SAM2 не возвращает scores, используем дефолтное значение
            else:
                return []
        else:
            # Если masks уже обработаны
            mask = masks[0] if len(masks) > 0 else None
            score = 0.9
            
        if mask is None:
            return []
            
        # Находим координаты ограничивающего прямоугольника для маски
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return []
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Формат: [x_min, y_min, x_max, y_max, score, class_id]
        # SAM2 не возвращает class_id, можно использовать 0
        bbox = [x_min, y_min, x_max, y_max, score, 0]
        return [bbox]

    def process_image_with_prompts(self, image_path, input_boxes=None, input_points=None, 
                                 input_labels=None, confidence_threshold=0.5):
        """
        Обрабатывает изображение с промптами и возвращает предсказанные bounding boxes.
        
        Args:
            image_path: путь к изображению
            input_boxes: список bounding boxes в формате [[x1, y1, x2, y2]]
            input_points: список точек в формате [[[x, y]]]
            input_labels: метки точек (1 - положительная, 0 - отрицательная)
            confidence_threshold: порог уверенности
            
        Returns:
            bounding_boxes: список предсказанных bounding boxes
            image_np: изображение в формате numpy
        """
        # Загрузка изображения
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Подготовка входных данных
        inputs = {}
        if input_boxes is not None:
            inputs['input_boxes'] = input_boxes
        if input_points is not None and input_labels is not None:
            inputs['input_points'] = input_points
            inputs['input_labels'] = input_labels
            
        inputs = self.processor(images=image, return_tensors="pt", **inputs).to(self.device)
        
        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Пост-обработка масок
        original_sizes = inputs["original_sizes"].cpu()
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)
        
        # Извлекаем bounding boxes из масок
        bounding_boxes = []
        for mask_batch in masks:
            for i in range(mask_batch.shape[1]):
                mask = mask_batch[0, i].numpy() > 0
                if np.any(mask):
                    y_indices, x_indices = np.where(mask)
                    x_min, x_max = np.min(x_indices), np.max(x_indices)
                    y_min, y_max = np.min(y_indices), np.max(y_indices)
                    # Используем дефолтный score, так как SAM2 не возвращает confidence для каждой маски
                    bbox = [x_min, y_min, x_max, y_max, 0.9, 0]
                    bounding_boxes.append(bbox)
        
        return bounding_boxes, image_np

    def process_image_automatic(self, image_path, confidence_threshold=0.5):
        """
        Обрабатывает изображение в автоматическом режиме (без промптов).
        В этом режиме SAM2 не работает напрямую, поэтому эмулируем базовую детекцию.
        """
        # Для автоматической детекции можно использовать разные стратегии:
        # 1. Сгенерировать равномерную сетку промптов
        # 2. Использовать простой детектор для получения первоначальных боксов
        # Здесь используем упрощенный подход с фиксированными промптами
        
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        # Создаем сетку точек для промптов
        points = []
        step = 100
        for y in range(step, h, step):
            for x in range(step, w, step):
                points.append([x, y])
        
        if points:
            input_points = [[points]]  # Формат: [[[[x1, y1], [x2, y2], ...]]]
            input_labels = [[[1] * len(points)]]  # Все положительные точки
            
            return self.process_image_with_prompts(
                image_path, input_points=input_points, input_labels=input_labels,
                confidence_threshold=confidence_threshold
            )
        else:
            return [], image_np

def calculate_iou(box1, box2):
    """
    Вычисляет Intersection over Union (IoU) двух bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    
    return intersection / union if union > 0 else 0.0

def read_yolo_labels(label_path, img_width, img_height):
    """
    Читает YOLO формат разметки и преобразует в абсолютные координаты.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        data = line.strip().split()
        if len(data) < 5:
            continue
            
        class_id = int(data[0])
        x_center = float(data[1]) * img_width
        y_center = float(data[2]) * img_height
        width = float(data[3]) * img_width
        height = float(data[4]) * img_height
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes.append([x1, y1, x2, y2, class_id])
    
    return boxes

def calculate_metrics_per_image(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Вычисляет Precision, Recall и средний IoU для одного изображения.
    """
    if not gt_boxes and not pred_boxes:
        return {'precision': 1.0, 'recall': 1.0, 'mean_iou': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    elif not gt_boxes:
        return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0}
    elif not pred_boxes:
        return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes)}
    
    tp = 0
    fp = 0
    fn = len(gt_boxes)
    
    ious = []
    gt_matched = [False] * len(gt_boxes)
    
    # Сортируем предсказания по уверенности (score)
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    
    for pred in pred_boxes_sorted:
        pred_box = pred[:4]
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue
            gt_box = gt[:4]
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            fn -= 1
            gt_matched[best_gt_idx] = True
            ious.append(best_iou)
        else:
            fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    mean_iou = sum(ious) / len(ious) if ious else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'mean_iou': mean_iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'ious': ious
    }

def visualize_results(image, pred_boxes, gt_boxes, output_path):
    """
    Визуализирует предсказания и ground truth на изображении.
    """
    result_image = image.copy()
    
    # Рисуем ground truth (зеленый)
    for gt in gt_boxes:
        x1, y1, x2, y2, class_id = map(int, gt[:5])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f'GT_{class_id}', (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Рисуем предсказания (красный)
    for pred in pred_boxes:
        x1, y1, x2, y2, score, class_id = map(int, pred[:4] + [pred[4] * 100, pred[5]])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(result_image, f'P_{class_id}_{score/100:.2f}', (x1, y1-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Добавляем метрики
    if pred_boxes or gt_boxes:
        metrics = calculate_metrics_per_image(pred_boxes, gt_boxes)
        cv2.putText(result_image, f'Precision: {metrics["precision"]:.2f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, f'Recall: {metrics["recall"]:.2f}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, f'IoU: {metrics["mean_iou"]:.2f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, result_image)

def evaluate_sam2_on_dataset(images_dir, labels_dir, output_dir, use_prompts=True):
    """
    Запускает оценку SAM2 на датасете и вычисляет метрики.
    """
    # Инициализация SAM2
    sam2_evaluator = SAM2InferenceWithMetrics()
    
    # Создание директории для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Поиск изображений
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(images_dir).glob(ext))
        image_paths.extend(Path(images_dir).glob(ext.upper()))
    
    print(f"Найдено {len(image_paths)} изображений")
    
    total_metrics = {
        'precision': [],
        'recall': [],
        'mean_iou': [],
        'tp': 0,
        'fp': 0,
        'fn': 0
    }
    
    for i, image_path in enumerate(image_paths):
        print(f"Обработка {i+1}/{len(image_paths)}: {image_path.name}")
        
        try:
            # Получаем предсказания от SAM2
            if use_prompts:
                # Можно добавить логику для получения промптов из внешнего детектора
                pred_boxes, image_np = sam2_evaluator.process_image_automatic(str(image_path))
            else:
                pred_boxes, image_np = sam2_evaluator.process_image_automatic(str(image_path))
            
            # Загружаем ground truth
            label_path = Path(labels_dir) / f"{image_path.stem}.txt"
            gt_boxes = read_yolo_labels(label_path, image_np.shape[1], image_np.shape[0])
            
            # Вычисляем метрики
            metrics = calculate_metrics_per_image(pred_boxes, gt_boxes)
            
            # Обновляем общую статистику
            for key in ['precision', 'recall', 'mean_iou']:
                total_metrics[key].append(metrics[key])
            total_metrics['tp'] += metrics['tp']
            total_metrics['fp'] += metrics['fp']
            total_metrics['fn'] += metrics['fn']
            
            # Визуализируем результат
            output_path = Path(output_dir) / f"result_{image_path.stem}.png"
            visualize_results(image_np, pred_boxes, gt_boxes, str(output_path))
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {e}")
            continue
    
    # Расчет итоговых метрик
    if total_metrics['precision']:
        avg_precision = np.mean(total_metrics['precision'])
        avg_recall = np.mean(total_metrics['recall'])
        avg_iou = np.mean(total_metrics['mean_iou'])
        
        micro_precision = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fp']) if (total_metrics['tp'] + total_metrics['fp']) > 0 else 0
        micro_recall = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fn']) if (total_metrics['tp'] + total_metrics['fn']) > 0 else 0
        
        print("\n" + "="*50)
        print("ИТОГОВЫЕ МЕТРИКИ SAM2:")
        print("="*50)
        print(f"Обработано изображений: {len(total_metrics['precision'])}")
        print(f"Средняя Precision: {avg_precision:.4f}")
        print(f"Средняя Recall: {avg_recall:.4f}")
        print(f"Средний IoU: {avg_iou:.4f}")
        print(f"Total TP: {total_metrics['tp']}")
        print(f"Total FP: {total_metrics['fp']}")
        print(f"Total FN: {total_metrics['fn']}")
        print(f"Micro Precision: {micro_precision:.4f}")
        print(f"Micro Recall: {micro_recall:.4f}")
        
        # Сохранение метрик в файл
        with open(Path(output_dir) / 'metrics.txt', 'w') as f:
            f.write("SAM2 Metrics Summary\n")
            f.write("====================\n")
            f.write(f"Processed images: {len(total_metrics['precision'])}\n")
            f.write(f"Average Precision: {avg_precision:.4f}\n")
            f.write(f"Average Recall: {avg_recall:.4f}\n")
            f.write(f"Average IoU: {avg_iou:.4f}\n")
            f.write(f"Total TP: {total_metrics['tp']}\n")
            f.write(f"Total FP: {total_metrics['fp']}\n")
            f.write(f"Total FN: {total_metrics['fn']}\n")
            f.write(f"Micro Precision: {micro_precision:.4f}\n")
            f.write(f"Micro Recall: {micro_recall:.4f}\n")
    
    return total_metrics

# Пример использования
if __name__ == "__main__":
    # Настройки путей
    IMAGES_DIR = "Safety--&-Security-1/test/images"
    LABELS_DIR = "Safety--&-Security-1/test/labels" 
    OUTPUT_DIR = "sam2_transformers_results"
    
    # Запуск оценки
    metrics = evaluate_sam2_on_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR
    )
# # 🎯 Пример использования
# if __name__ == "__main__":
#     # Настройки путей (ЗАМЕНИТЕ НА СВОИ!)
#     MODEL_CONFIG = "sam2.1_hiera_b+.yaml"  # конфиг модели SAM2
#     CHECKPOINT_PATH = "sam2.1_hiera_base_plus.pt"  # веса модели SAM2
#     IMAGES_DIR = "Safety--&-Security-1/test/images"
#     LABELS_DIR = "Safety--&-Security-1/test/labels" 
#     OUTPUT_DIR = "sam2_inference_results"
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Запуск оценки
#     metrics = evaluate_sam2_on_dataset(
#         model_cfg=MODEL_CONFIG,
#         checkpoint=CHECKPOINT_PATH,
#         images_dir=IMAGES_DIR,
#         labels_dir=LABELS_DIR,
#         output_dir=OUTPUT_DIR,
#         device=DEVICE
#     )