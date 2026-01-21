from rex_omni import RexOmniVisualize, RexOmniWrapper
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def remove_duplicate_detections(predictions, iou_threshold=0.5):
    if not isinstance(predictions, dict):
        return predictions
    
    filtered_predictions = {}
    
    for category, detections in predictions.items():
        filtered_detections = []
        boxes = [det['coords'] for det in detections if det['type'] == 'box']
        
        if len(boxes) <= 1:
            filtered_predictions[category] = detections
            continue
        
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if keep[j] and calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    keep[j] = False
        
        for i, detection in enumerate(detections):
            if keep[i]:
                filtered_detections.append(detection)
        
        filtered_predictions[category] = filtered_detections
    
    return filtered_predictions

def is_cow_completely_in_frame(bbox, image_size, margin_threshold=0.02):
    img_width, img_height = image_size
    margin_x = img_width * margin_threshold
    margin_y = img_height * margin_threshold
    
    x1, y1, x2, y2 = bbox
    
    if (x1 <= margin_x or y1 <= margin_y or 
        x2 >= img_width - margin_x or y2 >= img_height - margin_y):
        return False
    
    return True

def get_weight_color(weight, min_weight, max_weight):
    if max_weight == min_weight:
        return (0, 255, 0)
    
    normalized = (weight - min_weight) / (max_weight - min_weight)
    
    red = int(255 * normalized)
    blue = int(255 * (1 - normalized))
    green = 100
    
    return (red, green, blue)

def normalized_to_pixel_coordinates(normalized_boxes, img_width, img_height):
    pixel_boxes = []
    
    for box in normalized_boxes:
        x_center, y_center, w, h = box
        
        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = w * img_width
        height_pixel = h * img_height
        
        x1 = x_center_pixel - (width_pixel / 2)
        y1 = y_center_pixel - (height_pixel / 2)
        x2 = x_center_pixel + (width_pixel / 2)
        y2 = y_center_pixel + (height_pixel / 2)
        
        pixel_boxes.append([x1, y1, x2, y2])
    
    return pixel_boxes

def pixel_to_normalized_coordinates(pixel_boxes, img_width, img_height):
    normalized_boxes = []
    
    for box in pixel_boxes:
        x1, y1, x2, y2 = box
        
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height
        
        normalized_boxes.append([x_center, y_center, w, h])
    
    return normalized_boxes

def calculate_box_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def merge_boxes(box1, box2):
    """Объединяет два bounding box'а в один"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    merged_x1 = min(x1_1, x1_2)
    merged_y1 = min(y1_1, y1_2)
    merged_x2 = max(x2_1, x2_2)
    merged_y2 = max(y2_1, y2_2)
    
    return [merged_x1, merged_y1, merged_x2, merged_y2]

class CowBBoxRefiner:
    def __init__(self, rex_model):
        self.rex_model = rex_model
        
    def refine_bbox_with_visual_prompting(self, image, original_bbox, image_size):
        """Улучшает bounding box с помощью visual prompting"""
        try:
            img_width, img_height = image_size
            
            # Преобразуем оригинальный bbox в нормализованные координаты
            normalized_bbox = pixel_to_normalized_coordinates([original_bbox], img_width, img_height)[0]
            
            # Используем visual prompting для улучшения детекции
            results = self.rex_model.inference(
                images=image,
                task="visual_prompting",
                visual_prompt_boxes=[normalized_bbox],
            )
            
            if results and results[0]["success"]:
                predictions = results[0]["extracted_predictions"]
                refined_boxes = []
                
                for category, detections in predictions.items():
                    for detection in detections:
                        if detection['type'] == 'box':
                            # Преобразуем обратно в пиксельные координаты
                            norm_coords = detection['coords']
                            pixel_coords = normalized_to_pixel_coordinates([norm_coords], img_width, img_height)[0]
                            refined_boxes.append(pixel_coords)
                
                # Выбираем наилучший улучшенный bbox
                if refined_boxes:
                    best_refined_bbox = self._select_best_refined_bbox(original_bbox, refined_boxes)
                    return best_refined_bbox
                
        except Exception as e:
            print(f"Ошибка при улучшении bbox: {e}")
        
        return original_bbox
    
    def _select_best_refined_bbox(self, original_bbox, refined_boxes):
        """Выбирает наилучший улучшенный bbox на основе IoU и площади"""
        best_bbox = original_bbox
        best_score = -1
        
        for refined_bbox in refined_boxes:
            # Вычисляем IoU с оригинальным bbox
            iou = calculate_iou(original_bbox, refined_bbox)
            
            # Вычисляем соотношение площадей
            original_area = calculate_box_area(original_bbox)
            refined_area = calculate_box_area(refined_bbox)
            area_ratio = min(original_area, refined_area) / max(original_area, refined_area)
            
            # Комбинированная оценка: IoU + стабильность размера
            score = iou * 0.7 + area_ratio * 0.3
            
            if score > best_score:
                best_score = score
                best_bbox = refined_bbox
        
        # Если найден хороший кандидат (IoU > 0.3), используем его
        if best_score > 0.3:
            return best_bbox
        else:
            return original_bbox
    
    def refine_detections_with_visual_prompting(self, image, initial_predictions, image_size):
        """Улучшает все детекции с помощью visual prompting"""
        enhanced_predictions = initial_predictions.copy()
        
        if 'cow full body without head' not in enhanced_predictions:
            return enhanced_predictions
            
        img_width, img_height = image_size
        refined_detections = []
        
        # Для каждой обнаруженной коровы улучшаем bbox
        for detection in enhanced_predictions['cow full body without head']:
            if detection['type'] == 'box':
                original_bbox = detection['coords']
                
                # Улучшаем bbox с помощью visual prompting
                refined_bbox = self.refine_bbox_with_visual_prompting(image, original_bbox, image_size)
                
                # Проверяем, действительно ли bbox улучшился
                iou_with_original = calculate_iou(original_bbox, refined_bbox)
                original_area = calculate_box_area(original_bbox)
                refined_area = calculate_box_area(refined_bbox)
                
                # Считаем улучшенным, если IoU > 0.5 и площадь изменилась не слишком сильно
                is_improved = (iou_with_original > 0.5 and 
                             0.5 <= refined_area / original_area <= 2.0)
                
                refined_detection = detection.copy()
                refined_detection['coords'] = refined_bbox
                if is_improved:
                    refined_detection['bbox_refined'] = True
                    print(f"  Улучшен bbox: IoU={iou_with_original:.3f}, площадь: {original_area:.0f} -> {refined_area:.0f}")
                
                refined_detections.append(refined_detection)
        
        # Заменяем оригинальные детекции улучшенными
        enhanced_predictions['cow full body without head'] = refined_detections
        
        return enhanced_predictions

def create_detailed_visualization(image, predictions, output_path, min_weight, max_weight):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)
    
    for i, pred in enumerate(predictions):
        bbox = pred['bbox']
        weight = pred['estimated_weight']
        area_ratio = pred['area_ratio']
        relative_size = pred['relative_size']
        
        color = np.array(get_weight_color(weight, min_weight, max_weight)) / 255.0
        
        # Рисуем bounding box
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        # Добавляем индикатор улучшения
        enhancement_note = ""
        if pred.get('bbox_refined'):
            enhancement_note = "\n[Refined]"
            
        info_text = f"Cow {i+1}\n{weight:.0f} kg\nArea: {area_ratio:.3f}\nSize: {relative_size:.2f}x{enhancement_note}"
        
        ax.annotate(
            info_text, 
            (bbox[0], bbox[1]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    ax.set_title(f'Cow Weight Estimation - {len(predictions)} cows detected', fontsize=16)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Детальная визуализация сохранена: {output_path}")

def process_folder_refined(folder_path, output_folder="output", avg_cow_weight=600, area_threshold=0.05):
    model_path = "IDEA-Research/Rex-Omni"
    
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=512,
        temperature=0.75,
        top_p=0.5,
        top_k=10,
        repetition_penalty=1,
    )
    
    # Инициализируем улучшатель bbox'ов
    bbox_refiner = CowBBoxRefiner(rex_model)
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "detailed"), exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    print(f"Найдено {len(image_paths)} изображений")
    
    image_results = []
    all_areas = []
    
    # Обрабатываем каждое изображение
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_size = image.size
            
            categories = ["cow full body without head"]
            results = rex_model.inference(images=image, task="detection", categories=categories)
            
            if results[0]["success"]:
                predictions = results[0]["extracted_predictions"]
                
                if not predictions:
                    print(f"На изображении {os.path.basename(image_path)} не обнаружено коров")
                    continue
                
                print(f"Обрабатывается {os.path.basename(image_path)}")
                
                # Убираем дубликаты
                filtered_predictions = remove_duplicate_detections(predictions)
                
                # Улучшаем bbox'ы с помощью visual prompting
                refined_predictions = bbox_refiner.refine_detections_with_visual_prompting(
                    image, filtered_predictions, image_size
                )
                
                img_width, img_height = image_size
                total_image_area = img_width * img_height
                
                # Собираем статистику по площадям
                for category, detections in refined_predictions.items():
                    for detection in detections:
                        if detection['type'] == 'box':
                            coords = detection['coords']
                            x1, y1, x2, y2 = coords
                            
                            if is_cow_completely_in_frame([x1, y1, x2, y2], image_size):
                                bbox_area = (x2 - x1) * (y2 - y1)
                                area_ratio = bbox_area / total_image_area
                                
                                if area_ratio > area_threshold:
                                    all_areas.append(area_ratio)
                
                image_results.append({
                    'path': image_path,
                    'image': image,
                    'size': image_size,
                    'predictions': refined_predictions,
                    'filename': os.path.basename(image_path)
                })
            
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_areas:
        print("Не обнаружено ни одной коровы на всех изображениях")
        return []
    
    global_mean_area = np.mean(all_areas)
    print(f"Глобальная средняя площадь: {global_mean_area:.4f}")
    
    all_cows = []
    
    # Собираем информацию о всех коровах
    for img_data in image_results:
        image = img_data['image']
        image_size = img_data['size']
        predictions = img_data['predictions']
        image_path = img_data['path']
        filename = img_data['filename']
        
        img_width, img_height = image_size
        total_image_area = img_width * img_height
        
        cows_in_image = []
        
        for category, detections in predictions.items():
            for detection in detections:
                if detection['type'] == 'box':
                    coords = detection['coords']
                    x1, y1, x2, y2 = coords
                    
                    if is_cow_completely_in_frame([x1, y1, x2, y2], image_size):
                        bbox_area = (x2 - x1) * (y2 - y1)
                        area_ratio = bbox_area / total_image_area
                        
                        if area_ratio > area_threshold:
                            relative_size = area_ratio / global_mean_area
                            estimated_weight = avg_cow_weight * relative_size
                            
                            cow_info = {
                                'category': category,
                                'bbox': [x1, y1, x2, y2],
                                'area_ratio': area_ratio,
                                'estimated_weight': estimated_weight,
                                'relative_size': relative_size,
                                'image_path': image_path,
                                'image_name': filename,
                                'bbox_refined': detection.get('bbox_refined', False)
                            }
                            
                            cows_in_image.append(cow_info)
                            all_cows.append(cow_info)
        
        # Создаем визуализацию для каждого изображения
        if cows_in_image:
            weights = [cow['estimated_weight'] for cow in cows_in_image]
            min_weight = min(weights)
            max_weight = max(weights)
            
            detailed_output = os.path.join(output_folder, "detailed", 
                                         f"detailed_{filename.replace('.jpg', '.png').replace('.jpeg', '.png')}")
            create_detailed_visualization(np.array(image), cows_in_image, detailed_output, min_weight, max_weight)
    
    if not all_cows:
        print("Не обнаружено ни одной коровы на всех изображениях")
        return []
    
    # Убираем дубликаты и сортируем по весу
    unique_cows = []
    seen = set()
    for cow in all_cows:
        key = (cow['image_name'], tuple(cow['bbox']))
        if key not in seen:
            seen.add(key)
            unique_cows.append(cow)
    
    unique_cows.sort(key=lambda x: x['estimated_weight'], reverse=True)
    
    # Выводим результаты
    print("\n" + "="*80)
    print("РЕЙТИНГ КОРОВ ПО ВЕСУ (с улучшением bbox'ов через visual prompting)")
    print("="*80)
    
    refined_count = sum(1 for cow in unique_cows if cow.get('bbox_refined'))
    print(f"Улучшено bbox'ов: {refined_count} из {len(unique_cows)}")
    
    for i, cow in enumerate(unique_cows, 1):
        refinement_indicator = " [REF]" if cow.get('bbox_refined') else ""
            
        print(f"{i:2d}. {cow['image_name']:50} | Вес: {cow['estimated_weight']:6.1f} кг | "
              f"Размер: {cow['relative_size']:.2f}x | Площадь: {cow['area_ratio']:.3f}{refinement_indicator}")
    
    # Сохраняем результаты в файл
    results_file = os.path.join(output_folder, "cow_weight_ranking_refined.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Рейтинг коров по весу (с улучшением bbox'ов через visual prompting)\n")
        f.write("="*70 + "\n")
        f.write(f"Улучшено bbox'ов: {refined_count} из {len(unique_cows)}\n\n")
        for i, cow in enumerate(unique_cows, 1):
            refinement_indicator = " [REF]" if cow.get('bbox_refined') else ""
                
            f.write(f"{i:2d}. {cow['image_name']:50} | Вес: {cow['estimated_weight']:6.1f} кг | "
                   f"Размер: {cow['relative_size']:.2f}x | Площадь: {cow['area_ratio']:.3f}{refinement_indicator}\n")
    
    # Выводим статистику
    if unique_cows:
        weights = [cow['estimated_weight'] for cow in unique_cows]
        print(f"\nСТАТИСТИКА:")
        print(f"Всего коров: {len(unique_cows)}")
        print(f"Улучшено bbox'ов: {refined_count}")
        print(f"Средний вес: {np.mean(weights):.1f} кг")
        print(f"Максимальный вес: {np.max(weights):.1f} кг")
        print(f"Минимальный вес: {np.min(weights):.1f} кг")
        print(f"Медианный вес: {np.median(weights):.1f} кг")
        print(f"Стандартное отклонение: {np.std(weights):.1f} кг")
    
    print(f"\nПолные результаты сохранены в: {results_file}")
    return unique_cows

def main():
    folder_path = "cows-1/test"
    
    average_cow_weight = 500
    area_threshold = 0.16
    
    all_cows = process_folder_refined(
        folder_path=folder_path,
        output_folder="cow_weight_results_refined",
        avg_cow_weight=average_cow_weight,
        area_threshold=area_threshold
    )

if __name__ == "__main__":
    main()