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
        return (0, 255, 0)  # Зеленый если все веса одинаковые
    
    normalized = (weight - min_weight) / (max_weight - min_weight)
    
    # Синий для легких, красный для тяжелых
    red = int(255 * normalized)
    blue = int(255 * (1 - normalized))
    green = 100
    
    return (red, green, blue)


def create_detailed_visualization(image, predictions, output_path, min_weight, max_weight):

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)
    
    for i, pred in enumerate(predictions):
        bbox = pred['bbox']
        weight = pred['estimated_weight']
        area_ratio = pred['area_ratio']
        relative_size = pred['relative_size']
        
        color = np.array(get_weight_color(weight, min_weight, max_weight)) / 255.0
        
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        info_text = f"Cow {i+1}\n{weight:.0f} kg\nArea: {area_ratio:.3f}\nSize: {relative_size:.2f}x"
        
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

def process_folder_corrected(folder_path, output_folder="output", avg_cow_weight=600, area_threshold=0.05):

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
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_size = image.size
            
            categories = ["cow"]
            results = rex_model.inference(images=image, task="detection", categories=categories)
            
            if results[0]["success"]:
                predictions = results[0]["extracted_predictions"]
                
                if not predictions:
                    print(f"На изображении {os.path.basename(image_path)} не обнаружено коров")
                    continue
                
                print(f"Обрабатывается {os.path.basename(image_path)}")
                
                filtered_predictions = remove_duplicate_detections(predictions)
                
                img_width, img_height = image_size
                total_image_area = img_width * img_height
                
                for category, detections in filtered_predictions.items():
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
                    'predictions': filtered_predictions,
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
                                'image_name': filename
                            }
                            
                            cows_in_image.append(cow_info)
                            all_cows.append(cow_info)
        
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
    
    unique_cows = []
    seen = set()
    for cow in all_cows:
        key = (cow['image_name'], tuple(cow['bbox']))
        if key not in seen:
            seen.add(key)
            unique_cows.append(cow)
    
    unique_cows.sort(key=lambda x: x['estimated_weight'], reverse=True)
    
    print("\n" + "="*80)
    print("РЕЙТИНГ КОРОВ ПО ВЕСУ (глобальная нормализация)")
    print("="*80)
    
    for i, cow in enumerate(unique_cows, 1):
        print(f"{i:2d}. {cow['image_name']:50} | Вес: {cow['estimated_weight']:6.1f} кг | "
              f"Размер: {cow['relative_size']:.2f}x | Площадь: {cow['area_ratio']:.3f}")
    
    results_file = os.path.join(output_folder, "cow_weight_ranking.txt")
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Рейтинг коров по весу (глобальная нормализация)\n")
        f.write("="*70 + "\n")
        for i, cow in enumerate(unique_cows, 1):
            f.write(f"{i:2d}. {cow['image_name']:50} | Вес: {cow['estimated_weight']:6.1f} кг | "
                   f"Размер: {cow['relative_size']:.2f}x | Площадь: {cow['area_ratio']:.3f}\n")
    
    if unique_cows:
        weights = [cow['estimated_weight'] for cow in unique_cows]
        print(f"\nСТАТИСТИКА:")
        print(f"Всего коров: {len(unique_cows)}")
        print(f"Средний вес: {np.mean(weights):.1f} кг")
        print(f"Максимальный вес: {np.max(weights):.1f} кг")
        print(f"Минимальный вес: {np.min(weights):.1f} кг")
        print(f"Медианный вес: {np.median(weights):.1f} кг")
        print(f"Стандартное отклонение: {np.std(weights):.1f} кг")
        
    
    print(f"\nПолные результаты сохранены в: {results_file}")
    return unique_cows


def main():
    folder_path = "cows-1/test"
    

    average_cow_weight = 500  # средний вес кг
    area_threshold = 0.09 # трешхолд, чтоб убрать галюцинации или обрезанные bounding box
    
    all_cows = process_folder_corrected(
        folder_path=folder_path,
        output_folder="cow_weight_results",
        avg_cow_weight=average_cow_weight,
        area_threshold=area_threshold
    )

if __name__ == "__main__":
    main()