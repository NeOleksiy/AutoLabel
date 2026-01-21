import os
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class YOLODatasetEvaluator:
    def __init__(self, dataset_path, class_names, model_path="IDEA-Research/Rex-Omni", text_threshold=0.2):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        self.class_names = class_names
        self.text_threshold = text_threshold  # Порог для текстового соответствия
        
        # Initialize RexOmni model
        self.rex = RexOmniWrapper(
            model_path=model_path,
            backend="transformers",
            max_tokens=4096,
            temperature=0.75,
            top_p=0.7,
            top_k=10,
            repetition_penalty=1,
        )
        
        # Initialize OwlViT model and processor
        try:
            print("Loading OwlViT model for text alignment validation...")
            self.owlvit_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.owlvit_model.eval()
            if torch.cuda.is_available():
                self.owlvit_model.to('cuda')
                print("OwlViT loaded on GPU")
            else:
                print("OwlViT loaded on CPU")
            self.owlvit_available = True
        except Exception as e:
            print(f"Warning: Could not load OwlViT model. Text-based filtering will be disabled. Error: {e}")
            self.owlvit_available = False
        
        self.stats = {
            'total_images': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_predictions': 0,
            'total_filtered_predictions': 0,  # Предсказания после фильтрации
            'total_ground_truths': 0,
            'high_text_score': 0,
            'low_text_score': 0
        }
    
    def check_text_alignment(self, image, bbox, text_query):
        """
        Проверяет соответствие между bounding box и текстовым запросом с помощью OwlViT.
        
        Args:
            image: PIL.Image - исходное изображение
            bbox: List[float] - координаты bounding box в формате [x0, y0, x1, y1]
            text_query: str - текстовое описание класса
            
        Returns:
            float: оценка соответствия (0-1), где 1 означает полное соответствие
        """
        if not self.owlvit_available:
            return 1.0  # Если модель не загружена, пропускаем проверку
        
        try:
            # Обрезаем изображение по bounding box
            cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            
            # Подготавливаем входные данные для OwlViT
            inputs = self.owlvit_processor(
                text=[text_query], 
                images=cropped_img, 
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Получаем предсказания
            with torch.no_grad():
                outputs = self.owlvit_model(**inputs)
            
            # Извлекаем логиты для соответствия изображение-текст
            logits = torch.max(outputs.logits)
            score = torch.sigmoid(logits).item()
            
            return score
        
        except Exception as e:
            print(f"Error in text alignment check: {e}")
            return 0.0
    
    def read_yolo_annotations(self, image_path):
        """Read YOLO format annotations and convert to absolute coordinates"""
        label_path = self.labels_path / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            return []
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return []
        
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # Convert YOLO format to absolute coordinates
                        x_center_abs = x_center * width
                        y_center_abs = y_center * height
                        w_abs = w * width
                        h_abs = h * height
                        
                        x0 = max(0, x_center_abs - w_abs / 2)
                        y0 = max(0, y_center_abs - h_abs / 2)
                        x1 = min(width, x_center_abs + w_abs / 2)
                        y1 = min(height, y_center_abs + h_abs / 2)
                        
                        if class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                        else:
                            class_name = f"class_{class_id}"
                        
                        annotations.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "bbox": [x0, y0, x1, y1],
                            "type": "box"
                        })
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")
        
        return annotations

    def safe_get_predictions(self, results):
        """Extract predictions from model results"""
        if not results or not isinstance(results, list):
            return []
        
        result = results[0]
        
        if not isinstance(result, dict):
            return []
        
        if "extracted_predictions" in result:
            extracted = result["extracted_predictions"]
            
            if isinstance(extracted, dict):
                predictions_list = []
                for category, preds in extracted.items():
                    for pred in preds:
                        pred["category"] = category
                        predictions_list.append(pred)
                return predictions_list
            elif isinstance(extracted, list):
                return extracted
        
        return []

    def convert_list_to_dict_format(self, predictions_list, include_text_score=True):
        """Convert list predictions back to dict format for visualization with text score info"""
        predictions_dict = {
            'high_confidence': {},  # Прошли текстовую проверку
            'low_text_score': {}    # Не прошли текстовую проверку
        }
        
        for pred in predictions_list:
            if isinstance(pred, dict) and 'category' in pred:
                category = pred['category']
                passes_threshold = pred.get('passes_text_threshold', True)
                text_score = pred.get('text_score', 1.0)
                
                # Выбираем словарь в зависимости от результата проверки
                if passes_threshold:
                    target_dict = predictions_dict['high_confidence']
                    # Добавляем скор к метке для отображения
                    if include_text_score:
                        display_category = f"{category} (score:{text_score:.2f})"
                    else:
                        display_category = category
                else:
                    target_dict = predictions_dict['low_text_score']
                    if include_text_score:
                        display_category = f"{category} (score:{text_score:.2f})"
                    else:
                        display_category = category
                
                if category not in target_dict:
                    target_dict[category] = []
                
                # Создаем копию с обновленной категорией для отображения
                pred_copy = pred.copy()
                pred_copy['category'] = display_category
                
                target_dict[category].append(pred_copy)
        
        return predictions_dict

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2
        
        # Calculate intersection area
        inter_x1 = max(x11, x21)
        inter_y1 = max(y11, y21)
        inter_x2 = min(x12, x22)
        inter_y2 = min(y12, y22)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        
        # Calculate union area
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def calculate_metrics_per_image(self, predictions, ground_truths, iou_threshold=0.5):
        """
        Calculate metrics for single image with improved matching (YOLO-style)
        """
        # Convert ground truths to YOLO format: [x1, y1, x2, y2, class_id]
        gt_boxes = []
        for gt in ground_truths:
            if 'bbox' in gt:
                class_id = gt.get('class_id', 0)
                # Map class name to ID if class_id not available
                if class_id == 0 and 'class_name' in gt:
                    try:
                        class_id = self.class_names.index(gt['class_name'])
                    except ValueError:
                        continue
                gt_boxes.append([gt['bbox'][0], gt['bbox'][1], gt['bbox'][2], gt['bbox'][3], class_id])
        
        # Convert predictions to YOLO format: [x1, y1, x2, y2, confidence, class_id]
        pred_boxes = []
        for pred in predictions:
            if isinstance(pred, dict) and 'coords' in pred:
                coords = pred['coords']
                # Get class ID from category name
                category = pred.get('category', '').split(' (score')[0]  # Удаляем часть со скором
                try:
                    class_id = self.class_names.index(category)
                except ValueError:
                    continue
                
                confidence = pred.get('score', 1.0)  # Use score if available, else default to 1.0
                pred_boxes.append([coords[0], coords[1], coords[2], coords[3], confidence, class_id])
        
        # Handle edge cases
        if not gt_boxes and not pred_boxes:
            return {'precision': 1.0, 'recall': 1.0, 'mean_iou': 1.0, 'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}
        elif not gt_boxes:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0, 'ious': []}
        elif not pred_boxes:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': 0, 'fn': len(gt_boxes), 'ious': []}
        
        # Sort predictions by confidence (YOLO-style matching)
        predictions_sorted = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
        
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        matched_ious = []
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(gt_boxes)
        
        for pred in predictions_sorted:
            pred_box = pred[:4]
            pred_class = pred[5]
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for i, gt in enumerate(gt_boxes):
                if gt_matched[i]:
                    continue
                    
                gt_box = gt[:4]
                gt_class = gt[4]
                
                # Only match if classes are the same
                if gt_class != pred_class:
                    continue
                    
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Check if match meets IoU threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                fn -= 1
                gt_matched[best_gt_idx] = True
                matched_ious.append(best_iou)
            else:
                fp += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        mean_iou = np.mean(matched_ious) if matched_ious else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'mean_iou': mean_iou,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ious': matched_ious
        }
    
    def calculate_detection_metrics(self, images_results):
        """
        Calculate comprehensive detection metrics (YOLO-style)
        images_results: list of tuples (predictions, ground_truths) for each image
        """
        total_metrics = {
            'precision': [],
            'recall': [],
            'mean_iou': [],
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'all_ious': []
        }
        
        for predictions, ground_truths in images_results:
            metrics = self.calculate_metrics_per_image(predictions, ground_truths, iou_threshold=0.5)
            
            total_metrics['precision'].append(metrics['precision'])
            total_metrics['recall'].append(metrics['recall'])
            total_metrics['mean_iou'].append(metrics['mean_iou'])
            total_metrics['tp'] += metrics['tp']
            total_metrics['fp'] += metrics['fp']
            total_metrics['fn'] += metrics['fn']
            total_metrics['all_ious'].extend(metrics['ious'])
        
        # Calculate comprehensive metrics
        avg_precision = np.mean(total_metrics['precision'])
        avg_recall = np.mean(total_metrics['recall'])
        avg_iou = np.mean(total_metrics['mean_iou'])
        overall_iou = np.mean(total_metrics['all_ious']) if total_metrics['all_ious'] else 0
        
        # Micro averages
        micro_precision = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fp']) if (total_metrics['tp'] + total_metrics['fp']) > 0 else 0
        micro_recall = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fn']) if (total_metrics['tp'] + total_metrics['fn']) > 0 else 0
        
        # F1 score
        f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        return {
            'macro_metrics': {
                'precision': avg_precision,
                'recall': avg_recall,
                'iou': avg_iou
            },
            'micro_metrics': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1_score': f1_score,
                'iou': overall_iou
            },
            'counts': {
                'tp': total_metrics['tp'],
                'fp': total_metrics['fp'],
                'fn': total_metrics['fn'],
                'total_images': len(images_results),
                'total_predictions': total_metrics['tp'] + total_metrics['fp'],
                'total_ground_truths': total_metrics['tp'] + total_metrics['fn']
            },
            'iou_analysis': {
                'mean_iou': overall_iou,
                'matched_pairs': len(total_metrics['all_ious']),
                'all_ious': total_metrics['all_ious']
            }
        }

    def create_visualization_with_custom_colors(self, image, predictions_dict, color_mapping):
        """
        Создает визуализацию с кастомными цветами для разных категорий
        
        Args:
            image: PIL.Image - исходное изображение
            predictions_dict: dict - словарь предсказаний
            color_mapping: dict - маппинг категорий на цвета в формате {category: (R, G, B)}
        """
        # Создаем визуализацию с использованием color_mapping
        # Если RexOmniVisualize поддерживает colors параметр
        try:
            # Способ 1: Попробуем передать colors параметр
            vis = RexOmniVisualize(
                image=image,
                predictions=predictions_dict,
                font_size=10,
                draw_width=2,
                show_labels=True,
                colors=color_mapping  # Передаем словарь цветов
            )
            return vis
        except TypeError as e:
            # Если не поддерживается colors параметр, используем альтернативный подход
            print(f"Note: RexOmniVisualize doesn't support colors parameter directly. Using default colors.")
            # Создаем обычную визуализацию
            vis = RexOmniVisualize(
                image=image,
                predictions=predictions_dict,
                font_size=10,
                draw_width=2,
                show_labels=True,
            )
            # Если нужно изменить цвета после создания, можно использовать PIL для постобработки
            # Но это сложнее, поэтому пока просто используем стандартные цвета
            return vis

    def evaluate_dataset(self, output_dir="evaluation_results", max_images=None):
        """Run evaluation on the entire dataset with text alignment filtering"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subdirectories for visualizations
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
        
        images_results = []  # Store (predictions, ground_truths) for each image
        
        image_files = list(self.images_path.glob("*.*"))
        if max_images:
            image_files = image_files[:max_images]
            
        print(f"Evaluating on {len(image_files)} images...")
        print(f"Text alignment threshold: {self.text_threshold}")
        
        for i, image_path in tqdm(enumerate(image_files), total=len(image_files)):
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Read ground truth annotations
                ground_truths = self.read_yolo_annotations(image_path)
                self.stats['total_ground_truths'] += len(ground_truths)
                
                # Run inference
                results = self.rex.inference(
                    images=image, 
                    task="detection", 
                    categories=self.class_names
                )
                
                # Extract predictions
                predictions = self.safe_get_predictions(results)
                
                # Filter only box predictions and apply text alignment check
                box_predictions = []
                for pred in predictions:
                    if isinstance(pred, dict) and pred.get("type") == "box":
                        # Проверяем текстовое соответствие для каждого bounding box
                        if 'coords' in pred and 'category' in pred:
                            text_score = self.check_text_alignment(
                                image, 
                                pred['coords'], 
                                pred['category']
                            )
                            pred['text_score'] = text_score
                            
                            # Добавляем флаг, проходит ли предсказание текстовый порог
                            passes_threshold = text_score >= self.text_threshold
                            pred['passes_text_threshold'] = passes_threshold
                            
                            # Считаем статистику по текстовым скорам
                            if passes_threshold:
                                self.stats['high_text_score'] += 1
                            else:
                                self.stats['low_text_score'] += 1
                            
                            # Всегда добавляем в box_predictions для визуализации, но метрики считаем только для прошедших порог
                            box_predictions.append(pred)
                        else:
                            # Если нет координат или категории, сохраняем с дефолтными значениями
                            pred['text_score'] = 1.0
                            pred['passes_text_threshold'] = True
                            self.stats['high_text_score'] += 1
                            box_predictions.append(pred)
                
                # Обновляем статистику
                self.stats['total_predictions'] += len(box_predictions)
                
                # Для метрик используем только предсказания, прошедшие текстовый порог
                filtered_predictions = [p for p in box_predictions if p.get('passes_text_threshold', True)]
                self.stats['total_filtered_predictions'] += len(filtered_predictions)
                
                # Store results for this image (используем фильтрованные предсказания для метрик)
                images_results.append((filtered_predictions, ground_truths))
                
                # Convert to dict format for visualization
                predictions_dict = self.convert_list_to_dict_format(box_predictions, include_text_score=True)
                
                # Save visualizations with different approaches
                try:
                    # ВАРИАНТ 1: Раздельные визуализации с разными цветами
                    # Создаем цветовую схему для high_confidence (зеленый) и low_text_score (красный)
                    
                    # Для high_confidence
                    if predictions_dict['high_confidence']:
                        # Создаем маппинг цветов для категорий с высоким скором
                        high_color_mapping = {}
                        for category in predictions_dict['high_confidence'].keys():
                            # Зеленый цвет в формате строки для RexOmniVisualize
                            # Попробуем разные форматы
                            high_color_mapping[category] = 'green'  # или (0, 255, 0)
                        
                        # Пробуем создать визуализацию с цветами
                        try:
                            vis_high = RexOmniVisualize(
                                image=image,
                                predictions=predictions_dict['high_confidence'],
                                font_size=10,
                                draw_width=2,
                                show_labels=True,
                            )
                            vis_high_filename = f"{image_path.stem}_high_confidence.jpg"
                            vis_high.save(str(vis_path / vis_high_filename))
                        except Exception as e:
                            print(f"High confidence visualization error for {image_path}: {e}")
                    
                    # Для low_text_score
                    if predictions_dict['low_text_score']:
                        # Создаем маппинг цветов для категорий с низким скором
                        low_color_mapping = {}
                        for category in predictions_dict['low_text_score'].keys():
                            # Красный цвет
                            low_color_mapping[category] = 'red'  # или (255, 0, 0)
                        
                        # Пробуем создать визуализацию с цветами
                        try:
                            vis_low = RexOmniVisualize(
                                image=image,
                                predictions=predictions_dict['low_text_score'],
                                font_size=10,
                                draw_width=2,
                                show_labels=True,
                            )
                            vis_low_filename = f"{image_path.stem}_low_confidence.jpg"
                            vis_low.save(str(vis_path / vis_low_filename))
                        except Exception as e:
                            print(f"Low confidence visualization error for {image_path}: {e}")
                    
                    # ВАРИАНТ 2: Комбинированная визуализация (все предсказания на одном изображении)
                    if predictions_dict['high_confidence'] or predictions_dict['low_text_score']:
                        # Создаем комбинированный словарь для визуализации
                        combined_predictions = {}
                        for cat in set(list(predictions_dict['high_confidence'].keys()) + 
                                     list(predictions_dict['low_text_score'].keys())):
                            combined_predictions[cat] = []
                            if cat in predictions_dict['high_confidence']:
                                combined_predictions[cat].extend(predictions_dict['high_confidence'][cat])
                            if cat in predictions_dict['low_text_score']:
                                combined_predictions[cat].extend(predictions_dict['low_text_score'][cat])
                        
                        # Создаем комбинированную визуализацию
                        vis_combined = RexOmniVisualize(
                            image=image,
                            predictions=combined_predictions,
                            font_size=10,
                            draw_width=2,
                            show_labels=True,
                        )
                        vis_combined_filename = f"{image_path.stem}_combined.jpg"
                        vis_combined.save(str(vis_path / vis_combined_filename))
                    
                    # ВАРИАНТ 3: Создаем текстовый файл с метаданными для цветовой дифференциации
                    metadata = {
                        'image': image_path.name,
                        'high_confidence_predictions': [],
                        'low_text_score_predictions': []
                    }
                    
                    for pred in box_predictions:
                        pred_info = {
                            'category': pred.get('category', ''),
                            'coords': pred.get('coords', []),
                            'text_score': pred.get('text_score', 0),
                            'passes_threshold': pred.get('passes_text_threshold', False)
                        }
                        
                        if pred.get('passes_text_threshold', False):
                            metadata['high_confidence_predictions'].append(pred_info)
                        else:
                            metadata['low_text_score_predictions'].append(pred_info)
                    
                    # Сохраняем метаданные
                    metadata_path = vis_path / f"{image_path.stem}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                except Exception as e:
                    print(f"Visualization error for {image_path}: {e}")
                
                self.stats['successful_inferences'] += 1
                self.stats['total_images'] += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                self.stats['failed_inferences'] += 1
                self.stats['total_images'] += 1
                continue
        
        # Print enhanced statistics
        print(f"\n=== PROCESSING STATISTICS ===")
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"  Successful: {self.stats['successful_inferences']}")
        print(f"  Failed: {self.stats['failed_inferences']}")
        print(f"\nText alignment filtering (threshold: {self.text_threshold}):")
        print(f"  Total predictions before filtering: {self.stats['total_predictions']}")
        print(f"  Total predictions after filtering: {self.stats['total_filtered_predictions']}")
        print(f"  High text score predictions: {self.stats['high_text_score']}")
        print(f"  Low text score predictions: {self.stats['low_text_score']}")
        
        if self.stats['total_predictions'] > 0:
            filter_ratio = self.stats['low_text_score'] / self.stats['total_predictions'] * 100
            print(f"  Filtered out: {filter_ratio:.1f}%")
        
        print(f"\nTotal ground truths: {self.stats['total_ground_truths']}")
        
        # Calculate metrics using YOLO-style calculation
        if images_results:
            metrics = self.calculate_detection_metrics(images_results)
            
            # Save metrics
            metrics_path = output_path / "detection_metrics.json"
            with open(metrics_path, 'w') as f:
                def convert_types(obj):
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_types(item) for item in obj]
                    return obj
                
                # Добавляем статистику по текстовой фильтрации в метрики
                metrics['text_filtering_stats'] = {
                    'text_threshold': self.text_threshold,
                    'total_predictions': self.stats['total_predictions'],
                    'filtered_predictions': self.stats['total_filtered_predictions'],
                    'high_text_score': self.stats['high_text_score'],
                    'low_text_score': self.stats['low_text_score'],
                    'filter_ratio': self.stats['low_text_score'] / self.stats['total_predictions'] if self.stats['total_predictions'] > 0 else 0
                }
                
                json.dump(convert_types(metrics), f, indent=2)
            
            # Print summary
            self.print_metrics_summary(metrics)
            
            return metrics
        else:
            print("No valid predictions or ground truths to evaluate!")
            return None
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary with YOLO-style metrics and text filtering info"""
        print("\n" + "="*60)
        print("YOLO-STYLE DETECTION METRICS SUMMARY (WITH TEXT FILTERING)")
        print("="*60)
        
        macro = metrics['macro_metrics']
        micro = metrics['micro_metrics']
        counts = metrics['counts']
        iou_analysis = metrics['iou_analysis']
        text_stats = metrics.get('text_filtering_stats', {})
        
        print(f"\nProcessed images: {counts['total_images']}")
        print(f"Total detections (after text filtering): {counts['total_predictions']}")
        print(f"Total ground truth objects: {counts['total_ground_truths']}")
        
        if text_stats:
            print(f"\n--- Text Filtering Statistics ---")
            print(f"Text threshold: {text_stats.get('text_threshold', self.text_threshold)}")
            print(f"Predictions before filtering: {text_stats.get('total_predictions', 0)}")
            print(f"Predictions after filtering: {text_stats.get('filtered_predictions', 0)}")
            print(f"Filtered out: {text_stats.get('low_text_score', 0)} ({text_stats.get('filter_ratio', 0)*100:.1f}%)")
        
        print(f"\n--- Per-image Metrics (Macro) ---")
        print(f"Average Precision: {macro['precision']:.4f}")
        print(f"Average Recall:    {macro['recall']:.4f}")
        print(f"Average IoU:       {macro['iou']:.4f}")
        
        print(f"\n--- Overall Metrics (Micro) ---")
        print(f"Micro Precision:   {micro['precision']:.4f}")
        print(f"Micro Recall:      {micro['recall']:.4f}")
        print(f"F1-Score:          {micro['f1_score']:.4f}")
        print(f"Overall IoU:       {micro['iou']:.4f}")
        
        print(f"\n--- Detection Counts ---")
        print(f"True Positives (TP):  {counts['tp']}")
        print(f"False Positives (FP): {counts['fp']}")
        print(f"False Negatives (FN): {counts['fn']}")
        
        print(f"\n--- IoU Analysis ---")
        print(f"Mean IoU:          {iou_analysis['mean_iou']:.4f}")
        print(f"Matched pairs:     {iou_analysis['matched_pairs']}")

# Usage example
if __name__ == "__main__":
    # Specify your class names
    class_names = ['-', 'Apple', 'Artichoke', 'Asparagus', 'Avocado', 'Banana', 'Beans', 'Beetroot', 'Blackberries', 'Blueberries', 'Book', 'Broccoli', 'Brussel Sprouts', 'Butter', 'Cabbage', 'Cantaloupe', 'Carrots', 'Cauliflower', 'Cerealbox', 'Cheese', 'Clementine', 'Coffee', 'Corn', 'Cucumber', 'Detergent', 'Drinks', 'Egg', 'Eggplant', 'Eggs', 'Fish', 'Galia', 'Grapes', 'Honeydew', 'Juice', 'Lemon', 'Lettuce', 'Meat', 'Milk', 'Mushroom', 'Mushrooms', 'Nectarine', 'Onion', 'Orange', 'Oranges', 'Peas', 'Pineapple', 'Plum', 'Pomegranate', 'Potato', 'Raspberries', 'Salad', 'Sauce', 'Shallot', 'Spinach', 'Squash', 'Strawberries', 'Strawberry', 'Sweetcorn', 'Tofu', 'Tomato', 'Tomatoes', 'Watermelon', 'Yogurt', 'Zucchini', 'apple', 'apples', 'asparagus', 'aubergine', 'bacon', 'banana', 'bananas', 'bazlama', 'beef', 'blueberries', 'bread', 'broccoli', 'butter', 'carrot', 'carrots', 'cheese', 'chicken', 'chicken_breast', 'chocolate', 'chocolate chips', 'corn', 'courgettes', 'cream', 'cream cheese', 'dates', 'eggs', 'flour', 'ginger', 'goat_cheese', 'green beans', 'green bell pepper', 'green chilies', 'green_beans', 'ground_beef', 'ham', 'heavy_cream', 'juice', 'lemon', 'lemons', 'lettuce', 'lime', 'mango', 'meat', 'milk', 'mineral water', 'mushroom', 'mushrooms', 'olive', 'olives', 'onion', 'orange', 'parsley', 'peach', 'peppers', 'potato', 'potatoes', 'red bell pepper', 'red grapes', 'red onion', 'salami', 'sauce', 'sausage', 'shrimp', 'spinach', 'spring onion', 'strawberries', 'strawberry', 'sugar', 'sweet_potato', 'tomato', 'tomato paste', 'tomatoes', 'yellow bell pepper', 'yoghurt']
    # class_names = ['car', 'door', 'handrail', 'sidewalk', 'staircase', 'street_light', 'window']
    
    # Initialize evaluator with text threshold (рекомендуется начать с 0.2-0.3)
    evaluator = YOLODatasetEvaluator(
        dataset_path="Grocery1-1/test",
        class_names=class_names,
        model_path="IDEA-Research/Rex-Omni",
        text_threshold=0.15  # Порог текстового соответствия
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        output_dir="evaluation_results",
        max_images=100  # Set to None for all images, or specify a number
    )