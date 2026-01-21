import os
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
from collections import defaultdict

class OpenImagesDatasetEvaluator:
    def __init__(self, dataset_path, model_path="IDEA-Research/Rex-Omni"):
        self.dataset_path = Path(dataset_path)
        
        # Диагностика структуры папок
        print(f"Dataset path: {self.dataset_path}")
        print(f"Path exists: {self.dataset_path.exists()}")
        
        if self.dataset_path.exists():
            print("Contents of dataset directory:")
            for item in self.dataset_path.iterdir():
                print(f"  - {item.name} (dir: {item.is_dir()})")
        
        # Пытаемся найти правильные пути
        self.images_path = self._find_images_path()
        self.labels_path = self._find_labels_path()
        self.metadata_path = self._find_metadata_path()
        
        print(f"Images path: {self.images_path}")
        print(f"Labels path: {self.labels_path}")
        print(f"Metadata path: {self.metadata_path}")
        
        # Загружаем mapping LabelName -> human-readable name
        self.label_name_to_class_name = self._load_label_name_mapping()
        
        # Получаем все классы из classes.csv
        self.class_names = self._get_classes_from_csv()
        print(f"Loaded {len(self.class_names)} classes from Open Images V7")
        
        # Создаем mapping имя класса -> id для быстрого доступа
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Загружаем все ground truth из detections.csv
        self.all_ground_truths = self._load_all_ground_truths()
        print(f"Loaded ground truths for {len(self.all_ground_truths)} images from detections.csv")
        
        # Initialize RexOmni model
        self.rex = RexOmniWrapper(
            model_path=model_path,
            backend="transformers",
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            top_k=10,
            repetition_penalty=1,
        )
        
        self.stats = {
            'total_images': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_predictions': 0,
            'total_ground_truths': 0
        }

    def _load_label_name_mapping(self):
        """Загружаем mapping из LabelName в человеко-читаемые имена классов"""
        mapping_paths = [
            self.metadata_path / "class-descriptions.csv",
            self.metadata_path / "classes.csv",
            self.dataset_path / "class-descriptions.csv"
        ]
        
        label_name_to_class_name = {}
        
        for mapping_path in mapping_paths:
            if mapping_path.exists():
                try:
                    print(f"Loading label mapping from: {mapping_path}")
                    df = pd.read_csv(mapping_path, header=None, names=['LabelName', 'ClassName'])
                    
                    for _, row in df.iterrows():
                        label_name_to_class_name[row['LabelName']] = row['ClassName']
                    
                    print(f"Loaded {len(label_name_to_class_name)} label mappings")
                    return label_name_to_class_name
                    
                except Exception as e:
                    print(f"Error reading mapping file {mapping_path}: {e}")
        
        print("WARNING: No label mapping file found!")
        return {}

    def _load_all_ground_truths(self):
        """Загружаем все ground truth из detections.csv"""
        detections_csv_path = self.labels_path / "detections.csv"
        
        if not detections_csv_path.exists():
            # Пробуем найти в других местах
            possible_paths = [
                self.labels_path / "detections.csv",
                self.dataset_path / "detections.csv",
                self.labels_path / "labels.csv"
            ]
            for path in possible_paths:
                if path.exists():
                    detections_csv_path = path
                    break
        
        if not detections_csv_path.exists():
            print(f"WARNING: detections.csv not found at any location")
            return {}
        
        try:
            print(f"Loading detections from: {detections_csv_path}")
            df = pd.read_csv(detections_csv_path)
            print(f"Columns in detections.csv: {df.columns.tolist()}")
            print(f"Total rows in detections.csv: {len(df)}")
            
            # Создаем словарь для хранения аннотаций по image_id
            ground_truths = {}
            valid_annotations = 0
            skipped_annotations = 0
            
            for _, row in df.iterrows():
                image_id = row['ImageID']
                
                # Получаем класс через mapping
                label_name = row['LabelName']
                if label_name not in self.label_name_to_class_name:
                    skipped_annotations += 1
                    continue
                
                class_name = self.label_name_to_class_name[label_name]
                
                if class_name not in self.class_to_id:
                    skipped_annotations += 1
                    continue
                
                # Получаем координаты (относительные)
                x_min = row['XMin']
                y_min = row['YMin'] 
                x_max = row['XMax']
                y_max = row['YMax']
                
                # Проверяем валидность координат
                if x_min >= x_max or y_min >= y_max:
                    skipped_annotations += 1
                    continue
                
                if image_id not in ground_truths:
                    ground_truths[image_id] = []
                
                ground_truths[image_id].append({
                    "class_id": self.class_to_id[class_name],
                    "class_name": class_name,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "type": "box"
                })
                valid_annotations += 1
            
            print(f"Valid annotations: {valid_annotations}, Skipped: {skipped_annotations}")
            return ground_truths
            
        except Exception as e:
            print(f"Error reading detections.csv: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _find_images_path(self):
        """Находим путь к изображениям"""
        possible_paths = [
            self.dataset_path / "data",
            self.dataset_path / "images",
            self.dataset_path / "train",
            self.dataset_path / "test",
            self.dataset_path / "validation",
            self.dataset_path,
        ]
        
        for path in possible_paths:
            if path.exists():
                image_files = list(path.glob("*.*"))
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                has_images = any(f.suffix.lower() in image_extensions for f in image_files)
                if has_images:
                    print(f"Found images at: {path} ({len(image_files)} files)")
                    return path
        
        print("WARNING: No images directory found!")
        return self.dataset_path / "data"

    def _find_labels_path(self):
        """Находим путь к labels"""
        possible_paths = [
            self.dataset_path / "labels",
            self.dataset_path / "labels" / "detections",
            self.dataset_path,
        ]
        
        for path in possible_paths:
            if path.exists():
                # Ищем detections.csv вместо .txt файлов
                csv_files = list(path.glob("*.csv"))
                if csv_files:
                    print(f"Found labels at: {path} ({len(csv_files)} files)")
                    return path
        
        print("WARNING: No labels directory found!")
        return self.dataset_path / "labels"

    def _find_metadata_path(self):
        """Находим путь к metadata"""
        possible_paths = [
            self.dataset_path / "metadata",
            self.dataset_path,
        ]
        
        for path in possible_paths:
            if path.exists():
                metadata_files = list(path.glob("*.csv")) + list(path.glob("*.json"))
                if metadata_files:
                    print(f"Found metadata at: {path} ({len(metadata_files)} files)")
                    return path
        
        print("WARNING: No metadata directory found!")
        return self.dataset_path / "metadata"

    def _get_classes_from_csv(self):
        """Получаем все классы из файла classes.csv"""
        classes_csv_path = self.metadata_path / "classes.csv"
        
        if not classes_csv_path.exists():
            print(f"Classes CSV not found at {classes_csv_path}")
            for possible_path in [
                self.dataset_path / "classes.csv",
                self.metadata_path / "class-descriptions.csv",
                self.metadata_path / "categories.csv"
            ]:
                if possible_path.exists():
                    classes_csv_path = possible_path
                    print(f"Found classes file at: {classes_csv_path}")
                    break
        
        if not classes_csv_path.exists():
            print("No classes CSV file found, using default classes")
            return self._get_default_open_images_classes()
        
        try:
            df = pd.read_csv(classes_csv_path)
            print(f"Columns in classes file: {df.columns.tolist()}")
            print(f"First few rows:\n{df.head()}")
            
            # Если файл class-descriptions.csv, используем вторую колонку
            if len(df.columns) == 2 and df.columns[0] == '/m/011k07' and df.columns[1] == 'Tortoise':
                class_names = df.iloc[:, 1].tolist()
            else:
                possible_class_columns = ['class_name', 'name', 'label', 'display_name', 'category', 'Class Name']
                class_column = None
                
                for col in possible_class_columns:
                    if col in df.columns:
                        class_column = col
                        break
                
                if class_column is None:
                    class_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    print(f"Using column '{class_column}' as class names")
                
                class_names = df[class_column].tolist()
            
            class_names = [str(name) for name in class_names if pd.notna(name) and str(name).strip() != '']
            class_names = list(dict.fromkeys(class_names))  # Remove duplicates while preserving order
            
            print(f"First 10 classes: {class_names[:10]}")
            print(f"Total classes: {len(class_names)}")
            return class_names
            
        except Exception as e:
            print(f"Error reading classes CSV: {e}")
            return self._get_default_open_images_classes()

    def _get_default_open_images_classes(self):
        """Резервный список основных классов Open Images V7"""
        return [
            'Person', 'Vehicle', 'Car', 'Bicycle', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat',
            'Traffic light', 'Fire hydrant', 'Stop sign', 'Parking meter', 'Bench', 'Bird', 'Cat', 'Dog', 'Horse',
            'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie',
            'Suitcase', 'Frisbee', 'Skis', 'Snowboard', 'Sports ball', 'Kite', 'Baseball bat', 'Baseball glove',
            'Skateboard', 'Surfboard', 'Tennis racket', 'Bottle', 'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon',
            'Bowl', 'Banana', 'Apple', 'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut',
            'Cake', 'Chair', 'Couch', 'Potted plant', 'Bed', 'Dining table', 'Toilet', 'TV', 'Laptop', 'Mouse',
            'Remote', 'Keyboard', 'Cell phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book',
            'Clock', 'Vase', 'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush'
        ]

    def get_image_files(self):
        """Получаем список файлов изображений"""
        if not self.images_path.exists():
            print(f"Images path does not exist: {self.images_path}")
            return []
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.images_path.glob(ext))
            image_files.extend(self.images_path.glob(ext.upper()))
        
        print(f"Found {len(image_files)} image files")
        
        if not image_files:
            subdirs = [d for d in self.images_path.iterdir() if d.is_dir()]
            print(f"Checking {len(subdirs)} subdirectories for images")
            
            for subdir in subdirs:
                for ext in image_extensions:
                    image_files.extend(subdir.glob(ext))
                    image_files.extend(subdir.glob(ext.upper()))
            
            print(f"Total images found after checking subdirectories: {len(image_files)}")
        
        return image_files

    def read_open_images_annotations(self, image_path):
        """Read Open Images V7 format annotations from preloaded detections.csv"""
        image_id = image_path.stem
        
        if image_id in self.all_ground_truths:
            return self.all_ground_truths[image_id]
        else:
            return []

    # Остальные методы остаются без изменений
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

    def convert_list_to_dict_format(self, predictions_list):
        """Convert list predictions back to dict format for visualization"""
        predictions_dict = {}
        for pred in predictions_list:
            if isinstance(pred, dict) and 'category' in pred:
                category = pred['category']
                if category not in predictions_dict:
                    predictions_dict[category] = []
                pred_copy = pred.copy()
                del pred_copy['category']
                predictions_dict[category].append(pred_copy)
        return predictions_dict

    def calculate_iou_batch(self, boxes1, boxes2):
        """Векторизованный расчет IoU между двумя наборами боксов"""
        # boxes1: [N, 4], boxes2: [M, 4]
        x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
        
        # Вычисляем площади
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        
        # Вычисляем пересечения
        inter_x1 = np.maximum(x11, x21.T)
        inter_y1 = np.maximum(y11, y21.T)
        inter_x2 = np.minimum(x12, x22.T)
        inter_y2 = np.minimum(y12, y22.T)
        
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Вычисляем IoU
        union_area = area1 + area2.T - inter_area
        iou = inter_area / union_area
        
        return iou

    def calculate_metrics_per_image(self, predictions, ground_truths, iou_threshold=0.5):
        """
        Calculate metrics for single image with optimized matching
        """
        if not ground_truths and not predictions:
            return {'precision': 1.0, 'recall': 1.0, 'mean_iou': 1.0, 'tp': 0, 'fp': 0, 'fn': 0, 'ious': []}
        elif not ground_truths:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': len(predictions), 'fn': 0, 'ious': []}
        elif not predictions:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': 0, 'fn': len(ground_truths), 'ious': []}
        
        # Prepare ground truth arrays
        gt_boxes = []
        gt_classes = []
        for gt in ground_truths:
            if 'bbox' in gt:
                class_id = gt.get('class_id', 0)
                if class_id == 0 and 'class_name' in gt:
                    class_id = self.class_to_id.get(gt['class_name'], -1)
                if class_id != -1:
                    gt_boxes.append(gt['bbox'])
                    gt_classes.append(class_id)
        
        if not gt_boxes:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': len(predictions), 'fn': 0, 'ious': []}
        
        gt_boxes = np.array(gt_boxes)
        gt_classes = np.array(gt_classes)
        
        # Prepare prediction arrays
        pred_boxes = []
        pred_classes = []
        pred_confidences = []
        for pred in predictions:
            if isinstance(pred, dict) and 'coords' in pred:
                coords = pred['coords']
                category = pred.get('category', '')
                class_id = self.class_to_id.get(category, -1)
                if class_id != -1:
                    pred_boxes.append(coords)
                    pred_classes.append(class_id)
                    pred_confidences.append(pred.get('score', 1.0))
        
        if not pred_boxes:
            return {'precision': 0.0, 'recall': 0.0, 'mean_iou': 0.0, 'tp': 0, 'fp': 0, 'fn': len(ground_truths), 'ious': []}
        
        pred_boxes = np.array(pred_boxes)
        pred_classes = np.array(pred_classes)
        pred_confidences = np.array(pred_confidences)
        
        # Sort predictions by confidence
        sort_idx = np.argsort(-pred_confidences)
        pred_boxes = pred_boxes[sort_idx]
        pred_classes = pred_classes[sort_idx]
        pred_confidences = pred_confidences[sort_idx]
        
        # Calculate IoU matrix
        iou_matrix = self.calculate_iou_batch(pred_boxes, gt_boxes)
        
        # Create class matching matrix
        class_match_matrix = (pred_classes[:, None] == gt_classes[None, :])
        
        # Combine IoU and class matching
        valid_matrix = (iou_matrix >= iou_threshold) & class_match_matrix
        
        # Greedy matching
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)
        tp = 0
        matched_ious = []
        
        for i in range(len(pred_boxes)):
            # Find best unmatched ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j in range(len(gt_boxes)):
                if not gt_matched[j] and valid_matrix[i, j] and iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_gt_idx = j
            
            if best_gt_idx != -1:
                tp += 1
                gt_matched[best_gt_idx] = True
                matched_ious.append(best_iou)
        
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
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
        """Calculate comprehensive detection metrics with optimization"""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        all_precisions = []
        all_recalls = []
        all_ious = []
        
        for predictions, ground_truths in images_results:
            metrics = self.calculate_metrics_per_image(predictions, ground_truths)
            
            all_precisions.append(metrics['precision'])
            all_recalls.append(metrics['recall'])
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            all_ious.extend(metrics['ious'])
        
        # Calculate metrics
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_iou = np.mean(all_ious) if all_ious else 0
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
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
                'iou': avg_iou
            },
            'counts': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'total_images': len(images_results),
                'total_predictions': total_tp + total_fp,
                'total_ground_truths': total_tp + total_fn
            },
            'iou_analysis': {
                'mean_iou': avg_iou,
                'matched_pairs': len(all_ious),
                'all_ious': all_ious
            }
        }

    def evaluate_dataset(self, output_dir="evaluation_results", max_images=None):
        """Run evaluation on the entire dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
        
        images_results = []
        
        image_files = self.get_image_files()
        if max_images:
            image_files = image_files[:max_images]
            
        print(f"Evaluating on {len(image_files)} images...")
        print(f"Using {len(self.class_names)} classes")
        
        if not image_files:
            print("No images found to evaluate!")
            return None
        
        for i, image_path in tqdm(enumerate(image_files), total=len(image_files)):
            try:
                image = Image.open(image_path).convert("RGB")
                
                # Read ground truth annotations from preloaded data
                ground_truths = self.read_open_images_annotations(image_path)
                self.stats['total_ground_truths'] += len(ground_truths)
                
                # Run inference
                results = self.rex.inference(
                    images=image, 
                    task="detection", 
                    categories=self.class_names
                )
                
                predictions = self.safe_get_predictions(results)
                self.stats['total_predictions'] += len(predictions)
                
                box_predictions = [pred for pred in predictions if isinstance(pred, dict) and pred.get("type") == "box"]
                
                images_results.append((box_predictions, ground_truths))
                
                # Visualization
                if box_predictions:
                    try:
                        predictions_dict = self.convert_list_to_dict_format(box_predictions)
                        vis = RexOmniVisualize(
                            image=image,
                            predictions=predictions_dict,
                            font_size=12,
                            draw_width=2,
                            show_labels=True,
                        )
                        vis_filename = f"{image_path.stem}_detection.jpg"
                        vis.save(str(vis_path / vis_filename))
                    except Exception as e:
                        print(f"Visualization error for {image_path}: {e}")
                
                self.stats['successful_inferences'] += 1
                self.stats['total_images'] += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                self.stats['failed_inferences'] += 1
                self.stats['total_images'] += 1
                continue
        
        # Print statistics
        print(f"\nProcessed {self.stats['total_images']} images:")
        print(f"  Successful: {self.stats['successful_inferences']}")
        print(f"  Failed: {self.stats['failed_inferences']}")
        print(f"  Total predictions: {self.stats['total_predictions']}")
        print(f"  Total ground truths: {self.stats['total_ground_truths']}")
        
        # Calculate metrics
        if images_results:
            metrics = self.calculate_detection_metrics(images_results)
            
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
                
                json.dump(convert_types(metrics), f, indent=2)
            
            self.print_metrics_summary(metrics)
            
            return metrics
        else:
            print("No valid predictions or ground truths to evaluate!")
            return None

    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary"""
        print("\n" + "="*60)
        print("DETECTION METRICS SUMMARY")
        print("="*60)
        
        macro = metrics['macro_metrics']
        micro = metrics['micro_metrics']
        counts = metrics['counts']
        iou_analysis = metrics['iou_analysis']
        
        print(f"\nProcessed images: {counts['total_images']}")
        print(f"Total detections: {counts['total_predictions']}")
        print(f"Total ground truth objects: {counts['total_ground_truths']}")
        
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
    evaluator = OpenImagesDatasetEvaluator(
        dataset_path="fiftyone/open-images-v7/test",
        model_path="IDEA-Research/Rex-Omni"
    )
    
    metrics = evaluator.evaluate_dataset(
        output_dir="open_images_evaluation_results",
        max_images=100
    )