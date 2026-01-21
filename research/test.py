import os
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

class YOLODatasetEvaluator:
    def __init__(self, dataset_path, class_names, model_path="IDEA-Research/Rex-Omni"):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / "images"
        self.labels_path = self.dataset_path / "labels"
        self.class_names = class_names
        
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

    def convert_list_to_dict_format(self, predictions_list):
        """Convert list predictions back to dict format for visualization"""
        predictions_dict = {}
        for pred in predictions_list:
            if isinstance(pred, dict) and 'category' in pred:
                category = pred['category']
                if category not in predictions_dict:
                    predictions_dict[category] = []
                pred_copy = pred.copy()
                predictions_dict[category].append(pred_copy)
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
    
    def calculate_detection_metrics(self, predictions, ground_truths, iou_threshold=0.5):
        """
        Calculate detection metrics with IoU analysis
        
        Метрики рассчитываются по следующему принципу:
        - Precision = TP / (TP + FP) - точность предсказаний
        - Recall = TP / (TP + FN) - полнота обнаружения
        - F1-Score = 2 * (Precision * Recall) / (Precision + Recall) - гармоническое среднее
        - mAP = средняя точность по всем классам
        - IoU = Intersection over Union - мера пересечения предсказания и ground truth
        """
        metrics = {
            'per_class': {},
            'overall': {},
            'iou_analysis': {}
        }
        
        all_classes = set()
        
        # Collect all classes from predictions and ground truths
        for gt in ground_truths:
            all_classes.add(gt['class_name'])
        
        pred_classes = set()
        for pred in predictions:
            if isinstance(pred, dict) and 'category' in pred:
                pred_classes.add(pred['category'])
        
        # IoU analysis storage
        iou_values = []
        matched_pairs = []
        
        for class_name in all_classes:
            class_gt = [gt for gt in ground_truths if gt['class_name'] == class_name]
            class_pred = [pred for pred in predictions if isinstance(pred, dict) and pred.get('category') == class_name]
            
            # Match predictions with ground truths
            tp = 0
            fp = 0
            fn = len(class_gt)
            
            used_gt = set()
            class_iou_values = []
            
            for pred in class_pred:
                if 'coords' not in pred:
                    fp += 1
                    continue
                
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(class_gt):
                    if i in used_gt:
                        continue
                    
                    iou = self.calculate_iou(pred['coords'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp += 1
                    fn -= 1
                    used_gt.add(best_gt_idx)
                    class_iou_values.append(best_iou)
                    iou_values.append(best_iou)
                    matched_pairs.append({
                        'class': class_name,
                        'iou': best_iou,
                        'pred_bbox': pred['coords'],
                        'gt_bbox': class_gt[best_gt_idx]['bbox']
                    })
                else:
                    fp += 1
            
            # Calculate metrics for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate IoU statistics for this class
            mean_iou = np.mean(class_iou_values) if class_iou_values else 0
            median_iou = np.median(class_iou_values) if class_iou_values else 0
            min_iou = np.min(class_iou_values) if class_iou_values else 0
            max_iou = np.max(class_iou_values) if class_iou_values else 0
            
            metrics['per_class'][class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'support': len(class_gt),
                'iou_mean': mean_iou,
                'iou_median': median_iou,
                'iou_min': min_iou,
                'iou_max': max_iou,
                'matched_detections': len(class_iou_values)
            }
        
        # Calculate overall metrics
        total_tp = sum(metrics['per_class'][cls]['tp'] for cls in metrics['per_class'])
        total_fp = sum(metrics['per_class'][cls]['fp'] for cls in metrics['per_class'])
        total_fn = sum(metrics['per_class'][cls]['fn'] for cls in metrics['per_class'])
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Calculate mAP (mean Average Precision)
        if metrics['per_class']:
            map_score = np.mean([metrics['per_class'][cls]['precision'] for cls in metrics['per_class']])
        else:
            map_score = 0
        
        # Calculate overall IoU statistics
        overall_mean_iou = np.mean(iou_values) if iou_values else 0
        overall_median_iou = np.median(iou_values) if iou_values else 0
        overall_min_iou = np.min(iou_values) if iou_values else 0
        overall_max_iou = np.max(iou_values) if iou_values else 0
        
        metrics['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'mAP': map_score,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'iou_mean': overall_mean_iou,
            'iou_median': overall_median_iou,
            'iou_min': overall_min_iou,
            'iou_max': overall_max_iou,
            'total_matched_pairs': len(iou_values)
        }
        
        metrics['iou_analysis'] = {
            'matched_pairs': matched_pairs,
            'all_iou_values': iou_values
        }
        
        return metrics
    
    def evaluate_dataset(self, output_dir="evaluation_results", max_images=None):
        """Run evaluation on the entire dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create subdirectories for visualizations
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
        
        all_predictions = []
        all_ground_truths = []
        
        image_files = list(self.images_path.glob("*.*"))
        if max_images:
            image_files = image_files[:max_images]
            
        print(f"Evaluating on {len(image_files)} images...")
        
        for i, image_path in tqdm(enumerate(image_files)):
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Read ground truth annotations
                ground_truths = self.read_yolo_annotations(image_path)
                all_ground_truths.extend(ground_truths)
                self.stats['total_ground_truths'] += len(ground_truths)
                
                # Run inference
                results = self.rex.inference(
                    images=image, 
                    task="detection", 
                    categories=self.class_names
                )
                
                # Extract predictions
                predictions = self.safe_get_predictions(results)
                self.stats['total_predictions'] += len(predictions)
                
                # Filter only box predictions
                box_predictions = [pred for pred in predictions if isinstance(pred, dict) and pred.get("type") == "box"]
                all_predictions.extend(box_predictions)
                
                # Convert to dict format for visualization
                predictions_dict = self.convert_list_to_dict_format(box_predictions)
                
                # Save visualization
                if predictions_dict:
                    try:
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
        if all_predictions and all_ground_truths:
            metrics = self.calculate_detection_metrics(all_predictions, all_ground_truths)
            
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
                
                json.dump(convert_types(metrics), f, indent=2)
            
            # Print summary
            self.print_metrics_summary(metrics)
            
            return metrics
        else:
            print("No valid predictions or ground truths to evaluate!")
            return None
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary with IoU analysis"""
        print("\n" + "="*80)
        print("DETECTION METRICS SUMMARY")
        print("="*80)
        
        print(f"\nOverall Metrics:")
        overall = metrics['overall']
        print(f"Precision: {overall['precision']:.4f}")
        print(f"Recall:    {overall['recall']:.4f}")
        print(f"F1-Score:  {overall['f1_score']:.4f}")
        print(f"mAP:       {overall['mAP']:.4f}")
        print(f"TP:        {overall['total_tp']}")
        print(f"FP:        {overall['total_fp']}")
        print(f"FN:        {overall['total_fn']}")
        
        print(f"\nIoU Analysis (Overall):")
        print(f"Mean IoU:   {overall['iou_mean']:.4f}")
        print(f"Median IoU: {overall['iou_median']:.4f}")
        print(f"Min IoU:    {overall['iou_min']:.4f}")
        print(f"Max IoU:    {overall['iou_max']:.4f}")
        print(f"Matched pairs: {overall['total_matched_pairs']}")
        
        if metrics['per_class']:
            print(f"\nPer-Class Metrics:")
            print(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU Mean':<10} {'Support':<10}")
            print("-" * 85)
            
            for class_name, class_metrics in metrics['per_class'].items():
                print(f"{class_name:<25} {class_metrics['precision']:<10.4f} "
                      f"{class_metrics['recall']:<10.4f} {class_metrics['f1_score']:<10.4f} "
                      f"{class_metrics['iou_mean']:<10.4f} {class_metrics['support']:<10}")

# Usage example
if __name__ == "__main__":
    # Specify your class names
    class_names = ['occupied', 'empty']
    
    # Initialize evaluator
    evaluator = YOLODatasetEvaluator(
        dataset_path="Parking-Space-1/test",
        class_names=class_names,
        model_path="IDEA-Research/Rex-Omni"
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        output_dir="evaluation_results",
        max_images=None  # Set to None for all images, or specify a number
    )