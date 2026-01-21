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
            temperature=0.75,
            top_p=0.7,
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
                category = pred.get('category', '')
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

    def evaluate_dataset(self, output_dir="evaluation_results", max_images=None):
        """Run evaluation on the entire dataset"""
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
                self.stats['total_predictions'] += len(predictions)
                
                # Filter only box predictions
                box_predictions = [pred for pred in predictions if isinstance(pred, dict) and pred.get("type") == "box"]
                
                # Store results for this image
                images_results.append((box_predictions, ground_truths))
                
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
                
                json.dump(convert_types(metrics), f, indent=2)
            
            # Print summary
            self.print_metrics_summary(metrics)
            
            return metrics
        else:
            print("No valid predictions or ground truths to evaluate!")
            return None
    
    def print_metrics_summary(self, metrics):
        """Print formatted metrics summary with YOLO-style metrics"""
        print("\n" + "="*60)
        print("YOLO-STYLE DETECTION METRICS SUMMARY (FOR COMPARISON)")
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
    # Specify your class names
    class_names = ['car', 'door', 'handrail', 'sidewalk', 'staircase', 'street_light', 'window']
    
    # Initialize evaluator
    evaluator = YOLODatasetEvaluator(
        dataset_path="Safety--&-Security-1/test",
        class_names=class_names,
        model_path="IDEA-Research/Rex-Omni"
    )
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        output_dir="evaluation_results",
        max_images=150  # Set to None for all images, or specify a number
    )