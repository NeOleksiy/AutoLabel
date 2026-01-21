import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rex_omni import RexOmniVisualize, RexOmniWrapper
import os
import json
import glob
from pathlib import Path
from tqdm.notebook import tqdm

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height
    
    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_classification_metrics(predictions, ground_truths, iou_threshold=0.5):
    """Calculate classification metrics and IoU"""
    tp, fp, fn = 0, 0, 0
    ious = []
    
    # Match predictions with ground truths
    matched_gt = set()
    
    # Sort predictions by confidence score (descending)
    try:
        sorted_predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
    except (TypeError, KeyError) as e:
        # If we can't sort, use as-is
        sorted_predictions = predictions
    
    for pred in sorted_predictions:
        # Ensure pred is a dictionary
        if not isinstance(pred, dict):
            continue
            
        # Try to get bbox from either 'bbox' or 'coords'
        pred_box = pred.get('bbox', [])
        if not pred_box and 'coords' in pred:
            pred_box = pred['coords']
            
        pred_class = pred.get('category', '')
        
        if not pred_box or not pred_class:
            continue
            
        max_iou = 0
        matched_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
                
            gt_box = gt.get('bbox', [])
            gt_class = gt.get('category', '')
            
            # Only compare if classes match and boxes are valid
            if pred_class == gt_class and pred_box and gt_box:
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > max_iou:
                    max_iou = iou
                    matched_idx = i
        
        if max_iou >= iou_threshold and matched_idx != -1:
            tp += 1
            matched_gt.add(matched_idx)
            ious.append(max_iou)
        else:
            fp += 1
    
    # Calculate false negatives (unmatched ground truths)
    fn = len(ground_truths) - len(matched_gt)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(ious) if ious else 0
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'ious': ious
    }

def load_yolo_annotations(txt_path, img_width, img_height):
    """Load YOLO format annotations and convert to absolute coordinates"""
    annotations = []
    
    if not os.path.exists(txt_path):
        return annotations
    
    # Your class names
    class_names = ['occupied-parking-spots', 'parking spot']
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                
                # Convert to [x1, y1, x2, y2] format
                x1 = max(0, x_center - width / 2)
                y1 = max(0, y_center - height / 2)
                x2 = min(img_width, x_center + width / 2)
                y2 = min(img_height, y_center + height / 2)
                
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                    annotations.append({
                        'category': class_name,
                        'bbox': [x1, y1, x2, y2],
                        'score': 1.0  # Ground truth has score 1.0
                    })
            except (ValueError, IndexError) as e:
                continue
    
    return annotations

def resize_image(image, target_size=512):
    """Resize image maintaining aspect ratio"""
    # Calculate new dimensions maintaining aspect ratio
    width, height = image.size
    
    # Determine the scaling factor
    scale = target_size / max(width, height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image, scale

def draw_predictions_manual(image, predictions, metrics=None):
    """Manual drawing of predictions on image with better visibility"""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Define colors for different classes with better contrast
    colors = {
        'occupied-parking-spots': (255, 0, 0),  # Red
        'parking spot': (0, 0, 255)  # Blue
    }
    
    # Try to load a larger font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Draw predictions with thicker lines and better text visibility
    for pred in predictions:
        # Get bbox coordinates
        bbox = pred.get('bbox', [])
        if not bbox and 'coords' in pred:
            bbox = pred['coords']
            
        if bbox and len(bbox) == 4:
            category = pred.get('category', 'unknown')
            color = colors.get(category, (0, 255, 0))  # Green as fallback
            
            # Draw thicker rectangle
            draw.rectangle(bbox, outline=color, width=5)
            
            # Draw label with background for better readability
            score = pred.get('score', 1.0)
            label = f"{category} ({score:.2f})"
            
            # Calculate text size
            if hasattr(font, 'getbbox'):
                bbox_text = font.getbbox(label)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                # Fallback for older PIL versions
                text_width, text_height = font.getsize(label)
            
            # Draw background for text
            text_bg = [bbox[0], bbox[1] - text_height - 5, 
                      bbox[0] + text_width + 10, bbox[1]]
            draw.rectangle(text_bg, fill=color)
            
            # Draw text
            draw.text((bbox[0] + 5, bbox[1] - text_height - 5), label, fill=(255, 255, 255), font=font)
    
    # Draw metrics if provided
    if metrics:
        metrics_text = [
            f"Precision: {metrics['precision']:.3f}",
            f"Recall: {metrics['recall']:.3f}",
            f"F1-Score: {metrics['f1_score']:.3f}",
            f"Mean IoU: {metrics['mean_iou']:.3f}",
            f"TP: {metrics['true_positives']} FP: {metrics['false_positives']} FN: {metrics['false_negatives']}"
        ]
        
        # Draw background for text
        text_height_total = len(metrics_text) * 30
        draw.rectangle([10, 10, 350, 10 + text_height_total], fill=(0, 0, 0, 180))
        
        # Draw metrics text with larger font
        for i, text in enumerate(metrics_text):
            draw.text((15, 15 + i * 30), text, fill=(255, 255, 255), font=font)
    
    return img

def save_results_with_metrics(image, predictions, metrics, output_path):
    """Save image with drawn predictions and metrics text"""
    try:
        # Convert predictions to the format expected by RexOmniVisualize
        vis_predictions = []
        for pred in predictions:
            vis_pred = pred.copy()
            # Rename 'coords' to 'bbox' if needed
            if 'coords' in vis_pred and 'bbox' not in vis_pred:
                vis_pred['bbox'] = vis_pred['coords']
                # Remove coords to avoid confusion
                if 'coords' in vis_pred:
                    del vis_pred['coords']
            # Ensure we have required fields
            if 'score' not in vis_pred:
                vis_pred['score'] = 1.0
            vis_predictions.append(vis_pred)
        
        # Create visualization using RexOmniVisualize with better visibility
        vis_image = RexOmniVisualize(
            image=image,
            predictions=vis_predictions,
            font_size=24,  # Larger font
            draw_width=5,  # Thicker lines
            show_labels=True,
        )
        
        # Convert to PIL Image for drawing text
        result_img = Image.fromarray(np.array(vis_image))
        
        # Add metrics text with better visibility
        draw = ImageDraw.Draw(result_img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
        
        metrics_text = [
            f"Precision: {metrics['precision']:.3f}",
            f"Recall: {metrics['recall']:.3f}",
            f"F1-Score: {metrics['f1_score']:.3f}",
            f"Mean IoU: {metrics['mean_iou']:.3f}",
            f"TP: {metrics['true_positives']} FP: {metrics['false_positives']} FN: {metrics['false_negatives']}"
        ]
        
        # Draw background for text
        text_height = len(metrics_text) * 30
        draw.rectangle([10, 10, 350, 10 + text_height], fill=(0, 0, 0, 180))
        
        # Draw metrics text
        for i, text in enumerate(metrics_text):
            draw.text((15, 15 + i * 30), text, fill=(255, 255, 255), font=font)
                
    except Exception as e:
        # Fallback to manual drawing
        result_img = draw_predictions_manual(image, predictions, metrics)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_img.save(output_path)
    return result_img

def format_predictions(predictions):
    """Format predictions from dictionary to list format"""
    # If predictions is a dictionary, convert it to list format
    if isinstance(predictions, dict):
        formatted_predictions = []
        for category, items in predictions.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        # Ensure the item has category and coords
                        item_copy = item.copy()
                        item_copy['category'] = category
                        # Add default score if missing
                        if 'score' not in item_copy:
                            item_copy['score'] = 1.0
                        formatted_predictions.append(item_copy)
        return formatted_predictions
    
    # If it's already a list, just return it
    elif isinstance(predictions, list):
        return predictions
    
    else:
        return []

def filter_predictions(predictions, confidence_threshold=0.5):
    """Filter predictions by confidence score"""
    filtered = []
    for pred in predictions:
        score = pred.get('score', 1.0)
        if score >= confidence_threshold:
            filtered.append(pred)
    return filtered

def process_single_image(model, image_path, output_dir, categories, confidence_threshold=0.5):
    """Process single image and calculate metrics"""
    try:
        # Skip checkpoint files
        if 'checkpoint' in image_path or '.ipynb_checkpoints' in image_path:
            return None
            
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image_resized, scale_factor = resize_image(image, target_size=512)
        
        # Perform inference on resized image
        results = model.inference(images=image_resized, task="detection", categories=categories)
        
        if not results[0]["success"]:
            return None
        
        predictions = results[0]["extracted_predictions"]
        
        # Format predictions to proper format
        formatted_predictions = format_predictions(predictions)
        
        # Filter predictions by confidence
        filtered_predictions = filter_predictions(formatted_predictions, confidence_threshold)
        
        # Scale bounding boxes back to original image size
        for pred in filtered_predictions:
            # Check if we have coords or bbox
            coords = pred.get('coords', [])
            if not coords:
                coords = pred.get('bbox', [])
                
            if coords and len(coords) == 4:
                pred['bbox'] = [
                    coords[0] / scale_factor,
                    coords[1] / scale_factor,
                    coords[2] / scale_factor,
                    coords[3] / scale_factor
                ]
        
        # Find corresponding label file
        image_name = Path(image_path).stem
        label_path = find_label_file(image_path)
        
        if label_path and os.path.exists(label_path):
            ground_truths = load_yolo_annotations(label_path, original_size[0], original_size[1])
            metrics = calculate_classification_metrics(filtered_predictions, ground_truths)
            
            # Save visualization (using original image)
            output_path = os.path.join(output_dir, f"{image_name}_result.jpg")
            save_results_with_metrics(image, filtered_predictions, metrics, output_path)
            
            return {
                'image_name': image_name,
                'metrics': metrics,
                'predictions': filtered_predictions,
                'ground_truths': ground_truths
            }
        else:
            # Still save predictions even without ground truth
            output_path = os.path.join(output_dir, f"{image_name}_result.jpg")
            dummy_metrics = {
                'true_positives': 0,
                'false_positives': len(filtered_predictions),
                'false_negatives': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'mean_iou': 0,
                'ious': []
            }
            save_results_with_metrics(image, filtered_predictions, dummy_metrics, output_path)
            return None
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def find_label_file(image_path):
    """Find corresponding label file for an image"""
    image_path = Path(image_path)
    
    # Skip checkpoint files
    if 'checkpoint' in str(image_path) or '.ipynb_checkpoints' in str(image_path):
        return None
    
    # Try different possible locations for label files
    possible_paths = [
        # If labels are in parallel 'labels' folder
        image_path.parent.parent / 'labels' / f"{image_path.stem}.txt",
        # If labels are in same folder but with different extension
        image_path.with_suffix('.txt'),
        # If labels are in 'labels' subfolder
        image_path.parent / 'labels' / f"{image_path.stem}.txt",
        # If in train/val/test structure
        image_path.parent.parent / 'labels' / image_path.parent.name / f"{image_path.stem}.txt",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def find_images_and_labels(dataset_path, split='test'):
    """Find all images and their corresponding label files in dataset for specific split"""
    dataset_path = Path(dataset_path)
    
    # Look for images in the specified split
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    
    # Priority: check split-specific folders first
    split_images_dir = dataset_path / split / 'images'
    if split_images_dir.exists():
        for ext in image_extensions:
            image_paths.extend(split_images_dir.glob(f'**/{ext}'))
            image_paths.extend(split_images_dir.glob(f'**/{ext.upper()}'))
    
    # Fallback: check general images folder
    if not image_paths:
        images_dir = dataset_path / 'images'
        if images_dir.exists():
            for ext in image_extensions:
                image_paths.extend(images_dir.glob(f'**/{ext}'))
                image_paths.extend(images_dir.glob(f'**/{ext.upper()}'))
    
    # Filter to only include images that have corresponding label files and exclude checkpoints
    valid_pairs = []
    for img_path in image_paths:
        # Skip checkpoint files
        if 'checkpoint' in str(img_path) or '.ipynb_checkpoints' in str(img_path):
            continue
            
        label_path = find_label_file(img_path)
        if label_path and os.path.exists(label_path):
            valid_pairs.append((str(img_path), label_path))
    
    return valid_pairs

def main():
    # Model path
    model_path = "IDEA-Research/Rex-Omni"
    
    # Create wrapper without sampling parameters to avoid warnings
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.0,
    )
    
    # Your categories
    categories = ['occupied-parking-spots', 'parking spot']
    
    # Dataset path and configuration
    dataset_path = "parking-space-finder-1"
    split = 'test'  # Change to 'val' if you want to use validation set
    output_dir = f"detection_results_{split}"
    confidence_threshold = 0.5
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image-label pairs for the specified split
    image_label_pairs = find_images_and_labels(dataset_path, split)
    
    if not image_label_pairs:
        print(f"No images with corresponding labels found in {dataset_path}/{split}")
        print("Please check the dataset path and structure")
        return
    
    print(f"Found {len(image_label_pairs)} images with labels in {split} set")
    
    all_results = []
    errors = []
    
    # Process each image with tqdm progress bar
    for image_path, label_path in tqdm(image_label_pairs, desc=f"Processing {split} images"):
        try:
            result = process_single_image(rex_model, image_path, output_dir, categories, confidence_threshold)
            if result:
                all_results.append(result)
        except Exception as e:
            errors.append(f"{os.path.basename(image_path)}: {str(e)}")
    
    # Print errors if any
    if errors:
        print(f"\nEncountered {len(errors)} errors:")
        for error in errors[:10]:  # Print first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    # Calculate overall metrics
    if all_results:
        overall_metrics = calculate_overall_metrics(all_results)
        
        print("\n" + "="*50)
        print(f"OVERALL RESULTS FOR {split.upper()} SET")
        print("="*50)
        print(f"Total images processed: {overall_metrics['total_images']}")
        print(f"Overall Precision: {overall_metrics['overall_precision']:.3f}")
        print(f"Overall Recall: {overall_metrics['overall_recall']:.3f}")
        print(f"Overall F1-Score: {overall_metrics['overall_f1_score']:.3f}")
        print(f"Overall Mean IoU: {overall_metrics['overall_mean_iou']:.3f}")
        print(f"Total TP: {overall_metrics['total_tp']}, FP: {overall_metrics['total_fp']}, FN: {overall_metrics['total_fn']}")
        
        # Save overall results
        results_output = {
            'overall_metrics': overall_metrics,
            'per_image_results': all_results,
            'confidence_threshold': confidence_threshold,
            'split': split
        }
        
        with open(os.path.join(output_dir, "overall_metrics.json"), 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_types(x) for x in obj]
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                else:
                    return obj
            
            json.dump(convert_types(results_output), f, indent=2)
        
        print(f"\nAll results saved to: {output_dir}")
    else:
        print("No results to process")

def calculate_overall_metrics(all_results):
    """Calculate overall metrics from all individual image results"""
    total_tp = sum(result['metrics']['true_positives'] for result in all_results)
    total_fp = sum(result['metrics']['false_positives'] for result in all_results)
    total_fn = sum(result['metrics']['false_negatives'] for result in all_results)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Collect all IoUs
    all_ious = []
    for result in all_results:
        all_ious.extend(result['metrics']['ious'])
    mean_iou = np.mean(all_ious) if all_ious else 0
    
    return {
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1_score': f1_score,
        'overall_mean_iou': mean_iou,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'total_images': len(all_results)
    }

if __name__ == "__main__":
    main()

    # categories = ['occupied-parking-spots', 'parking spot']
    
    # # Dataset path - replace with your actual dataset path
    # dataset_path = "parking-space-finder-1/test/"
    # output_dir = "/home/efimenko.aleksey7/rex/Rex-Omni/detection_results/"