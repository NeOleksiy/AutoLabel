import sys
import numpy as np
from PIL import Image
import torch
from pathlib import Path
from rex_omni import RexOmniWrapper
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional
import warnings
import logging
import time
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autolabel.log')
    ]
)
logger = logging.getLogger(__name__)


class AutoLabel:
    def __init__(self, 
                 model_params: Dict[str, Any],
                 task: str = "detection",
                 classes_for_similar_prompting: List[str] = None,
                 class_names: List[str] = None,
                 images_path: str = None):
        
        logger.info(f"Initializing AutoLabel with task: {task}, {len(class_names or [])} classes")
        
        self.model_params = model_params
        self.task = task
        self.classes_for_similar_prompting = classes_for_similar_prompting or []
        self.class_names = class_names or []
        self.images_path = Path(images_path) if images_path else None
        
        self.rex = None
        self.owlvit_processor = None
        self.owlvit_model = None
        self.owlvit_available = False
        
        self.text_alignment_cache = {}
        
        self._init_models()
        
        self.stats = {
            'total_images': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_predictions': 0,
            'total_filtered_predictions': 0,
            'visual_prompting_predictions': 0,
            'nms_filtered': 0
        }
        
        self.raw_predictions = {}
        self.filtered_predictions = {}
    
    def _init_models(self):
        logger.info("Initializing models...")
        try:
            logger.info(f"Initializing RexOmni with params: {self.model_params}")
            self.rex = RexOmniWrapper(
                model_path="IDEA-Research/Rex-Omni",
                backend="transformers",
                max_tokens=self.model_params.get('max_tokens', 1024),
                temperature=self.model_params.get('temperature', 0.75),
                top_p=self.model_params.get('top_p', 0.7),
                top_k=self.model_params.get('top_k', 10),
                repetition_penalty=self.model_params.get('repetition_penalty', 1),
            )
            logger.info("RexOmni initialized successfully")
            
            try:
                logger.info("Loading OwlViT model...")
                self.owlvit_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
                self.owlvit_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
                self.owlvit_model.eval()
                if torch.cuda.is_available():
                    self.owlvit_model.to('cuda')
                    logger.info("OwlViT moved to CUDA")
                self.owlvit_available = True
                logger.info("OwlViT initialized successfully")
            except Exception as e:
                logger.warning(f"Could not load OwlViT model: {e}")
                self.owlvit_available = False
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise RuntimeError(f"Error initializing models: {e}")
    

    def safe_get_predictions(self, results):
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
    

    def calculate_iou(self, boxes1, boxes2):
        x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        inter_x1 = np.maximum(x11[:, np.newaxis], x21)
        inter_y1 = np.maximum(y11[:, np.newaxis], y21)
        inter_x2 = np.minimum(x12[:, np.newaxis], x22)
        inter_y2 = np.minimum(y12[:, np.newaxis], y22)
        
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1[:, np.newaxis] + area2 - inter_area
        
        return np.where(union_area > 0, inter_area / union_area, 0)
    

    def get_visual_prompts_from_predictions(self, predictions, max_prompts_per_class=2):
        if not predictions or not self.classes_for_similar_prompting:
            return []
        
        predictions_by_category = {}
        for pred in predictions:
            if 'category' not in pred or 'coords' not in pred:
                continue
            category = pred['category']
            if category in self.classes_for_similar_prompting:
                if category not in predictions_by_category:
                    predictions_by_category[category] = []
                predictions_by_category[category].append(pred)
        
        visual_prompts = []
        for category, cat_preds in predictions_by_category.items():
            sorted_preds = sorted(cat_preds, key=lambda x: x.get('score', 1.0), reverse=True)
            
            for pred in sorted_preds[:max_prompts_per_class]:
                visual_prompts.append({
                    'coords': pred['coords'],
                    'category': category,
                    'source': 'visual_prompting'
                })
        
        return visual_prompts
    

    def run_visual_prompting(self, image, visual_prompts):
        if not visual_prompts:
            return []
        
        prompt_boxes = [prompt['coords'] for prompt in visual_prompts]
        results = self.rex.inference(
            images=image,
            task="visual_prompting",
            visual_prompt_boxes=prompt_boxes,
        )
        
        predictions = self.safe_get_predictions(results)
        
        for pred in predictions:
            pred['source'] = 'visual_prompting'
        
        self.stats['visual_prompting_predictions'] += len(predictions)
        
        return predictions
            

    
    def check_text_alignment_batch(self, image, bboxes, text_queries):
        if not self.owlvit_available or not bboxes:
            return [1.0] * len(bboxes)
        
        cropped_images = []
        valid_indices = []
        
        for i, (bbox, text_query) in enumerate(zip(bboxes, text_queries)):

            cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            cropped_images.append(cropped_img)
            valid_indices.append(i)

        
        if not cropped_images:
            return [0.0] * len(bboxes)
        
        cache_key = f"{image.size}_{hash(str(bboxes))}_{hash(str(text_queries))}"
        if cache_key in self.text_alignment_cache:
            logger.info(f"Using cached text alignment results for {len(bboxes)} boxes")
            return self.text_alignment_cache[cache_key]
        
        # Обработка батча
        inputs = self.owlvit_processor(
            text=text_queries[:len(cropped_images)], 
            images=cropped_images, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.owlvit_model(**inputs)
        
        # Извлекаем логиты
        scores = []
        for i in range(len(cropped_images)):
            logits = torch.max(outputs.logits[i])
            score = torch.sigmoid(logits).item()
            scores.append(score)
        
        # Собираем результаты для всех боксов
        final_scores = [0.0] * len(bboxes)
        for idx, score in zip(valid_indices, scores):
            final_scores[idx] = score
        
        # Кэшируем результаты
        self.text_alignment_cache[cache_key] = final_scores
        logger.info(f"Text alignment checked for {len(bboxes)} boxes")
        
        return final_scores
    

    def apply_iou_threshold(self, predictions, iou_threshold=0.9):
        if not predictions:
            return []
        
        # Группируем предсказания по категориям
        predictions_by_category = {}
        for pred in predictions:
            if 'category' not in pred:
                continue
            category = pred['category']
            if category not in predictions_by_category:
                predictions_by_category[category] = []
            predictions_by_category[category].append(pred)
        
        filtered_predictions = []
        
        for category, cat_preds in predictions_by_category.items():
            if not cat_preds:
                continue
                
            boxes = []
            scores = []
            pred_indices = []
            
            for idx, pred in enumerate(cat_preds):
                if 'coords' in pred:
                    boxes.append(pred['coords'])
                    scores.append(pred.get('score', 1.0))
                    pred_indices.append(idx)
            
            if not boxes:
                filtered_predictions.extend(cat_preds)
                continue
            
            boxes_np = np.array(boxes)
            scores_np = np.array(scores)
            
            # Сортируем по confidence
            sorted_indices = np.argsort(-scores_np)
            boxes_sorted = boxes_np[sorted_indices]
            
            # Векторизованный NMS
            keep = []
            while boxes_sorted.shape[0] > 0:
                # Берем бокс с наибольшим confidence
                keep.append(sorted_indices[0])
                
                if boxes_sorted.shape[0] == 1:
                    break
                
                ious = self.calculate_iou(boxes_sorted[0:1], boxes_sorted[1:])
                
                mask = ious[0] < iou_threshold
                boxes_sorted = boxes_sorted[1:][mask]
                sorted_indices = sorted_indices[1:][mask]
            
            for idx in keep:
                filtered_predictions.append(cat_preds[idx])
            
            self.stats['nms_filtered'] += (len(cat_preds) - len(keep))
        
        return filtered_predictions
    

    def inference(self, 
                  generate_visual_prompting: bool = False,
                  images_path: str = None):
        logger.info(f"Starting inference with visual prompting: {generate_visual_prompting}")
        
        if images_path:
            self.images_path = Path(images_path)
        
        if not self.rex: 
            self._init_models()
        
        if not self.images_path or not self.images_path.exists():
            raise ValueError(f"Images path {self.images_path} does not exist")
        
        logger.info(f"Scanning images in {self.images_path}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.images_path.glob(f"*{ext}")))
            image_files.extend(list(self.images_path.glob(f"*{ext.upper()}")))
        
        
        logger.info(f"Found {len(image_files)} images, processing first 30")
        
        self.raw_predictions = {}
        self.filtered_predictions = {}
        self.stats = {
            'total_images': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_predictions': 0,
            'total_filtered_predictions': 0,
            'visual_prompting_predictions': 0,
            'nms_filtered': 0
        }
        

        image_files = image_files
        

        
        for i, image_path in enumerate(image_files):
            try:

                
                logger.info(f"Processing image {i+1}/{len(image_files)}: {image_path.name}")
                
                image = Image.open(image_path).convert("RGB")
                
                logger.debug(f"Running detection on {image_path.name}")
                results = self.rex.inference(
                    images=image, 
                    task=self.task, 
                    categories=self.class_names
                )
                
                standard_predictions = self.safe_get_predictions(results)
                box_predictions = [pred for pred in standard_predictions 
                                  if isinstance(pred, dict) and pred.get("type") == "box"]
                
                logger.info(f"Found {len(box_predictions)} box predictions in {image_path.name}")
                
                for pred in box_predictions:
                    pred['source'] = 'detection'
                
                all_predictions = box_predictions.copy()
                
                if generate_visual_prompting and box_predictions:
                    logger.debug(f"Running visual prompting on {image_path.name}")

                    visual_prompts = self.get_visual_prompts_from_predictions(box_predictions, max_prompts_per_class=2)
                    
                    visual_prompting_predictions = self.run_visual_prompting(image, visual_prompts)
                    
                    visual_box_predictions = [pred for pred in visual_prompting_predictions 
                                            if isinstance(pred, dict) and pred.get("type") == "box"]
                    
                    logger.info(f"Visual prompting added {len(visual_box_predictions)} predictions")
                    
                    all_predictions.extend(visual_box_predictions)
                
                self.raw_predictions[image_path.name] = {
                    'image': image,
                    'predictions': all_predictions,
                    'path': str(image_path)
                }
                
                self.stats['successful_inferences'] += 1
                self.stats['total_predictions'] += len(all_predictions)
                
                logger.info(f"Successfully processed {image_path.name}: {len(all_predictions)} predictions")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                self.stats['failed_inferences'] += 1
                continue
            
            self.stats['total_images'] += 1
        
        
        logger.info(f"Inference completed: {self.stats['successful_inferences']} successful, {self.stats['failed_inferences']} failed")
        logger.info(f"Total predictions: {self.stats['total_predictions']}")
        
        return True
    
    def filter(self, 
               text_threshold: Optional[float] = None,
               use_similar_prompting: bool = True,
               iou_threshold: float = 0.9,
               max_lower_bound: float = 0.9,
               min_lower_bound: float = 0.1):

        start_time = time.time()
        logger.info(f"Starting filter with params: text_threshold={text_threshold}, "
                   f"use_similar_prompting={use_similar_prompting}, iou_threshold={iou_threshold}, "
                   f"bounds=[{min_lower_bound}, {max_lower_bound}]")
        
        self.filtered_predictions = {}
        self.stats['total_filtered_predictions'] = 0
        self.stats['nms_filtered'] = 0
        
        # Очищаем кэш при изменении text_threshold
        if text_threshold != getattr(self, '_last_text_threshold', None):
            self.text_alignment_cache = {}
            self._last_text_threshold = text_threshold
        
        for img_name, data in self.raw_predictions.items():
            image = data['image']
            predictions = data['predictions'].copy()
            
            filter_start = time.time()
            if not use_similar_prompting:
                predictions = [p for p in predictions if p.get('source') != 'visual_prompting']
            logger.debug(f"Source filtering took {time.time() - filter_start:.3f}s")
            
            nms_start = time.time()
            nms_filtered = self.apply_iou_threshold(predictions, iou_threshold)
            logger.debug(f"NMS took {time.time() - nms_start:.3f}s for {len(predictions)} predictions")
            
            if text_threshold is not None and text_threshold > 0 and self.owlvit_available and nms_filtered:
                text_start = time.time()
                
                bboxes = []
                text_queries = []
                pred_indices = []
                
                for idx, pred in enumerate(nms_filtered):
                    if 'coords' in pred and 'category' in pred:
                        bboxes.append(pred['coords'])
                        text_queries.append(pred['category'])
                        pred_indices.append(idx)
                
                if bboxes:
                    text_scores = self.check_text_alignment_batch(image, bboxes, text_queries)
                    
                    text_filtered = []
                    for idx, pred in enumerate(nms_filtered):
                        if idx in pred_indices:
                            score_idx = pred_indices.index(idx)
                            score = text_scores[score_idx]
                            pred['text_score'] = score
                            if score >= text_threshold:
                                text_filtered.append(pred)
                        else:
                            text_filtered.append(pred)
                    
                    filtered = text_filtered
                    logger.debug(f"Text filtering took {time.time() - text_start:.3f}s for {len(bboxes)} boxes")
                else:
                    filtered = nms_filtered
            else:
                filtered = nms_filtered
            
            if filtered and (max_lower_bound < 1.0 or min_lower_bound > 0):
                size_start = time.time()
                
                widths = []
                heights = []
                valid_indices = []
                
                for idx, pred in enumerate(filtered):
                    if 'coords' in pred:
                        w = pred['coords'][2] - pred['coords'][0]
                        h = pred['coords'][3] - pred['coords'][1]
                        widths.append(w)
                        heights.append(h)
                        valid_indices.append(idx)
                
                if widths and heights:
                    widths_np = np.array(widths)
                    heights_np = np.array(heights)
                    
                    avg_width = np.mean(widths_np)
                    avg_height = np.mean(heights_np)
                    
                    width_ratios = widths_np / avg_width if avg_width > 0 else np.ones_like(widths_np)
                    height_ratios = heights_np / avg_height if avg_height > 0 else np.ones_like(heights_np)
                    
                    size_mask = (min_lower_bound <= width_ratios) & (width_ratios <= max_lower_bound) & \
                               (min_lower_bound <= height_ratios) & (height_ratios <= max_lower_bound)
                    
                    size_filtered = []
                    for idx, pred in enumerate(filtered):
                        if idx in valid_indices:
                            mask_idx = valid_indices.index(idx)
                            if size_mask[mask_idx]:
                                size_filtered.append(pred)
                        else:
                            size_filtered.append(pred)
                    
                    filtered = size_filtered
                    
                logger.debug(f"Size filtering took {time.time() - size_start:.3f}s")
            
            self.filtered_predictions[img_name] = {
                'image': image,
                'predictions': filtered,
                'path': data['path']
            }
            
            self.stats['total_filtered_predictions'] += len(filtered)
        
        total_time = time.time() - start_time
        logger.info(f"Filter completed in {total_time:.2f}s: "
                   f"{self.stats['total_filtered_predictions']} predictions after filtering")
        
        return True
    
    def get_image_with_bboxes(self, img_name, show_filtered=True):
        if show_filtered:
            data = self.filtered_predictions.get(img_name)
        else:
            data = self.raw_predictions.get(img_name)
    
        
        image = data['image']
        predictions = data['predictions']
        
        logger.debug(f"Creating visualization for {img_name} with {len(predictions)} predictions")
        
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(image)
        
        # Цвета для разных source
        source_colors = {
            'detection': 'red',
            'visual_prompting': 'blue'
        }
        
        for pred in predictions:
            if 'coords' in pred:
                coords = pred['coords']
                category = pred.get('category', 'Unknown')
                score = pred.get('score', 1.0)
                text_score = pred.get('text_score', None)
                source = pred.get('source', 'detection')
                
                color = source_colors.get(source, 'green')
                
                rect = patches.Rectangle(
                    (coords[0], coords[1]), 
                    coords[2] - coords[0], 
                    coords[3] - coords[1],
                    linewidth=2, 
                    edgecolor=color, 
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                label = f"{category}: {score:.2f}"
                if text_score is not None:
                    label += f" (text: {text_score:.2f})"
                if source != 'detection':
                    label += f" [{source}]"
                
                ax.text(
                    coords[0], 
                    coords[1] - 5, 
                    label,
                    bbox=dict(facecolor='yellow', alpha=0.7),
                    fontsize=9,
                    color='black'
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig