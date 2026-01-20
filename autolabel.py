import numpy as np
from PIL import Image
import torch
from pathlib import Path
from rex_omni import RexOmniWrapper
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Union
from utils.schema import TaskConfig
from utils.logger import setup_logger
import warnings
import time
warnings.filterwarnings('ignore')


logger = setup_logger('autolabel', 'autolabel.log')




class AutoLabel:
    def __init__(self, 
                 model_params: Dict[str, Any],
                 task: str = "detection",
                 classes_for_similar_prompting: List[str] = None,
                 class_names: List[str] = None,
                 images_path: str = None,
                 system_prompt: str = None,
                 task_config: TaskConfig = None):
        
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
        self.task = None
        
        self._init_models(system_prompt, task_config)
        
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
    
    def _init_models(self, system_prompt: str = None, task_config: TaskConfig = None):
        logger.info("Initializing models...")
        try:
            logger.info(f"Initializing RexOmni with params: {self.model_params}")
            rex_params = {
                'model_path': "IDEA-Research/Rex-Omni",
                'backend': "transformers",
                'max_tokens': self.model_params.get('max_tokens', 1024),
                'temperature': self.model_params.get('temperature', 0.75),
                'top_p': self.model_params.get('top_p', 0.7),
                'top_k': self.model_params.get('top_k', 10),
                'repetition_penalty': self.model_params.get('repetition_penalty', 1),
            }
            
            if system_prompt:
                rex_params['system_prompt'] = system_prompt
            if task_config:
                rex_params['task_config'] = task_config
            
            self.rex = RexOmniWrapper(**rex_params)
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
        
        for pred, visual_prompt in zip(predictions, visual_prompts):
            pred['source'] = 'visual_prompting'
            if 'category' not in pred:
                pred['category'] = visual_prompt['category']
        
        self.stats['visual_prompting_predictions'] += len(predictions)
        
        return predictions
    
    def react(self, 
            image_name: str, 
            predictions: List[Dict[str, Any]], 
            iou_threshold: float = 0.9,
            n_repeats: int = 2) -> List[Dict[str, Any]]:

        logger.info(f"Starting REACT method for {image_name}, n_repeats={n_repeats}")
        
        # Получаем изображение
        if image_name not in self.raw_predictions:
            logger.error(f"Image {image_name} not found in raw_predictions")
            return predictions
        
        image_data = self.raw_predictions[image_name]
        image = image_data['image']
        
        # Копируем текущие предсказания
        current_predictions = predictions.copy()
        
        # Собираем информацию о уже найденных объектах для промта
        found_objects_by_category = {}
        for pred in current_predictions:
            if 'category' in pred and 'coords' in pred:
                category = pred['category']
                if category not in found_objects_by_category:
                    found_objects_by_category[category] = []
                # Сохраняем упрощенную информацию о боксе
                coords = pred['coords']
                found_objects_by_category[category].append({
                    'bbox': f"[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]",
                    'area': (coords[2] - coords[0]) * (coords[3] - coords[1])
                })
        
        for repeat_idx in range(n_repeats):
            logger.info(f"REACT iteration {repeat_idx + 1}/{n_repeats}")
            
            # Создаем промт с информацией об уже найденных объектах
            base_prompt = "You are a precise object detection assistant. "
            
            if found_objects_by_category:
                base_prompt += "I have already detected the following objects:\n"
                
                for category, objects in found_objects_by_category.items():
                    base_prompt += f"- {category}: {len(objects)} objects "
                    if objects:
                        # Вычисляем средний размер для этой категории
                        avg_area = sum(obj['area'] for obj in objects) / len(objects)
                        base_prompt += f"(average size: {avg_area:.0f} pixels^2)\n"
                    else:
                        base_prompt += "\n"
                
                base_prompt += "\nNow, please find NEW objects that I might have missed. "
                base_prompt += "Focus on:\n"
                base_prompt += "1. Smaller objects that could be overlooked\n"
                base_prompt += "2. Objects partially occluded or at image edges\n"
                base_prompt += "3. Objects with unusual poses or orientations\n"
                base_prompt += "4. Objects that are similar to found ones but in different locations\n"
                base_prompt += "\nDo NOT repeat detections of already found objects. "
                base_prompt += "If you're unsure whether something is new, check if it overlaps significantly with existing detections."
            else:
                base_prompt += "Please detect objects in the image carefully."
            
            # Добавляем инструкцию по формату вывода
            base_prompt += "\n\nOutput format: List of objects with categories and [x0, y0, x1, y1] bounding boxes."
            
            # Создаем временный task_config с обновленным промтом
            from utils.schema import TaskConfig
            
            react_task_config = TaskConfig(
                name="REACT Detection",
                prompt_template=base_prompt,
                description="Find new objects that were missed in previous passes",
                output_format="boxes",
                requires_categories=False,
            )
            
            # Запускаем inference с новым промтом
            try:
                # Подготовка параметров для inference
                inference_params = {
                    'images': image,
                    'task': "detection",
                    'task_config': react_task_config,
                }
                
                # Передаем категории, если они заданы
                if self.class_names:
                    inference_params['categories'] = self.class_names
                
                # Выполняем inference
                results = self.rex.inference(**inference_params)
                
                # Извлекаем предсказания
                new_predictions = self.safe_get_predictions(results)
                new_box_predictions = [pred for pred in new_predictions 
                                      if isinstance(pred, dict) and pred.get("type") == "box"]
                
                logger.info(f"REACT iteration {repeat_idx + 1} found {len(new_box_predictions)} new predictions")
                
                # Добавляем метку о источнике
                for pred in new_box_predictions:
                    pred['source'] = f'react_{repeat_idx + 1}'
                
                # Добавляем новые предсказания к текущим
                current_predictions.extend(new_box_predictions)
                
                # Обновляем информацию о найденных объектах для следующей итерации
                for pred in new_box_predictions:
                    if 'category' in pred and 'coords' in pred:
                        category = pred['category']
                        coords = pred['coords']
                        if category not in found_objects_by_category:
                            found_objects_by_category[category] = []
                        found_objects_by_category[category].append({
                            'bbox': f"[{coords[0]:.0f}, {coords[1]:.0f}, {coords[2]:.0f}, {coords[3]:.0f}]",
                            'area': (coords[2] - coords[0]) * (coords[3] - coords[1])
                        })
                
                # Применяем NMS для удаления дубликатов
                current_predictions = self.apply_iou_threshold(current_predictions, iou_threshold)
                
                logger.info(f"After REACT iteration {repeat_idx + 1}: {len(current_predictions)} total predictions")
                
            except Exception as e:
                logger.error(f"Error in REACT iteration {repeat_idx + 1}: {e}")
                break
        
        # Обновляем статистику
        self.stats['total_predictions'] += (len(current_predictions) - len(predictions))
        
        logger.info(f"REACT completed. Found {len(current_predictions) - len(predictions)} additional predictions")
        
        return current_predictions
            

    
    def check_text_alignment_batch(self, image, bboxes, text_queries):
        if not self.owlvit_available or not bboxes:
            return [1.0] * len(bboxes)
        
        cropped_images = []
        valid_indices = []
        
        # Убеждаемся, что image - это PIL.Image, а не строка
        if isinstance(image, str):
            logger.warning(f"Image is a string, trying to load: {image}")
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                logger.error(f"Could not load image from string: {e}")
                return [0.0] * len(bboxes)
        
        for i, (bbox, text_query) in enumerate(zip(bboxes, text_queries)):
            try:
                cropped_img = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                cropped_images.append(cropped_img)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Could not crop bbox {bbox}: {e}")
                # Пропускаем этот bbox
                continue
        
        if not cropped_images:
            return [0.0] * len(bboxes)
        
        # Создаем cache key
        try:
            cache_key = f"{image.size}_{hash(str(bboxes))}_{hash(str(text_queries))}"
        except Exception as e:
            logger.warning(f"Could not create cache key: {e}")
            cache_key = f"{hash(str(bboxes))}_{hash(str(text_queries))}"
        
        if cache_key in self.text_alignment_cache:
            logger.info(f"Using cached text alignment results for {len(bboxes)} boxes")
            return self.text_alignment_cache[cache_key]
        
        # Обработка батча
        try:
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
                try:
                    logits = torch.max(outputs.logits[i])
                    score = torch.sigmoid(logits).item()
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Could not get score for image {i}: {e}")
                    scores.append(0.0)
        except Exception as e:
            logger.error(f"Error in OwlViT inference: {e}")
            return [0.0] * len(bboxes)
        
        # Собираем результаты для всех боксов
        final_scores = [0.0] * len(bboxes)
        for idx, score in zip(valid_indices, scores):
            if idx < len(final_scores):
                final_scores[idx] = score
        
        # Кэшируем результаты
        self.text_alignment_cache[cache_key] = final_scores
        logger.info(f"Text alignment checked for {len(cropped_images)} boxes")
        
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
                        image: Union[str, Path, Image.Image],
                        generate_visual_prompting: bool = False,
                        task_config: Optional[TaskConfig] = None) -> Dict[str, Any]:

        logger.info(f"Starting inference for single image with visual prompting: {generate_visual_prompting}")
        
        try:
            # Загрузка изображения если передан путь
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                logger.info(f"Loading image: {image_path.name}")
                pil_image = Image.open(image_path).convert("RGB")
                image_name = image_path.name
            else:
                pil_image = image
                image_name = f"image_{int(time.time())}"
            
            logger.debug(f"Running detection on {image_name}")
            
            # Подготовка параметров для inference
            inference_params = {
                'images': pil_image,
                'task': "detection",
            }
            
            if task_config and not task_config.requires_categories:
                pass
            elif self.class_names:
                # Передаем категории отдельно
                inference_params['categories'] = self.class_names
            
            
            results = self.rex.inference(**inference_params)
            print(results)
            
            standard_predictions = self.safe_get_predictions(results)
            box_predictions = [pred for pred in standard_predictions 
                              if isinstance(pred, dict) and pred.get("type") == "box"]
            
            logger.info(f"Found {len(box_predictions)} box predictions in {image_name}")
            
            for pred in box_predictions:
                pred['source'] = 'detection'
            
            all_predictions = box_predictions.copy()
            
            if generate_visual_prompting and box_predictions:
                logger.debug(f"Running visual prompting on {image_name}")

                visual_prompts = self.get_visual_prompts_from_predictions(box_predictions, max_prompts_per_class=2)
                
                visual_prompting_predictions = self.run_visual_prompting(pil_image, visual_prompts)
                
                visual_box_predictions = [pred for pred in visual_prompting_predictions 
                                        if isinstance(pred, dict) and pred.get("type") == "box"]
                
                logger.info(f"Visual prompting added {len(visual_box_predictions)} predictions")
                
                all_predictions.extend(visual_box_predictions)
            
            # Сохраняем raw predictions для статистики
            self.raw_predictions[image_name] = {
                'image': pil_image,
                'predictions': all_predictions,
                'path': str(image) if isinstance(image, (str, Path)) else image_name
            }
            
            self.stats['successful_inferences'] += 1
            self.stats['total_predictions'] += len(all_predictions)
            self.stats['total_images'] += 1
            
            logger.info(f"Successfully processed {image_name}: {len(all_predictions)} predictions")
            
            return {
                'image_name': image_name,
                'image': pil_image,
                'predictions': all_predictions,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.stats['failed_inferences'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    
    def filter(self,
                     image_name: str,
                     predictions: List[Dict[str, Any]],
                     text_threshold: Optional[float] = None,
                     use_similar_prompting: bool = True,
                     iou_threshold: float = 0.9,
                     max_lower_bound: float = 0.9,
                     min_lower_bound: float = 0.1) -> List[Dict[str, Any]]:

        logger.info(f"Filtering predictions for {image_name}")
        
        start_time = time.time()
        
        # Очищаем кэш при изменении text_threshold
        if text_threshold != getattr(self, '_last_text_threshold', None):
            self.text_alignment_cache = {}
            self._last_text_threshold = text_threshold
        
        # Получаем изображение
        if image_name in self.raw_predictions:
            image = self.raw_predictions[image_name]['image']
        else:
            logger.warning(f"Image {image_name} not found in raw_predictions")
            return []
        
        # Копируем предсказания
        filtered = predictions.copy()
        
        # Фильтрация по source
        if not use_similar_prompting:
            filtered = [p for p in filtered if p.get('source') != 'visual_prompting']
        
        # Применяем NMS
        filtered = self.apply_iou_threshold(filtered, iou_threshold)
        
        # Фильтрация по text alignment
        if text_threshold is not None and text_threshold > 0 and self.owlvit_available and filtered:
            bboxes = []
            text_queries = []
            pred_indices = []
            
            for idx, pred in enumerate(filtered):
                if 'coords' in pred and 'category' in pred:
                    bboxes.append(pred['coords'])
                    text_queries.append(pred['category'])
                    pred_indices.append(idx)
            
            if bboxes:
                text_scores = self.check_text_alignment_batch(image, bboxes, text_queries)
                
                text_filtered = []
                for idx, pred in enumerate(filtered):
                    if idx in pred_indices:
                        score_idx = pred_indices.index(idx)
                        score = text_scores[score_idx]
                        pred['text_score'] = score
                        if score >= text_threshold:
                            text_filtered.append(pred)
                    else:
                        text_filtered.append(pred)
                
                filtered = text_filtered
        
        # Фильтрация по размеру
        if filtered and (max_lower_bound < 1.0 or min_lower_bound > 0):
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
        
        # Сохраняем отфильтрованные предсказания
        self.filtered_predictions[image_name] = {
            'image': image,
            'predictions': filtered,
            'path': self.raw_predictions[image_name]['path']
        }
        
        self.stats['total_filtered_predictions'] += len(filtered)
        
        logger.info(f"Filtering completed in {time.time() - start_time:.2f}s: "
                   f"{len(filtered)} predictions after filtering")
        
        return filtered
    
    
    def apply_additional_task(self, 
                             image: Union[str, Path, Image.Image],
                             predictions: List[Dict[str, Any]],
                             task_type: str,
                             system_promt: str = "You are a human pose estimation assistant. Detect people and their keypoints accurately.",
                             promt: str = "Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format.") -> Dict[str, Any]:

        logger.info(f"Applying keypoint detection with prompt: {promt}")
        
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        else:
            # Уже объект PIL.Image
            pil_image = image
        
        match task_type:
            case "human_pose":
                from tasks import HumanPoseKeypointTask
                self.task = HumanPoseKeypointTask(logger,system_promt=system_promt,task_promt=promt, model_params=self.model_params)
                logger.info("HumanPoseKeypointTask initialized successfully")
            case "animal_pose":
                from tasks import AnimalPoseKeypointTask
                self.task = AnimalPoseKeypointTask(logger, self.model_params)
                logger.info("AnimalPoseKeypointTask initialized successfully")
            case "face_keypoint":
                from tasks import FaceKeypointTask
                self.task = FaceKeypointTask(logger,system_promt=system_promt,task_promt=promt, model_params=self.model_params)
                logger.info("FaceKeypointTask initialized successfully")
            case _:
                logger.info("Please input existing task type")
                return {
                    'success': False,
                    'keypoint_result': None
                }

        
        bboxes = []
        class_names = []
        
        for pred in predictions:
            if 'coords' in pred and 'category' in pred:
                bboxes.append(pred['coords'])
                class_names.append(pred['category'])
        
        
        logger.info(f"Starting keypoint detection for {len(bboxes)} objects")
        
        result = self.task.mark_up(pil_image, bboxes, class_names[0])
        
        logger.info(f"Keypoint detection completed: {result['success_count']} with keypoints")
        
        return {
            'success': True,
            'keypoint_result': result
        }
            
    
    def get_image_with_bboxes(self, img_name, show_filtered=True, 
                             keypoint_result: Optional[Dict[str, Any]] = None):
        if show_filtered:
            data = self.filtered_predictions.get(img_name)
        else:
            data = self.raw_predictions.get(img_name)
        
        if not data:
            logger.warning(f"No data found for image: {img_name}")
            return None
        
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
        
        # Отображение ключевых точек если переданы
        if keypoint_result and keypoint_result.get('success', False):
            kp_result = keypoint_result.get('keypoint_result', {})
            keypoints_list = kp_result.get('keypoints', [])
            keypoint_names = kp_result.get('keypoint_names', [])
            
            # Скелетные связи (COCO format)
            skeleton = [
                (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Ноги
                (5, 11), (6, 12),  # Тело
                (5, 7), (7, 9), (6, 8), (8, 10),  # Руки
                (1, 2), (1, 3), (2, 4), (3, 5), (4, 6),  # Голова и плечи
                (0, 1), (0, 2)  # Нос-глаза
            ]
            
            colors = plt.cm.hsv(np.linspace(0, 1, len(keypoint_names)))
            
            for person_idx, person_keypoints in enumerate(keypoints_list):
                # Рисуем ключевые точки
                for kp_idx, (kp, color) in enumerate(zip(person_keypoints, colors)):
                    if kp[0] > 0 and kp[1] > 0:  # Проверяем, что точка видима
                        ax.scatter(kp[0], kp[1], s=50, c=[color], marker='o', 
                                 edgecolors='white', linewidths=1)
                        # Подписываем точки
                        if kp_idx < len(keypoint_names):
                            ax.text(kp[0] + 5, kp[1] + 5, keypoint_names[kp_idx], 
                                  fontsize=8, color='white', 
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
                
                # Рисуем скелетные связи
                for connection in skeleton:
                    start_idx, end_idx = connection
                    if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints)):
                        start_kp = person_keypoints[start_idx]
                        end_kp = person_keypoints[end_idx]
                        # Проверяем, что обе точки видимы
                        if start_kp[0] > 0 and start_kp[1] > 0 and end_kp[0] > 0 and end_kp[1] > 0:
                            ax.plot([start_kp[0], end_kp[0]], [start_kp[1], end_kp[1]], 
                                   color='yellow', linewidth=2, alpha=0.7)
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    


if __name__ == "__main__":
    # Параметры модели
    model_params = {
        'max_tokens': 1024,
        'temperature': 0.75,
        'top_p': 0.7,
        'top_k': 10,
        'repetition_penalty': 1
    }
    
    # Классы для детекции (добавим 'person' для тестирования keypoints)
    class_names = ['water','eggs', 'yogurt', 'banana']
    
    # Путь к изображению
    image_path = "/home/efimenko.aleksey7/rex/Rex-Omni/grocery-shopping-images-2/test/images/aaa.jpg"
    
    print(f"\n{'='*60}")
    print(f"Processing image: {image_path}")
    print(f"Classes: {class_names}")
    print(f"{'='*60}\n")
    
    try:
        # Проверяем существование файла
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            print(f"❌ Error: File not found at {image_path}")
            exit(1)
        
        # Загружаем изображение заранее
        print("Step 0: Loading image...")
        pil_image = Image.open(image_path_obj).convert("RGB")
        print(f"   Image loaded: {pil_image.size}")
        
        # Создаем простой task_config
        categories_str = ", ".join(class_names)
        task_config = TaskConfig(
            name="Detection",
            prompt_template=f"Detect {categories_str}. Output the bounding box coordinates in [x0, y0, x1, y1] format.",
            description="",
            output_format="boxes",
            requires_categories=False,
        )
        
        # Создаем экземпляр AutoLabel
        auto_label = AutoLabel(
            model_params=model_params,
            task="detection",
            classes_for_similar_prompting=['cat'],
            class_names=class_names,
            system_prompt="You are a object detection assistant.",
            task_config=task_config
        )
        
        # 1. Выполняем inference для одного изображения
        print("\nStep 1: Running inference...")
        result = auto_label.inference(
            image=pil_image,
            generate_visual_prompting=True
        )
        
        if not result['success']:
            print(f"❌ Inference failed: {result.get('error', 'Unknown error')}")
            exit(1)
        
        image_name = result['image_name']
        predictions = result['predictions']
        
        print(f"✅ Inference successful!")
        print(f"   Found {len(predictions)} predictions")
        

        print("\nPredictions details:")
        for i, pred in enumerate(predictions):
            category = pred.get('category', 'Unknown')
            coords = pred.get('coords', [])
            score = pred.get('score', 0.0)
            source = pred.get('source', 'unknown')
            coords_str = f"[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]" if coords else "[]"
            print(f"   {i+1}. {category:20s} - score: {score:.3f}, source: {source}, bbox: {coords_str}")

        reactults = auto_label.react(
                image_name=image_name,
                predictions=predictions.copy(),
        )

        print("\n React Predictions details:")
        for i, pred in enumerate(reactults):
            category = pred.get('category', 'Unknown')
            coords = pred.get('coords', [])
            score = pred.get('score', 0.0)
            source = pred.get('source', 'unknown')
            coords_str = f"[{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}, {coords[3]:.1f}]" if coords else "[]"
            print(f"   {i+1}. {category:20s} - score: {score:.3f}, source: {source}, bbox: {coords_str}")
        

        
        # 2. Фильтруем предсказания
        print("\nStep 2: Filtering predictions...")
        
        if predictions:
            filtered_predictions = auto_label.filter(
                image_name=image_name,
                predictions=predictions.copy(),
                text_threshold=0.05,
                use_similar_prompting=True,
                iou_threshold=0.9,
                max_lower_bound=1,
                min_lower_bound=0
            )
            
            print(f"✅ Filtering completed!")
            print(f"   Before filtering: {len(predictions)} predictions")
            print(f"   After filtering: {len(filtered_predictions)} predictions")
        else:
            filtered_predictions = []
            print("⚠️  No predictions to filter")
        
        # 3. Тестируем метод apply_additional_task для keypoints
        print("\nStep 3: Testing apply_additional_task for keypoints...")
        
        # Фильтруем только 'person' предсказания для keypoints
        person_predictions = filtered_predictions#[p for p in filtered_predictions if p.get('category') == 'person']
        
        print(f"   Found {len(person_predictions)} predictions for keypoint detection")
        
        keypoint_result = auto_label.apply_additional_task(
            image=pil_image,
            predictions=person_predictions,
            task_type="animal_pose",
            system_promt="You are a animal pose estimation assistant. Detect animal and their keypoints accurately.",
            promt="Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format."
        )
        
        if keypoint_result['success']:
            print(f"      ✅ Keypoint detection successful!")
            kp_result = keypoint_result['keypoint_result']
            print(f"         Detected keypoints for {kp_result['success_count']} persons")
            
            # Выводим информацию о keypoints
            if kp_result.get('keypoints'):
                print(f"         Keypoint names: {', '.join(kp_result['keypoint_names'][:5])}...")
                print(f"         Total keypoints per person: {len(kp_result['keypoint_names'])}")
        else:
            print(f"      ❌ Keypoint detection failed: {keypoint_result.get('error', 'Unknown error')}")
        
        # 4. Создаем визуализации
        print("\nStep 4: Creating visualizations...")
        
        output_dir = Path("output_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Визуализация detection
        try:
            fig_detection = auto_label.get_image_with_bboxes(image_name, show_filtered=True)
            if fig_detection:
                detection_path = output_dir / f"{image_path_obj.stem}_detection.png"
                fig_detection.savefig(detection_path, dpi=150, bbox_inches='tight')
                plt.close(fig_detection)
                print(f"   ✓ Detection visualization saved to: {detection_path}")
        except Exception as e:
            print(f"   ⚠️  Could not create detection visualization: {e}")
        
        # Визуализация с keypoints
        if 'keypoint_result' in locals() and keypoint_result.get('success', False):
            try:
                fig_kp = auto_label.get_image_with_bboxes(
                    image_name, 
                    show_filtered=True,
                    keypoint_result=keypoint_result
                )
                if fig_kp:
                    kp_path = output_dir / f"{image_path_obj.stem}_keypoints.png"
                    fig_kp.savefig(kp_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_kp)
                    print(f"   ✓ Keypoint visualization saved to: {kp_path}")
            except Exception as e:
                print(f"   ⚠️  Could not create keypoint visualization: {e}")
        
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up resources...")
        print("Done!")