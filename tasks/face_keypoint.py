from PIL import Image
from rex_omni import RexOmniWrapper
from typing import List, Dict, Any, Optional, Union
import warnings
from pathlib import Path
from utils.schema import TaskConfig
warnings.filterwarnings('ignore')


class FaceKeypointTask:
    
    def __init__(self, logger,
                system_promt: str = "You are a human face estimation assistant. Detect people face and their keypoints accurately.",
                task_promt: str = "Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format.",
                model_params: Dict[str, Any] = None):
        self.logger = logger
        self.model_params = model_params or {
            'max_tokens': 1024,
            'temperature': 0.75,
            'top_p': 0.7,
            'top_k': 10,
            'repetition_penalty': 1
        }
        self.system_promt = system_promt
        # Keypoints for human pose (COCO format)
        self.keypoints_list = [
            "left_eye", "right_eye", "left_ear", "right_ear","mouth","nose","left_eyebrow","right_eyebrow"
        ]
        
        # Task config for keypoint detection
        self.task_config = TaskConfig(
            name="Keypoint",
            prompt_template=task_promt,
            description="Detect faces with keypoints",
            output_format="keypoints",
            requires_categories=True,
            requires_keypoint_type=True,
        )
        
        self.rex = None
        self._init_model()
    
    def _init_model(self):
        try:
            self.logger.info("Initializing FaceKeypointTask model...")
            
            categories = ['person face']
            keypoints_str = ", ".join(self.keypoints_list)
            
            prompt_template = self.task_config.prompt_template.replace(
                "{categories}", ", ".join(categories)
            ).replace(
                "{keypoints}", keypoints_str
            )
            
            task_config = TaskConfig(
                name=self.task_config.name,
                prompt_template=prompt_template,
                description=self.task_config.description,
                output_format=self.task_config.output_format,
                requires_categories=False,
                requires_keypoint_type=False,
            )
            
            # Initialize RexOmniWrapper
            self.rex = RexOmniWrapper(
                model_path="IDEA-Research/Rex-Omni",
                backend="transformers",
                max_tokens=self.model_params.get('max_tokens', 1024),
                temperature=self.model_params.get('temperature', 0.75),
                top_p=self.model_params.get('top_p', 0.7),
                top_k=self.model_params.get('top_k', 10),
                repetition_penalty=self.model_params.get('repetition_penalty', 1),
                task_config=task_config,
                system_prompt=self.system_promt
            )
            
            self.logger.info("FaceKeypointTask initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing FaceKeypointTask: {e}")
            raise RuntimeError(f"Error initializing FaceKeypointTask: {e}")
    
    def mark_up(self, image: Union[str, Path, Image.Image], 
                bboxes: List[List], 
                class_names: List[str]) -> Dict[str, Any]:
        
        try:
            self.logger.info(f"Starting keypoint detection for {len(bboxes)} persons")
            
            # Загрузка изображения если передан путь
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            else:
                pil_image = image
            
            bboxes = []
            indices = []

            for i, box in enumerate(bboxes):
                bboxes.append(box)
                indices.append(i)
            
            
            self.logger.info(f"Found {len(bboxes)} person bboxes for keypoint detection")
            
            # Используем visual prompting для keypoint detection
            self.logger.info("👤 Performing person keypointing...")
            results = self.rex.inference(
                images=image, task="keypoint", keypoint_type="face", categories=["person face"]
            )
            
            # Парсим результаты
            all_keypoints = []
            all_scores = []
            
            if results and isinstance(results, list) and len(results) > 0:
                result = results[0]
                
                if "extracted_predictions" in result:
                    extracted = result["extracted_predictions"]
                    
                    if isinstance(extracted, dict):
                        for preds in extracted.get('person', []):
                            if 'keypoints' in preds:
                                keypoints_data = preds['keypoints']
                                
                                person_keypoints = []
                                person_scores = []
                                
                                for kp_name in self.keypoints_list:
                                    if kp_name in keypoints_data:
                                        kp_coords = keypoints_data[kp_name]
                                        if isinstance(kp_coords, list) and len(kp_coords) >= 2:
                                            person_keypoints.append([kp_coords[0], kp_coords[1]])
                                            person_scores.append(preds.get('score', 0.5))
                                        else:
                                            person_keypoints.append([0, 0])  # Not visible
                                            person_scores.append(0.0)
                                    else:
                                        person_keypoints.append([0, 0])  # Not visible
                                        person_scores.append(0.0)
                                
                                all_keypoints.append(person_keypoints)
                                all_scores.append(person_scores)
        
            
            result = {
                'success_count': len(all_keypoints),
                'keypoints': all_keypoints,
                'scores': all_scores,
                'bboxes': bboxes,
                'keypoint_names': self.keypoints_list,
                'image_size': pil_image.size
            }
            
            self.logger.info(f"Keypoint detection completed: {len(all_keypoints)} persons with keypoints")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in mark_up for keypoints: {e}")