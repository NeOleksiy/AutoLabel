import torch
from PIL import Image
from typing import List, Dict, Any
from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor


class SegmentationTask:
    
    def __init__(self, logger):
        self.logger = logger
        # Инициализация модели SAM3
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        model = build_sam3_image_model()
        processor = Sam3Processor(model)
        
    def mark_up(self, image: Image.Image, bboxes: List[List], class_names: List[str], promt: str) -> Dict[str, Any]:
        try:
            self.logger.info(f"Начинаем сегментацию для {len(bboxes)} объектов")
            
            if len(bboxes) != len(class_names):
                raise ValueError(f"Количество bboxes ({len(bboxes)}) не совпадает с количеством class_names ({len(class_names)})")
            
            # Подготовка изображения для модели
            inference_state = self.processor.set_image(image)
            
            all_masks = []
            all_boxes = []
            all_scores = []
            
            for i, bbox in enumerate(bboxes):
                try:
                    class_name = class_names[i]
                    self.logger.info(f"Обработка объекта {i+1}: класс '{class_name}', bbox {bbox}")
                    

                    output = self.processor.set_text_prompt(
                        state=inference_state, 
                        prompt=promt
                    )
                    
                    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                    
                    if len(masks) > 0:
                        # Выбираем лучшую маску по оценке
                        best_idx = torch.argmax(scores).item()
                        all_masks.append(masks[best_idx])
                        all_boxes.append(boxes[best_idx].tolist())
                        all_scores.append(scores[best_idx].item())
                        
                        self.logger.info(f"  Объект {i+1}: найдена маска с оценкой {scores[best_idx].item():.3f}")
                    else:
                        self.logger.warning(f"  Объект {i+1}: не найдено подходящих масок")
                        all_masks.append(None)
                        all_boxes.append(bbox)
                        all_scores.append(0.0)
                        
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке объекта {i+1}: {str(e)}")
                    all_masks.append(None)
                    all_boxes.append(bbox)
                    all_scores.append(0.0)
            
            result = {
                'masks': all_masks,
                'boxes': all_boxes,
                'scores': all_scores,
                'class_names': class_names,
                'image_size': image.size,
                'success_count': sum(1 for mask in all_masks if mask is not None)
            }
            
            self.logger.info(f"Сегментация завершена. Успешно обработано: {result['success_count']}/{len(bboxes)} объектов")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка в процессе сегментации: {str(e)}")
            raise


# import torch
# from PIL import Image
# from sam3.sam3.model_builder import build_sam3_image_model
# from sam3.sam3.model.sam3_image_processor import Sam3Processor

# # model = build_sam3_image_model(bpe_path=, checkpoint_path=)
# processor = Sam3Processor(model)

# image = Image.open("/home/efimenko.aleksey/rex/Logistic-1/test/images/1564562879122-79_jpg.rf.626a15f304f18d604db601e15fd5092f.jpg")
# inference_state = processor.set_image(image)

# output = processor.set_text_prompt(state=inference_state, prompt="<YOUR_TEXT_PROMPT>")

# masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
