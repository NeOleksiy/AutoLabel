import matplotlib.pyplot as plt
import torch
from PIL import Image

from rex_omni import RexOmniVisualize, RexOmniWrapper


def main():
    # Model path - replace with your actual model path
    model_path = "IDEA-Research/Rex-Omni"

    print("🚀 Initializing Rex Omni model...")

    # Create wrapper with custom parameters
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",  # Choose "transformers" or "vllm"
        max_tokens=4096,
        temperature=0.1,
        top_p=0.90,
        top_k=10,
        repetition_penalty=1,
    )

    # Load image
    image_path = "Grocery1-1/test/images/20211009_181739_jpg.rf.21a6e363a8436fd343cfd005f4757f48.jpg"
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"✅ Image loaded successfully!")
    print(f"📏 Image size: {image.size}")
    
    visual_prompts = [
        [0.040625, 0.87890625, 0.0765625, 0.165625],
        # [0.3875, 0.2951388888888889, 0.059375, 0.09305555555555556],

    ]
    
    def normalized_to_pixel_coordinates(normalized_boxes, img_width, img_height):
        """
        Преобразует нормализованные bounding box в пиксельные координаты
        Формат: [x_center, y_center, width, height] -> [x1, y1, x2, y2]
        """
        pixel_boxes = []
        
        for box in normalized_boxes:
            x_center, y_center, w, h = box
            
            # Преобразуем в абсолютные координаты
            x_center_pixel = x_center * img_width
            y_center_pixel = y_center * img_height
            width_pixel = w * img_width
            height_pixel = h * img_height
            
            # Вычисляем координаты углов
            x1 = x_center_pixel - (width_pixel / 2)
            y1 = y_center_pixel - (height_pixel / 2)
            x2 = x_center_pixel + (width_pixel / 2)
            y2 = y_center_pixel + (height_pixel / 2)
            
            pixel_boxes.append([x1, y1, x2, y2])
        
        return pixel_boxes
    
    # Преобразуем координаты
    pixel_coordinates = normalized_to_pixel_coordinates(visual_prompts, width, height)
    
    print("📐 Преобразованные координаты в пикселях:")
    for i, coords in enumerate(pixel_coordinates):
        x1, y1, x2, y2 = coords
        print(f"Box {i+1}: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
        print(f"   Ширина: {x2-x1:.1f}px, Высота: {y2-y1:.1f}px")

    print("🎯 Performing object pointing...")
    results = rex_model.inference(
        images=image,
        task="visual_prompting",
        visual_prompt_boxes=pixel_coordinates,
    )

    # Process results
    result = results[0]
    if result["success"]:
        predictions = result["extracted_predictions"]
        vis_image = RexOmniVisualize(
            image=image,
            predictions=predictions,
            font_size=30,
            draw_width=10,
            show_labels=True,
        )

        # Save visualization
        output_path = (
            "exa.jpg"
        )
        vis_image.save(output_path)
    else:
        print(f"❌ Inference failed: {result['error']}")


if __name__ == "__main__":
    main()