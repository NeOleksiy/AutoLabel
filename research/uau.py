from PIL import Image
import os

def yolo_to_pixel_bbox(image_path):
    """
    Преобразует нормализованные координаты YOLO в абсолютные координаты пикселей.
    Предполагает структуру: 
    - images/image.jpg
    - labels/image.txt
    
    Args:
        image_path (str): Путь к изображению из папки images
        
    Returns:
        list: Список словарей с информацией о bounding boxes
    """
    # Получаем путь к соответствующему файлу аннотаций в папке labels
    image_dir = os.path.dirname(image_path)
    dataset_dir = os.path.dirname(image_dir)  # Поднимаемся на уровень выше images
    labels_dir = os.path.join(dataset_dir, "labels")
    
    # Получаем имя файла без расширения
    image_filename = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_filename)
    annotation_path = os.path.join(labels_dir, base_name + ".txt")
    
    # Открываем изображение для получения размеров
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    bboxes = []
    
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                
                if len(data) < 5:
                    continue
                
                class_id = int(data[0])
                x_center_norm = float(data[1])
                y_center_norm = float(data[2])
                width_norm = float(data[3])
                height_norm = float(data[4])
                
                # Сохраняем нормализованные координаты
                normalized_coords = [x_center_norm, y_center_norm, width_norm, height_norm]
                
                # Преобразуем нормализованные координаты в пиксельные
                x_center = x_center_norm * img_width
                y_center = y_center_norm * img_height
                width = width_norm * img_width
                height = height_norm * img_height
                
                # Конвертируем из центра и размеров в угловые координаты
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                pixel_coords = (x_min, y_min, x_max, y_max)
                
                bboxes.append({
                    'class_id': class_id,
                    'normalized_coords': normalized_coords,
                    'pixel_coords': pixel_coords,
                    'image_size': (img_width, img_height)
                })
                
    except FileNotFoundError:
        print(f"Файл аннотации {annotation_path} не найден")
        return []
    
    return bboxes

# Пример использования
if __name__ == "__main__":
    # Пример пути: "dataset/images/image001.jpg"
    # Соответствующая аннотация: "dataset/labels/image001.txt"
    bboxes = yolo_to_pixel_bbox("Grocery1-1/test/images/20211009_181739_jpg.rf.21a6e363a8436fd343cfd005f4757f48.jpg")
    
    for bbox in bboxes:
        print(f"Class: {bbox['class_id']}")
        print(f"Нормализованные координаты: {bbox['normalized_coords']}")
        print(f"Пиксельные координаты: ({bbox['pixel_coords'][0]:.2f}, {bbox['pixel_coords'][1]:.2f}) - ({bbox['pixel_coords'][2]:.2f}, {bbox['pixel_coords'][3]:.2f})")
        print(f"Размер изображения: {bbox['image_size']}")
        print()

