from PIL import Image
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import cv2
import os
import glob
from pathlib import Path

from rex_omni import RexOmniVisualize, RexOmniWrapper


class AnimalWeightAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        # Оставляем только ключевые точки, важные для определения веса
        self.keypoint_names = [
            "neck", "root of tail",  # ось тела
            "left shoulder", "right shoulder",  # ширина плеч
            "left hip", "right hip"  # ширина бедер
        ]
    
    def parse_keypoints_from_prediction(self, prediction):
        keypoints = []
        
        for point_name in self.keypoint_names:
            if point_name in prediction['keypoints']:
                # Координаты уже в числовом формате [x, y]
                coords = prediction['keypoints'][point_name]
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    x, y = float(coords[0]), float(coords[1])
                    keypoints.append([x, y])
                else:
                    # Если формат неправильный, используем [0, 0]
                    keypoints.append([0, 0])
            else:
                # Если точки нет в предсказании, используем [0, 0]
                keypoints.append([0, 0])
        
        return np.array(keypoints)
    
    def extract_body_measurements(self, keypoints):
        """Извлекаем измерения тела только из важных ключевых точек"""
        # Фильтруем валидные точки (не [0,0])
        valid_mask = ~np.all(keypoints == [0, 0], axis=1)
        valid_keypoints = keypoints[valid_mask]
        
        if len(valid_keypoints) < 3:
            return None
        
        measurements = {}
        
        # Площадь bounding box
        x_coords = valid_keypoints[:, 0]
        y_coords = valid_keypoints[:, 1]
        bbox_width = np.max(x_coords) - np.min(x_coords)
        bbox_height = np.max(y_coords) - np.min(y_coords)
        bbox_area = bbox_width * bbox_height
        
        measurements['bbox_area'] = bbox_area
        
        # Площадь выпуклой оболочки (по основным точкам тела)
        try:
            hull = ConvexHull(valid_keypoints)
            hull_area = hull.volume
        except:
            hull_area = bbox_area
        measurements['hull_area'] = hull_area
        
        # Длина тела (шея -> корень хвоста) - самый важный параметр
        neck_idx = 0  # neck
        tail_root_idx = 1  # root of tail
        
        if (valid_mask[neck_idx] and valid_mask[tail_root_idx] and 
            not np.all(keypoints[neck_idx] == [0, 0]) and 
            not np.all(keypoints[tail_root_idx] == [0, 0])):
            body_length = np.linalg.norm(keypoints[neck_idx] - keypoints[tail_root_idx])
        else:
            # Альтернатива: используем ширину bounding box
            body_length = bbox_width
        measurements['body_length'] = body_length
        
        # Ширина в плечах
        left_shoulder_idx = 2
        right_shoulder_idx = 3
        if (valid_mask[left_shoulder_idx] and valid_mask[right_shoulder_idx] and
            not np.all(keypoints[left_shoulder_idx] == [0, 0]) and
            not np.all(keypoints[right_shoulder_idx] == [0, 0])):
            shoulder_width = np.linalg.norm(keypoints[left_shoulder_idx] - keypoints[right_shoulder_idx])
        else:
            shoulder_width = bbox_width * 0.6
        measurements['shoulder_width'] = shoulder_width
        
        # Ширина в бедрах
        left_hip_idx = 4
        right_hip_idx = 5
        if (valid_mask[left_hip_idx] and valid_mask[right_hip_idx] and
            not np.all(keypoints[left_hip_idx] == [0, 0]) and
            not np.all(keypoints[right_hip_idx] == [0, 0])):
            hip_width = np.linalg.norm(keypoints[left_hip_idx] - keypoints[right_hip_idx])
        else:
            hip_width = bbox_width * 0.5
        measurements['hip_width'] = hip_width
        
        # Объем тела (приблизительный) - основной показатель для веса
        body_volume = body_length * shoulder_width * hip_width
        measurements['body_volume'] = body_volume
        
        # Пропорции тела
        if shoulder_width > 0 and hip_width > 0:
            measurements['shoulder_hip_ratio'] = shoulder_width / hip_width
        else:
            measurements['shoulder_hip_ratio'] = 1.0
        
        # Количество валидных точек
        measurements['valid_points_count'] = np.sum(valid_mask)
        
        return measurements
    
    def analyze_multiple_animals(self, all_predictions):
        """Анализирует всех животных из всех изображений"""
        all_measurements = []
        valid_predictions = []
        
        print(f"🔍 Analyzing {len(all_predictions)} animal predictions from all images...")
        
        # Собираем измерения для всех обнаруженных животных
        for img_idx, (image_path, predictions) in enumerate(all_predictions):
            for pred_idx, pred in enumerate(predictions):
                if 'keypoints' in pred:
                    keypoints = self.parse_keypoints_from_prediction(pred)
                    measurements = self.extract_body_measurements(keypoints)
                    
                    if measurements is not None and measurements['valid_points_count'] >= 3:
                        measurements['prediction_index'] = len(all_measurements)
                        measurements['instance_id'] = pred.get('instance_id', f'img{img_idx+1}_animal{pred_idx+1}')
                        measurements['image_path'] = image_path
                        measurements['image_index'] = img_idx
                        measurements['animal_index'] = pred_idx
                        all_measurements.append(measurements)
                        valid_predictions.append((image_path, pred))
        
        if not all_measurements:
            print("❌ No valid animals found for weight analysis")
            return None, None, None
        
        print(f"📊 Successfully processed {len(all_measurements)} animals from all images")
        
        # Создаем DataFrame для анализа
        df = pd.DataFrame(all_measurements)
        
        # Комбинированный показатель размера (основные параметры для веса)
        size_features = df[['body_length', 'shoulder_width', 'hip_width', 'body_volume']].copy()
        
        # Нормализуем признаки
        size_features_scaled = self.scaler.fit_transform(size_features)
        
        # Взвешенная комбинация (объем тела имеет наибольший вес)
        weights = np.array([0.2, 0.2, 0.2, 0.4])  # объем тела - 40%
        df['size_score'] = np.dot(size_features_scaled, weights)
        
        # Относительный вес (0-1)
        df['relative_weight'] = (df['size_score'] - df['size_score'].min()) / (df['size_score'].max() - df['size_score'].min())
        
        # Ранжирование по размеру (1 - самый крупный)
        df['size_rank'] = df['size_score'].rank(ascending=False).astype(int)
        
        # Категоризация по весу
        n_clusters = min(3, len(df))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['weight_category'] = kmeans.fit_predict(size_features_scaled)
            
            # Переименовываем категории от легких к тяжелым
            category_mapping = {}
            for category in sorted(df['weight_category'].unique()):
                category_data = df[df['weight_category'] == category]
                mean_size = category_data['size_score'].mean()
                category_mapping[category] = mean_size
            
            sorted_categories = sorted(category_mapping.items(), key=lambda x: x[1])
            new_mapping = {old_cat: new_cat for new_cat, (old_cat, _) in enumerate(sorted_categories)}
            df['weight_category'] = df['weight_category'].map(new_mapping)
        else:
            df['weight_category'] = 0
            new_mapping = {0: 0}
        
        return df, valid_predictions, new_mapping
    
    def create_visualization(self, image, predictions, analysis_df, category_mapping, output_path):
        """Создает визуализацию с весовыми категориями и рангами"""
        # Конвертируем PIL в OpenCV для рисования
        if isinstance(image, Image.Image):
            vis_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            vis_image = image.copy()
        
        # Цвета для категорий
        category_colors = {
            0: (0, 255, 0),    # Зеленый - легкие
            1: (255, 255, 0),  # Желтый - средние  
            2: (255, 0, 0)     # Красный - тяжелые
        }
        
        category_labels = {
            0: "LIGHT",
            1: "MEDIUM", 
            2: "HEAVY"
        }
        
        # Получаем животных для этого изображения
        image_animals = analysis_df[analysis_df['image_path'] == output_path.replace('_weight_analysis.jpg', '')]
        
        for _, row in image_animals.iterrows():
            # Находим соответствующее предсказание
            pred = None
            for img_path, pred_data in predictions:
                if img_path == row['image_path'] and pred_data.get('instance_id', '') == row.get('instance_id', ''):
                    pred = pred_data
                    break
            
            if pred is None:
                continue
                
            # Получаем bounding box из предсказания
            if 'bbox' in pred:
                bbox = pred['bbox']
                if len(bbox) >= 4:
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                else:
                    continue
            else:
                # Если bbox нет, вычисляем из ключевых точек
                keypoints = self.parse_keypoints_from_prediction(pred)
                valid_mask = ~np.all(keypoints == [0, 0], axis=1)
                valid_keypoints = keypoints[valid_mask]
                if len(valid_keypoints) > 0:
                    x_coords = valid_keypoints[:, 0]
                    y_coords = valid_keypoints[:, 1]
                    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                else:
                    continue
            
            category = int(row['weight_category'])
            color = category_colors.get(category, (255, 255, 255))
            
            # Рисуем bounding box
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Текст с информацией
            text = f"{category_labels[category]} | Rank: {int(row['size_rank'])} | W: {row['relative_weight']:.2f}"
            cv2.putText(vis_image, text, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Добавляем instance_id
            instance_id = row.get('instance_id', f"Animal {int(row['animal_index'])+1}")
            cv2.putText(vis_image, instance_id, (x_min, y_min - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Рисуем важные ключевые точки
            keypoints = self.parse_keypoints_from_prediction(pred)
            for i, kp in enumerate(keypoints):
                if not np.all(kp == [0, 0]):
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(vis_image, (x, y), 5, color, -1)
                    cv2.putText(vis_image, str(i), (x+5, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    
    def print_analysis_report(self, analysis_df, category_mapping):
        """Печатает детальный отчет анализа"""
        print("\n" + "="*60)
        print("🐘 СВОДНЫЙ АНАЛИЗ ВЕСА ЖИВОТНЫХ (ОПТИМИЗИРОВАННЫЙ)")
        print("="*60)
        
        category_labels = {0: "ЛЕГКИЕ", 1: "СРЕДНИЕ", 2: "ТЯЖЕЛЫЕ"}
        
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Всего обработано животных: {len(analysis_df)}")
        print(f"   Всего изображений: {analysis_df['image_path'].nunique()}")
        print(f"   Средний относительный вес: {analysis_df['relative_weight'].mean():.3f}")
        print(f"   Средняя длина тела: {analysis_df['body_length'].mean():.1f} px")
        print(f"   Средняя ширина плеч: {analysis_df['shoulder_width'].mean():.1f} px")
        print(f"   Средняя ширина бедер: {analysis_df['hip_width'].mean():.1f} px")
        
        for category in sorted(analysis_df['weight_category'].unique()):
            category_data = analysis_df[analysis_df['weight_category'] == category]
            
            print(f"\n📊 {category_labels[category]} животные ({len(category_data)} шт., {len(category_data)/len(analysis_df)*100:.1f}%):")
            print(f"   Средний относительный вес: {category_data['relative_weight'].mean():.3f}")
            print(f"   Средняя длина тела: {category_data['body_length'].mean():.1f} px")
            print(f"   Средний объем тела: {category_data['body_volume'].mean():.0f} px³")
        
        print(f"\n🏆 ТОП-10 САМЫХ КРУПНЫХ ЖИВОТНЫХ:")
        for _, animal in analysis_df.nlargest(10, 'size_score').iterrows():
            category_label = category_labels[int(animal['weight_category'])]
            instance_id = animal.get('instance_id', f"Animal {int(animal['animal_index'])+1}")
            image_name = os.path.basename(animal['image_path'])
            print(f"   {int(animal['size_rank']):2d}. {instance_id} "
                  f"({category_label}) - Вес: {animal['relative_weight']:.3f} "
                  f"Длина: {animal['body_length']:.0f}px "
                  f"[{image_name}]")


def process_images_folder(folder_path, output_folder, animal_category="cow"):
    """Обрабатывает все изображения в папке"""
    
    # Создаем выходную папку если не существует
    os.makedirs(output_folder, exist_ok=True)
    
    # Находим все изображения в папке
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, extension)))
        image_paths.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    print(f"📁 Found {len(image_paths)} images in folder: {folder_path}")
    
    if not image_paths:
        print("❌ No images found in the specified folder")
        return
    
    # Инициализируем модель
    print("🚀 Initializing Rex Omni model...")
    model_path = "IDEA-Research/Rex-Omni"
    
    rex_model = RexOmniWrapper(
        model_path=model_path,
        backend="transformers",
        max_tokens=2048,
        temperature=0.1,
        top_p=0.95,
        top_k=5,
        repetition_penalty=1,
    )
    
    # Обрабатываем каждое изображение
    all_predictions = []
    weight_analyzer = AnimalWeightAnalyzer()
    
    for i, image_path in enumerate(image_paths):
        print(f"\n🖼️  Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert("RGB")
            print(f"   📏 Image size: {image.size}")
            
            # Детекция ключевых точек
            results = rex_model.inference(
                images=image, 
                task="keypoint", 
                keypoint_type="animal", 
                categories=[animal_category]
            )
            
            result = results[0]
            if result["success"]:
                # Получаем предсказания для этого изображения
                predictions = result["extracted_predictions"].get(animal_category, [])
                print(f"   ✅ Found {len(predictions)} animals")
                
                # Сохраняем предсказания для общего анализа
                all_predictions.append((image_path, predictions))
                
                # Создаем визуализацию ключевых точек для этого изображения
                vis_image = RexOmniVisualize(
                    image=image,
                    predictions=result["extracted_predictions"],
                    font_size=6,
                    draw_width=6,
                    show_labels=True,
                )
                
                # Сохраняем визуализацию ключевых точек
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                keypoints_output = os.path.join(output_folder, f"{base_name}_keypoints.jpg")
                vis_image.save(keypoints_output)
                print(f"   💾 Keypoints visualization saved: {keypoints_output}")
                
            else:
                print(f"   ❌ Inference failed: {result['error']}")
                
        except Exception as e:
            print(f"   ❌ Error processing image {image_path}: {str(e)}")
    
    # Проводим общий анализ всех животных
    if all_predictions:
        print(f"\n📊 Starting global analysis of all animals...")
        analysis_df, valid_predictions, category_mapping = weight_analyzer.analyze_multiple_animals(all_predictions)
        
        if analysis_df is not None:
            # Печатаем сводный отчет
            weight_analyzer.print_analysis_report(analysis_df, category_mapping)
            
            # Создаем визуализации с весовыми категориями для каждого изображения
            print(f"\n🎨 Creating weight analysis visualizations...")
            for image_path, predictions in all_predictions:
                try:
                    image = Image.open(image_path).convert("RGB")
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(output_folder, f"{base_name}_weight_analysis.jpg")
                    
                    weighted_image = weight_analyzer.create_visualization(
                        image, valid_predictions, analysis_df, category_mapping, output_path
                    )
                    
                    weighted_image.save(output_path)
                    print(f"   💾 Weight analysis saved: {output_path}")
                    
                except Exception as e:
                    print(f"   ❌ Error creating visualization for {image_path}: {str(e)}")
            
            # Сохраняем сводную таблицу с данными
            csv_output = os.path.join(output_folder, "animals_weight_analysis.csv")
            analysis_df.to_csv(csv_output, index=False)
            print(f"\n💾 Full analysis data saved to: {csv_output}")
            
        else:
            print("❌ No valid animals found for weight analysis")
    else:
        print("❌ No successful predictions from any images")



def main():
    # Укажите путь к папке с изображениями и выходную папку
    input_folder = "cows-1/train"  # Замените на ваш путь
    output_folder = "weight_analysis_results"
    animal_category = "cow's body"  # Или "cat", "dog", etc.
    
    # Обрабатываем все изображения в папке
    process_images_folder(input_folder, output_folder, animal_category)


if __name__ == "__main__":
    main()