import os
import sys
import json
import streamlit as st
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import logging
from autolabel import AutoLabel
import time
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autolabel.log')
    ]
)
logger = logging.getLogger(__name__)


# Инициализация session_state
def init_session_state():
    """Инициализация состояния сессии"""
    if 'autolabel' not in st.session_state:
        st.session_state.autolabel = None
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'inference_params' not in st.session_state:
        st.session_state.inference_params = {}
    if 'filter_params' not in st.session_state:
        st.session_state.filter_params = {
            'text_threshold': 0.1,
            'use_similar_prompting': True,
            'iou_threshold': 0.9,
            'max_lower_bound': 1.0,
            'min_lower_bound': 0.0
        }
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'inference_complete' not in st.session_state:
        st.session_state.inference_complete = False
    if 'filter_applied' not in st.session_state:
        st.session_state.filter_applied = False


class UIApp:
    def __init__(self):
        init_session_state()
        logger.info("UIApp initialized")
    
    def run(self):
        st.set_page_config(layout="wide", page_title="AutoLabeling Tool")
        
        st.title("🖼️ AutoLabeling Tool")
        st.markdown("---")
        
        # Главный контейнер с двумя колонками
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("Настройки Inference")
            
            # Параметры для inference()
            task = st.selectbox(
                "Task",
                ["detection", "segmentation", "keypointing"],
                index=0,
                help="Тип задачи: detection - детекция объектов, segmentation - сегментация, keypointing - ключевые точки"
            )
            
            st.subheader("Параметры модели Rex-Omni")
            
            # Параметры модели в expander
            with st.expander("Параметры генерации", expanded=True):
                max_tokens = st.number_input(
                    "max_tokens",
                    min_value=1,
                    max_value=4096,
                    value=1024,
                    help="Максимальное количество токенов для генерации (1-4096)"
                )
                
                temperature = st.slider(
                    "temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05,
                    help="Контролирует случайность генерации: выше = более случайно, ниже = более детерминированно"
                )
                
                top_p = st.slider(
                    "top_p",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Нуклеусная выборка: учитывает только токены с суммарной вероятностью p"
                )
                
                top_k = st.number_input(
                    "top_k",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Ограничивает выборку k наиболее вероятными токенами"
                )
                
                repetition_penalty = st.slider(
                    "repetition_penalty",
                    min_value=1.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Штраф за повторение: выше = меньше повторений"
                )
            
            generate_visual_prompting = st.checkbox(
                "generate_visual_prompting",
                value=False,
                help="Использовать визуальный промптинг для поиска похожих объектов"
            )
            
            st.subheader("Классы")
            
            # Ввод class_names
            class_names_input = st.text_area(
                "class_names (один класс на строку, до 500 классов)",
                value="car\ndoor\nhandrail\nsidewalk\nstaircase\nstreet_light\nwindow",
                height=150,
                help="Массив классов для детекции. Каждый класс на новой строке"
            )
            
            if class_names_input:
                class_names = [c.strip() for c in class_names_input.split('\n') if c.strip()]
            else:
                class_names = []
                st.warning("Введите хотя бы один класс")
            
            # Ввод classes_for_similar_prompting
            classes_for_similar_prompting_input = st.text_area(
                "classes_for_similar_prompting (один класс на строку)",
                value="car\nwindow",
                height=100,
                help="Классы для которых будет использоваться визуальный промптинг. Оставьте пустым для всех классов"
            )
            
            if classes_for_similar_prompting_input:
                classes_for_similar_prompting = [c.strip() for c in classes_for_similar_prompting_input.split('\n') if c.strip()]
            else:
                classes_for_similar_prompting = class_names  # По умолчанию все классы
            
            # Путь к изображениям
            images_path = st.text_input(
                "images_path",
                value="./images",
                help="Путь к папке с изображениями для обработки"
            )
            
            # Проверка существования пути
            if images_path and not Path(images_path).exists():
                st.warning(f"Путь {images_path} не существует")
            
            # Кнопка запуска inference
            if st.button("🎯 Запустить Inference", type="primary", width='content'):
                if not class_names:
                    st.error("Пожалуйста, укажите хотя бы один класс в class_names")
                    st.stop()
                
                if not images_path or not Path(images_path).exists():
                    st.error(f"Папка {images_path} не существует")
                    st.stop()
                
                # Сохраняем параметры в session_state
                st.session_state.inference_params = {
                    'model_params': {
                        'max_tokens': max_tokens,
                        'temperature': temperature,
                        'top_p': top_p,
                        'top_k': top_k,
                        'repetition_penalty': repetition_penalty
                    },
                    'task': task,
                    'generate_visual_prompting': generate_visual_prompting,
                    'classes_for_similar_prompting': classes_for_similar_prompting,
                    'class_names': class_names,
                    'images_path': images_path
                }
                
                # Создаем экземпляр AutoLabel
                try:
                    logger.info("Creating AutoLabel instance...")
                    st.session_state.autolabel = AutoLabel(
                        model_params=st.session_state.inference_params['model_params'],
                        task=task,
                        classes_for_similar_prompting=classes_for_similar_prompting,
                        class_names=class_names,
                        images_path=images_path
                    )
                    
                    # Запускаем inference
                    with st.spinner("Выполняется inference..."):
                        success = st.session_state.autolabel.inference(
                            generate_visual_prompting=generate_visual_prompting
                        )
                    
                    if success:
                        st.session_state.inference_complete = True
                        st.session_state.processed_images = list(st.session_state.autolabel.raw_predictions.keys())
                        st.session_state.filter_applied = False
                        
                        logger.info(f"Inference completed successfully, processed {len(st.session_state.processed_images)} images")
                        
                        st.success("✅ Inference завершен!")
                        st.rerun()  # Перезапускаем для обновления интерфейса
                        
                except Exception as e:
                    logger.error(f"Error during inference: {str(e)}", exc_info=True)
                    st.error(f"Ошибка при выполнении inference: {str(e)}")
        
        with col2:
            st.header("Результаты")
            
            # Если inference выполнен, показываем результаты
            if st.session_state.inference_complete and st.session_state.autolabel:
                # Панель фильтрации в сайдбаре
                with st.sidebar:
                    st.header("⚙️ Настройки фильтрации")
                    st.markdown("---")
                    
                    # Параметры фильтрации без автоматического применения
                    text_threshold = st.slider(
                        "text_threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.filter_params['text_threshold'],
                        step=0.01,
                        help="Порог соответствия текста и изображения (0-1). Выше = строже"
                    )
                    
                    use_similar_prompting = st.checkbox(
                        "use_similar_prompting",
                        value=st.session_state.filter_params['use_similar_prompting'],
                        help="Использовать предсказания от визуального промптинга (похожие)"
                    )
                    
                    iou_threshold = st.slider(
                        "iou_threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.filter_params['iou_threshold'],
                        step=0.05,
                        help="Порог IoU для NMS (0-1). Выше = больше дубликатов удаляется"
                    )
                    
                    st.subheader("Границы размера")
                    col_bound1, col_bound2 = st.columns(2)
                    with col_bound1:
                        min_lower_bound = st.slider(
                            "min_lower_bound",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.filter_params['min_lower_bound'],
                            step=0.05,
                            help="Минимальный относительный размер bounding box"
                        )
                    
                    with col_bound2:
                        max_lower_bound = st.slider(
                            "max_lower_bound",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.filter_params['max_lower_bound'],
                            step=0.05,
                            help="Максимальный относительный размер bounding box"
                        )
                    
                    # Сохраняем параметры фильтрации
                    st.session_state.filter_params = {
                        'text_threshold': text_threshold,
                        'use_similar_prompting': use_similar_prompting,
                        'iou_threshold': iou_threshold,
                        'max_lower_bound': max_lower_bound,
                        'min_lower_bound': min_lower_bound
                    }
                    
                    # Кнопка применения фильтрации
                    if st.button("🔄 Применить фильтрацию", type="secondary", width='content'):
                        with st.spinner("Применяем фильтры..."):
                            start_time = time.time()
                            st.session_state.autolabel.filter(
                                text_threshold=text_threshold,
                                use_similar_prompting=use_similar_prompting,
                                iou_threshold=iou_threshold,
                                max_lower_bound=max_lower_bound,
                                min_lower_bound=min_lower_bound
                            )
                            filter_time = time.time() - start_time
                            st.session_state.filter_applied = True
                            
                            logger.info(f"Filter applied in {filter_time:.2f}s")
                        
                        st.success(f"✅ Фильтрация применена за {filter_time:.2f} секунд!")
                        st.rerun()
                
                # Основная область для отображения результатов
                st.header("📸 Результаты предсказаний")
                
                # Показываем статистику
                with st.expander("📊 Статистика обработки", expanded=False):
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Обработано изображений", st.session_state.autolabel.stats['successful_inferences'])
                        st.metric("Неудачных обработок", st.session_state.autolabel.stats['failed_inferences'])
                    with col_stat2:
                        st.metric("Всего предсказаний", st.session_state.autolabel.stats['total_predictions'])
                        if st.session_state.inference_params.get('generate_visual_prompting', False):
                            st.metric("Визуальных предсказаний", st.session_state.autolabel.stats['visual_prompting_predictions'])
                    with col_stat3:
                        st.metric("Отфильтровано NMS", st.session_state.autolabel.stats.get('nms_filtered', 0))
                        st.metric("Всего изображений", st.session_state.autolabel.stats['total_images'])
                    
                    if st.session_state.filter_applied:
                        st.metric("После фильтрации", st.session_state.autolabel.stats['total_filtered_predictions'])
                
                # Выбор изображения для просмотра
                if st.session_state.processed_images:
                    # Используем индекс для сохранения выбора
                    image_index = st.selectbox(
                        "Выберите изображение для просмотра",
                        range(len(st.session_state.processed_images)),
                        format_func=lambda x: st.session_state.processed_images[x],
                        help="Выберите изображение из обработанных"
                    )
                    
                    selected_image = st.session_state.processed_images[image_index]
                    
                    if selected_image:
                        # Две колонки для сравнения
                        col_view1, col_view2 = st.columns(2)
                        
                        with col_view1:
                            st.subheader("📋 Исходные предсказания")
                            fig_raw = st.session_state.autolabel.get_image_with_bboxes(
                                selected_image, 
                                show_filtered=False
                            )
                            if fig_raw:
                                st.pyplot(fig_raw, width='content')
                                plt.close()
                            
                            raw_data = st.session_state.autolabel.raw_predictions.get(selected_image, {})
                            raw_count = len(raw_data.get('predictions', []))
                            
                            # Статистика по source
                            sources = {}
                            for pred in raw_data.get('predictions', []):
                                source = pred.get('source', 'unknown')
                                sources[source] = sources.get(source, 0) + 1
                            
                            st.metric("Всего предсказаний", raw_count)
                            for source, count in sources.items():
                                st.caption(f"{source}: {count}")
                        
                        with col_view2:
                            st.subheader("✅ После фильтрации" if st.session_state.filter_applied else "⚠️ Фильтрация не применена")
                            
                            if st.session_state.filter_applied:
                                fig_filtered = st.session_state.autolabel.get_image_with_bboxes(
                                    selected_image, 
                                    show_filtered=True
                                )
                                if fig_filtered:
                                    st.pyplot(fig_filtered, width='content')
                                    plt.close()
                                
                                filtered_data = st.session_state.autolabel.filtered_predictions.get(selected_image, {})
                                filtered_count = len(filtered_data.get('predictions', []))
                                
                                # Статистика по source после фильтрации
                                sources_filtered = {}
                                for pred in filtered_data.get('predictions', []):
                                    source = pred.get('source', 'unknown')
                                    sources_filtered[source] = sources_filtered.get(source, 0) + 1
                                
                                st.metric("После фильтрации", filtered_count)
                                for source, count in sources_filtered.items():
                                    st.caption(f"{source}: {count}")
                                
                                # Процент сохраненных
                                if raw_count > 0:
                                    percent_kept = (filtered_count / raw_count) * 100
                                    st.metric("Сохранено", f"{percent_kept:.1f}%")
                            else:
                                st.info("Нажмите 'Применить фильтрацию' для отображения отфильтрованных результатов")
                
                # Кнопки для экспорта результатов
                st.markdown("---")
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    if st.button("📥 Экспорт JSON", type="secondary", width='content'):
                        self.export_results()
                
                with col_export2:
                    if st.button("🖼️ Сохранить изображения", type="secondary", width='content'):
                        self.save_images_with_bboxes()
                
                with col_export3:
                    if st.button("🧹 Очистить результаты", type="secondary", width='content'):
                        logger.info("Clearing results...")
                        st.session_state.inference_complete = False
                        st.session_state.autolabel = None
                        st.session_state.processed_images = []
                        st.session_state.selected_image = None
                        st.session_state.filter_applied = False
                        st.rerun()
            
            elif st.session_state.inference_complete and not st.session_state.autolabel:
                st.warning("Inference был выполнен, но данные потеряны. Пожалуйста, запустите inference снова.")
                st.session_state.inference_complete = False
            else:
                st.info("Настройте параметры и нажмите 'Запустить Inference' для начала обработки")
    
    def export_results(self):
        """Экспорт результатов в формате JSON"""
        if not st.session_state.autolabel:
            st.warning("Нет данных для экспорта")
            return
        
        logger.info("Exporting results to JSON...")
        
        export_data = {
            'inference_params': st.session_state.inference_params,
            'filter_params': st.session_state.filter_params,
            'stats': st.session_state.autolabel.stats,
            'predictions': {}
        }
        
        # Используем прогресс бар для экспорта
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img_name in enumerate(st.session_state.processed_images[:10]):  # Экспортируем первые 10
            status_text.text(f"Экспорт изображения {i+1}/10...")
            progress_bar.progress((i + 1) / 10)
            
            data = st.session_state.autolabel.filtered_predictions.get(img_name)
            if data:
                export_data['predictions'][img_name] = {
                    'path': data['path'],
                    'predictions': data['predictions']
                }
        
        # Сохраняем в JSON
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(export_data, f, indent=2, default=str)
            temp_path = f.name
        
        # Предлагаем скачать
        with open(temp_path, 'r') as f:
            st.download_button(
                label="📥 Скачать JSON",
                data=f,
                file_name="autolabel_results.json",
                mime="application/json",
                key="download_json"
            )
        
        # Очищаем прогресс бар
        progress_bar.empty()
        status_text.text("Экспорт завершен!")
        
        logger.info(f"Exported {len(export_data['predictions'])} images to JSON")
        
        # Удаляем временный файл
        import threading
        def cleanup_temp_file(path):
            import time
            time.sleep(5)  # Даем время на скачивание
            if os.path.exists(path):
                os.unlink(path)
        
        threading.Thread(target=cleanup_temp_file, args=(temp_path,)).start()
    
    def save_images_with_bboxes(self):
        """Сохранение изображений с bounding boxes"""
        if not st.session_state.autolabel:
            return
        
        logger.info("Saving images with bounding boxes...")
        
        import tempfile
        import shutil
        
        # Создаем временную папку
        temp_dir = tempfile.mkdtemp()
        
        try:
            saved_count = 0
            # Используем прогресс бар
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, img_name in enumerate(st.session_state.processed_images[:10]):  # Сохраняем первые 10
                status_text.text(f"Сохранение изображения {i+1}/10...")
                progress_bar.progress((i + 1) / 10)
                
                if st.session_state.filter_applied:
                    fig = st.session_state.autolabel.get_image_with_bboxes(img_name, show_filtered=True)
                else:
                    fig = st.session_state.autolabel.get_image_with_bboxes(img_name, show_filtered=False)
                    
                if fig:
                    save_path = os.path.join(temp_dir, f"{img_name}.png")
                    fig.savefig(save_path, bbox_inches='tight', dpi=150, format='png')
                    plt.close(fig)
                    saved_count += 1
            
            # Очищаем прогресс бар
            progress_bar.empty()
            status_text.empty()
            
            if saved_count > 0:
                # Создаем архив
                archive_path = os.path.join(tempfile.gettempdir(), "autolabel_images.zip")
                shutil.make_archive(archive_path.replace('.zip', ''), 'zip', temp_dir)
                
                # Предлагаем скачать архив
                with open(archive_path, 'rb') as f:
                    st.download_button(
                        label=f"📦 Скачать архив ({saved_count} изображений)",
                        data=f,
                        file_name="autolabel_images.zip",
                        mime="application/zip",
                        key="download_zip"
                    )
                
                st.success(f"✅ Сохранено {saved_count} изображений")
                logger.info(f"Saved {saved_count} images to archive")
            else:
                st.warning("Не удалось сохранить изображения")
            
        except Exception as e:
            logger.error(f"Error saving images: {e}")
            st.error(f"Ошибка при сохранении изображений: {e}")
        finally:
            # Очистка
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    logger.info("Starting AutoLabeling Tool")
    app = UIApp()
    app.run()