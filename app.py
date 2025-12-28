import streamlit as st
import sys
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import json

# Устанавливаем неинтерактивный бэкенд для matplotlib перед импортом matplotlib
import matplotlib
matplotlib.use('Agg')

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    'max_tokens': 1024,
    'temperature': 0.75,
    'top_p': 0.7,
    'top_k': 10,
    'repetition_penalty': 1
}

# Cache heavy operations
@st.cache_resource
def load_autolabel_module():
    """Lazy loading of AutoLabel module"""
    from autolabel import AutoLabel
    return AutoLabel

def initialize_app():
    """Initialize application state"""
    default_keys = [
        'autolabel', 'initialized', 'current_image', 'raw_result',
        'filtered_result', 'keypoint_result', 'images_list', 'model_params',
        'task_config', 'class_names', 'classes_for_similar',
        'generate_visual_prompting', 'images_path', 'save_path',
        'prompt_template', 'use_similar_prompting_filter', 'text_threshold',
        'iou_threshold', 'max_lower_bound', 'min_lower_bound',
        'keypoint_task_type', 'keypoint_system_prompt', 'keypoint_task_prompt',
        'raw_fig', 'filtered_fig', 'keypoint_fig'
    ]
    
    for key in default_keys:
        if key not in st.session_state:
            if key == 'model_params':
                st.session_state[key] = DEFAULT_MODEL_PARAMS
            elif key in ['text_threshold', 'iou_threshold', 'min_lower_bound', 'max_lower_bound']:
                st.session_state[key] = 0.0 if 'min' in key else (0.05 if 'text' in key else 1.0 if 'max' in key else 0.9)
            elif key == 'use_similar_prompting_filter':
                st.session_state[key] = True
            elif key == 'generate_visual_prompting':
                st.session_state[key] = True
            elif key == 'keypoint_task_type':
                st.session_state[key] = "human_pose"
            elif key == 'keypoint_system_prompt':
                st.session_state[key] = "You are a human pose estimation assistant. Detect people and their keypoints accurately."
            elif key == 'keypoint_task_prompt':
                st.session_state[key] = "Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format."
            elif key == 'prompt_template':
                st.session_state[key] = "Detect {categories_str}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
            else:
                st.session_state[key] = None

def display_image_with_bboxes(fig):
    """Display image with bounding boxes"""
    if fig:
        st.pyplot(fig)
        plt.close(fig)

def save_visualization(fig, image_name, suffix, save_path=None):
    """Save visualization to file"""
    if save_path and fig:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename without extension
        stem = Path(image_name).stem
        output_path = save_dir / f"{stem}_{suffix}.png"
        
        try:
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            st.error(f"Error saving {suffix}: {e}")
            return None
    return None

def main():
    st.set_page_config(
        page_title="AutoLabel UI",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏷️ AutoLabel UI")
    st.markdown("---")
    
    initialize_app()
    
    # Section 1: Parameter Initialization
    if not st.session_state.initialized:
        st.header("🔧 Parameter Initialization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            images_path = st.text_input(
                "Path to images folder",
                value=st.session_state.images_path or "",
                help="Absolute path to images folder"
            )
            
            save_path = st.text_input(
                "Path for saving results",
                value=st.session_state.save_path or "./output",
                help="Path for saving annotated images"
            )
            
            class_names_str = st.text_input(
                "Class names (comma-separated)",
                value=", ".join(st.session_state.class_names) if st.session_state.class_names else "forklift, pallet, pallet_truck, small_load_carrier, stillage",
                help="List of detection classes separated by commas"
            )
        
        with col2:
            prompt_template = st.text_area(
                "Prompt Template",
                value=st.session_state.prompt_template,
                height=100,
                help="Prompt template for detection"
            )
            
            classes_for_similar_str = st.text_input(
                "Classes for Similar Prompting (comma-separated)",
                value=", ".join(st.session_state.classes_for_similar) if st.session_state.classes_for_similar else "pallet, stillage",
                help="Classes for which similar prompting will be used"
            )
            
            generate_visual_prompting = st.checkbox(
                "Generate Visual Prompting",
                value=st.session_state.generate_visual_prompting,
                help="Whether to use visual prompting during inference"
            )
        
        st.markdown("---")
        
        if st.button("🚀 Initialize AutoLabel", type="primary", width='stretch'):
            with st.spinner("Initializing AutoLabel... This may take a few minutes..."):
                try:
                    # Parse parameters
                    class_names = [name.strip() for name in class_names_str.split(",")]
                    classes_for_similar = [name.strip() for name in classes_for_similar_str.split(",")] if classes_for_similar_str else []
                    
                    # Create TaskConfig
                    categories_str = ", ".join(class_names)
                    from utils.schema import TaskConfig
                    task_config = TaskConfig(
                        name="Detection",
                        prompt_template=prompt_template.format(categories_str=categories_str),
                        description="",
                        output_format="boxes",
                        requires_categories=False,
                    )
                    
                    # Save parameters to session state
                    st.session_state.model_params = DEFAULT_MODEL_PARAMS
                    st.session_state.task_config = task_config
                    st.session_state.class_names = class_names
                    st.session_state.classes_for_similar = classes_for_similar
                    st.session_state.generate_visual_prompting = generate_visual_prompting
                    st.session_state.images_path = images_path
                    st.session_state.save_path = save_path
                    st.session_state.prompt_template = prompt_template
                    
                    # Lazy initialization of AutoLabel
                    try:
                        AutoLabel = load_autolabel_module()
                        
                        # Initialize AutoLabel
                        st.session_state.autolabel = AutoLabel(
                            model_params=DEFAULT_MODEL_PARAMS,
                            task="detection",
                            classes_for_similar_prompting=classes_for_similar,
                            class_names=class_names,
                            images_path=images_path,
                            system_prompt="You are an object detection assistant.",
                            task_config=task_config
                        )
                        
                        # Get list of images
                        if images_path and Path(images_path).exists():
                            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                            st.session_state.images_list = [
                                str(f) for f in Path(images_path).iterdir() 
                                if f.suffix.lower() in image_extensions
                            ]
                            st.success(f"✅ Found {len(st.session_state.images_list)} images")
                        else:
                            st.warning("⚠️ Images folder not specified or does not exist")
                            st.session_state.images_list = []
                        
                        st.session_state.initialized = True
                        st.success("✅ AutoLabel successfully initialized!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error initializing AutoLabel: {str(e)}")
                        
                except Exception as e:
                    st.error(f"❌ Initialization error: {str(e)}")
    
    else:
        # AutoLabel is initialized
        st.sidebar.header("📊 Status")
        st.sidebar.success("✅ AutoLabel initialized")
        
        # Information about current parameters
        with st.sidebar.expander("📋 Current Parameters"):
            st.write(f"**Classes:** {', '.join(st.session_state.class_names)}")
            st.write(f"**Similar prompting for:** {', '.join(st.session_state.classes_for_similar)}")
            st.write(f"**Visual prompting:** {'Yes' if st.session_state.generate_visual_prompting else 'No'}")
            st.write(f"**Images folder:** {st.session_state.images_path}")
            st.write(f"**Save folder:** {st.session_state.save_path}")
        
        # Section 2: Image Selection and Inference
        st.header("🖼️ Image Selection and Inference")
        
        if st.session_state.images_list:
            # Image selection
            image_files = [Path(img).name for img in st.session_state.images_list]
            selected_image = st.selectbox(
                "Select image",
                options=image_files,
                index=0
            )
            
            # Find full path to selected image
            selected_path = next(img for img in st.session_state.images_list 
                               if Path(img).name == selected_image)
            
            # Display image
            col_img, col_actions = st.columns([2, 1])
            
            with col_img:
                st.subheader("Image")
                try:
                    image = Image.open(selected_path)
                    st.image(image, caption=selected_image, width='stretch')
                    st.session_state.current_image = image
                    st.session_state.current_image_path = selected_path
                    st.session_state.current_image_name = selected_image
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            with col_actions:
                st.subheader("Actions")
                
                # Button for inference
                if st.button("🎯 Run Markup", type="primary", width='stretch'):
                    with st.spinner("Running inference..."):
                        try:
                            # Run inference
                            result = st.session_state.autolabel.inference(
                                image=selected_path,
                                generate_visual_prompting=st.session_state.generate_visual_prompting
                            )
                            
                            if result['success']:
                                st.session_state.raw_result = result
                                
                                # Create visualization
                                fig_raw = st.session_state.autolabel.get_image_with_bboxes(
                                    result['image_name'],
                                    show_filtered=False
                                )
                                st.session_state.raw_fig = fig_raw
                                
                                st.success(f"✅ Inference complete: {len(result['predictions'])} predictions")
                            else:
                                st.error(f"Inference error: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Error during inference: {str(e)}")
            
            st.markdown("---")
            
            # Section 3: Results Display and Filtering
            if st.session_state.raw_result:
                st.header("📊 Results")
                
                # Display RawResult and FilterResult side by side
                col_raw, col_filter = st.columns(2)
                
                with col_raw:
                    st.subheader("Raw Result")
                    if st.session_state.raw_fig:
                        display_image_with_bboxes(st.session_state.raw_fig)
                    
                    # Raw result info
                    if st.session_state.raw_result and st.session_state.raw_result.get('predictions'):
                        predictions = st.session_state.raw_result['predictions']
                        st.metric("Number of predictions", len(predictions))
                        
                        # Save RawResult button
                        if st.button("💾 Save Raw Result", type="secondary", width='stretch'):
                            if st.session_state.raw_fig and st.session_state.save_path:
                                saved_path = save_visualization(
                                    st.session_state.raw_fig, 
                                    st.session_state.current_image_name, 
                                    "raw", 
                                    st.session_state.save_path
                                )
                                if saved_path:
                                    st.success(f"✅ Raw result saved: {saved_path}")
                            else:
                                st.warning("No raw result visualization available")
                
                # Filtering section
                with col_filter:
                    st.subheader("Filtering")
                    
                    # Filter parameters
                    with st.expander("Filter Settings", expanded=True):
                        text_threshold = st.slider(
                            "Text Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.text_threshold,
                            step=0.01,
                            help="Threshold for text alignment"
                        )
                        
                        use_similar_prompting = st.checkbox(
                            "Use Similar Prompting",
                            value=st.session_state.use_similar_prompting_filter
                        )
                        
                        iou_threshold = st.slider(
                            "IoU Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=st.session_state.iou_threshold,
                            step=0.01,
                            help="Threshold for NMS"
                        )
                        
                        col_bounds = st.columns(2)
                        with col_bounds[0]:
                            min_lower_bound = st.slider(
                                "Min Lower Bound",
                                min_value=0.0,
                                max_value=1.0,
                                value=st.session_state.min_lower_bound,
                                step=0.01,
                                help="Minimum relative size"
                            )
                        
                        with col_bounds[1]:
                            max_lower_bound = st.slider(
                                "Max Lower Bound",
                                min_value=0.0,
                                max_value=1.0,
                                value=st.session_state.max_lower_bound,
                                step=0.01,
                                help="Maximum relative size"
                            )
                        
                        # Save filter values
                        st.session_state.text_threshold = text_threshold
                        st.session_state.use_similar_prompting_filter = use_similar_prompting
                        st.session_state.iou_threshold = iou_threshold
                        st.session_state.min_lower_bound = min_lower_bound
                        st.session_state.max_lower_bound = max_lower_bound
                    
                    # Filter button
                    if st.button("🔍 Apply Filtering", type="primary", width='stretch'):
                        with st.spinner("Applying filters..."):
                            try:
                                filtered_predictions = st.session_state.autolabel.filter(
                                    image_name=st.session_state.raw_result['image_name'],
                                    predictions=st.session_state.raw_result['predictions'].copy(),
                                    text_threshold=text_threshold,
                                    use_similar_prompting=use_similar_prompting,
                                    iou_threshold=iou_threshold,
                                    max_lower_bound=max_lower_bound,
                                    min_lower_bound=min_lower_bound
                                )
                                
                                st.session_state.filtered_result = {
                                    'image_name': st.session_state.raw_result['image_name'],
                                    'predictions': filtered_predictions
                                }
                                
                                # Create filtered visualization
                                fig_filtered = st.session_state.autolabel.get_image_with_bboxes(
                                    st.session_state.raw_result['image_name'],
                                    show_filtered=True
                                )
                                st.session_state.filtered_fig = fig_filtered
                                
                                before = len(st.session_state.raw_result['predictions'])
                                after = len(filtered_predictions)
                                st.success(f"✅ Filtering complete: {after} predictions (was {before})")
                                
                            except Exception as e:
                                st.error(f"Filtering error: {str(e)}")
                    
                    # Display filtered result
                    if st.session_state.filtered_fig:
                        display_image_with_bboxes(st.session_state.filtered_fig)
                        
                        if st.session_state.filtered_result:
                            st.metric("Filtered predictions", len(st.session_state.filtered_result['predictions']))
                            
                            # Save FilterResult button
                            if st.button("💾 Save Filtered Result", type="secondary", width='stretch'):
                                if st.session_state.filtered_fig and st.session_state.save_path:
                                    saved_path = save_visualization(
                                        st.session_state.filtered_fig, 
                                        st.session_state.current_image_name, 
                                        "filtered", 
                                        st.session_state.save_path
                                    )
                                    if saved_path:
                                        st.success(f"✅ Filtered result saved: {saved_path}")
                                else:
                                    st.warning("No filtered result visualization available")
                
                st.markdown("---")
                
                # Section 4: Keypoint Task
                st.header("📍 Keypoint Task")
                
                col_kp_params, col_kp_result = st.columns([1, 2])
                
                with col_kp_params:
                    st.subheader("Keypoint Settings")
                    
                    keypoint_task_type = st.selectbox(
                        "Keypoint Task Type",
                        options=["human_pose", "animal_pose", "face_keypoint"],
                        index=["human_pose", "animal_pose", "face_keypoint"].index(
                            st.session_state.keypoint_task_type
                        ) if st.session_state.keypoint_task_type in ["human_pose", "animal_pose", "face_keypoint"] else 0
                    )
                    
                    system_prompt = st.text_area(
                        "System Prompt",
                        value=st.session_state.keypoint_system_prompt,
                        height=100
                    )
                    
                    task_prompt = st.text_area(
                        "Task Prompt",
                        value=st.session_state.keypoint_task_prompt,
                        height=100
                    )
                    
                    # Save keypoint parameters
                    st.session_state.keypoint_task_type = keypoint_task_type
                    st.session_state.keypoint_system_prompt = system_prompt
                    st.session_state.keypoint_task_prompt = task_prompt
                    
                    # Keypoint markup button
                    if st.button("🎯 Run Keypoint Markup", type="primary", width='stretch'):
                        with st.spinner("Running keypoint markup..."):
                            try:
                                # Use either filtered or raw predictions
                                predictions_to_use = (
                                    st.session_state.filtered_result['predictions'] 
                                    if st.session_state.filtered_result 
                                    else st.session_state.raw_result['predictions']
                                )
                                
                                if predictions_to_use:
                                    keypoint_result = st.session_state.autolabel.apply_additional_task(
                                        image=st.session_state.current_image_path,
                                        predictions=predictions_to_use,
                                        task_type=keypoint_task_type,
                                        system_promt=system_prompt,
                                        promt=task_prompt
                                    )
                                    
                                    st.session_state.keypoint_result = keypoint_result
                                    
                                    if keypoint_result['success']:
                                        # Create keypoint visualization
                                        fig_kp = st.session_state.autolabel.get_image_with_bboxes(
                                            st.session_state.raw_result['image_name'],
                                            show_filtered=True,
                                            keypoint_result=keypoint_result
                                        )
                                        st.session_state.keypoint_fig = fig_kp
                                        st.success("✅ Keypoint detection complete!")
                                    else:
                                        st.error("Keypoint detection failed")
                                else:
                                    st.warning("⚠️ No predictions available for keypoint detection")
                                        
                            except Exception as e:
                                st.error(f"Keypoint task error: {str(e)}")
                
                with col_kp_result:
                    st.subheader("Keypoint Result")
                    
                    if st.session_state.keypoint_fig:
                        display_image_with_bboxes(st.session_state.keypoint_fig)
                        
                        # Save Keypoint Result button
                        if st.button("💾 Save Keypoint Result", type="secondary", width='stretch'):
                            if st.session_state.keypoint_fig and st.session_state.save_path:
                                saved_path = save_visualization(
                                    st.session_state.keypoint_fig, 
                                    st.session_state.current_image_name, 
                                    "keypoints", 
                                    st.session_state.save_path
                                )
                                if saved_path:
                                    st.success(f"✅ Keypoint result saved: {saved_path}")
                            else:
                                st.warning("No keypoint visualization available")
                    
                    # Keypoint information
                    if st.session_state.keypoint_result and st.session_state.keypoint_result.get('success'):
                        kp_result = st.session_state.keypoint_result['keypoint_result']
                        st.metric("Successfully processed", kp_result.get('success_count', 0))
                        
                        if kp_result.get('keypoints'):
                            keypoint_names = kp_result.get('keypoint_names', [])
                            st.write(f"**Keypoint names:** {', '.join(keypoint_names[:5])}{'...' if len(keypoint_names) > 5 else ''}")
                            
                            # Show keypoints details
                            with st.expander("Show keypoints details"):
                                for person_idx, person_kps in enumerate(kp_result['keypoints'][:2]):
                                    st.write(f"**Person {person_idx + 1}:**")
                                    for kp_idx, (kp_name, kp_coords) in enumerate(zip(keypoint_names, person_kps)):
                                        if kp_idx < 5:
                                            st.write(f"  {kp_name}: {kp_coords}")
            
            else:
                st.info("ℹ️ Run inference on an image to access filtering and keypoint tasks.")
        
        else:
            st.warning("⚠️ No images found in the specified folder or folder does not exist.")
            
            # Show current parameters
            with st.expander("Current Parameters"):
                st.json({
                    "class_names": st.session_state.class_names,
                    "classes_for_similar": st.session_state.classes_for_similar,
                    "images_path": st.session_state.images_path,
                    "save_path": st.session_state.save_path
                })
        
        # Reset button
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🔄 Reset and Start Over", type="secondary", width='stretch'):
                # Clear UI state while preserving some parameters
                keys_to_keep = [
                    'model_params', 'class_names', 'classes_for_similar',
                    'generate_visual_prompting', 'images_path', 'save_path',
                    'prompt_template', 'text_threshold', 'iou_threshold',
                    'min_lower_bound', 'max_lower_bound', 'use_similar_prompting_filter',
                    'keypoint_task_type', 'keypoint_system_prompt', 'keypoint_task_prompt'
                ]
                
                # Create temporary copy of values to keep
                saved_values = {}
                for key in keys_to_keep:
                    if key in st.session_state:
                        saved_values[key] = st.session_state[key]
                
                # Clear all keys
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Restore saved values
                for key, value in saved_values.items():
                    st.session_state[key] = value
                
                # Reinitialize basic values
                st.session_state.initialized = False
                st.rerun()

if __name__ == "__main__":
    main()