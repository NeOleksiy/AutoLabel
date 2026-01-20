import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import io

# Import your code
from autolabel import AutoLabel
from utils.schema import TaskConfig

# Page settings
st.set_page_config(
    page_title="AutoLabel UI",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'autolabel' not in st.session_state:
    st.session_state.autolabel = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'current_image_name' not in st.session_state:
    st.session_state.current_image_name = None
if 'raw_predictions' not in st.session_state:
    st.session_state.raw_predictions = None
if 'filtered_predictions' not in st.session_state:
    st.session_state.filtered_predictions = None
if 'react_predictions' not in st.session_state:
    st.session_state.react_predictions = None
if 'keypoint_result' not in st.session_state:
    st.session_state.keypoint_result = None

def initialize_autolabel():
    """Initialize AutoLabel"""
    try:
        # Prepare parameters
        model_params = {
            'max_tokens': st.session_state.max_tokens,
            'temperature': st.session_state.temperature,
            'top_p': st.session_state.top_p,
            'top_k': st.session_state.top_k,
            'repetition_penalty': st.session_state.repetition_penalty
        }
        
        # Prepare task_config
        categories_str = ", ".join(st.session_state.class_names.split(','))
        task_config = TaskConfig(
            name="Detection",
            prompt_template=st.session_state.prompt_template.format(categories=categories_str),
            description="",
            output_format="boxes",
            requires_categories=False,
        )
        
        # Initialize AutoLabel
        autolabel = AutoLabel(
            model_params=model_params,
            task="detection",
            classes_for_similar_prompting=[x.strip() for x in st.session_state.similar_classes.split(',')] 
                if st.session_state.similar_classes else [],
            class_names=[x.strip() for x in st.session_state.class_names.split(',')],
            images_path=st.session_state.images_path,
            system_prompt="You are an object detection assistant.",
            task_config=task_config
        )
        
        st.session_state.autolabel = autolabel
        return True
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return False

def load_image(image_path):
    """Load image"""
    try:
        image = Image.open(image_path).convert("RGB")
        st.session_state.current_image = image
        st.session_state.current_image_name = Path(image_path).name
        return image
    except Exception as e:
        st.error(f"Image loading error: {str(e)}")
        return None

def run_inference():
    """Run inference"""
    try:
        if st.session_state.current_image and st.session_state.autolabel:
            result = st.session_state.autolabel.inference(
                image=st.session_state.current_image,
                generate_visual_prompting=st.session_state.generate_visual_prompting
            )
            
            if result['success']:
                st.session_state.raw_predictions = result['predictions']
                st.success(f"Inference successful! Found objects: {len(result['predictions'])}")
                return True
            else:
                st.error(f"Inference error: {result.get('error', 'Unknown error')}")
                return False
    except Exception as e:
        st.error(f"Inference execution error: {str(e)}")
        return False

def run_filter():
    """Run filtering"""
    try:
        if st.session_state.raw_predictions and st.session_state.autolabel:
            filtered = st.session_state.autolabel.filter(
                image_name=st.session_state.current_image_name,
                predictions=st.session_state.raw_predictions.copy(),
                text_threshold=st.session_state.filter_text_threshold,
                use_similar_prompting=st.session_state.use_similar_prompting,
                iou_threshold=st.session_state.filter_iou_threshold,
                max_lower_bound=st.session_state.max_lower_bound,
                min_lower_bound=st.session_state.min_lower_bound
            )
            
            st.session_state.filtered_predictions = filtered
            st.success(f"Filtering successful! Remaining objects: {len(filtered)}")
            return True
    except Exception as e:
        st.error(f"Filtering error: {str(e)}")
        return False

def run_react():
    """Run ReAct"""
    try:
        if st.session_state.autolabel and st.session_state.current_image_name:
            # Use either filtered predictions or raw predictions
            predictions = st.session_state.filtered_predictions or st.session_state.raw_predictions
            
            if predictions:
                react_result = st.session_state.autolabel.react(
                    image_name=st.session_state.current_image_name,
                    predictions=predictions.copy(),
                    iou_threshold=st.session_state.react_iou_threshold,
                    n_repeats=st.session_state.number_repeats
                )
                
                st.session_state.react_predictions = react_result
                st.success(f"ReAct successful! Total objects: {len(react_result)}")
                return True
            else:
                st.warning("No predictions for ReAct")
                return False
    except Exception as e:
        st.error(f"ReAct error: {str(e)}")
        return False

def run_keypoint_task():
    """Run keypoint task"""
    try:
        if st.session_state.autolabel and st.session_state.current_image:
            # Use either filtered predictions or raw predictions
            predictions = st.session_state.filtered_predictions or st.session_state.raw_predictions
            
            if predictions:
                result = st.session_state.autolabel.apply_additional_task(
                    image=st.session_state.current_image,
                    predictions=predictions,
                    task_type=st.session_state.keypoint_task_type,
                    system_promt=st.session_state.keypoint_system_prompt,
                    promt=st.session_state.keypoint_task_prompt
                )
                
                st.session_state.keypoint_result = result
                
                if result['success']:
                    kp_result = result['keypoint_result']
                    st.success(f"Keypoint task successful! Processed objects: {kp_result.get('success_count', 0)}")
                    return True
                else:
                    st.error("Keypoint task failed")
                    return False
            else:
                st.warning("No predictions for keypoint task")
                return False
    except Exception as e:
        st.error(f"Keypoint task error: {str(e)}")
        return False

def get_image_with_bboxes(image, predictions, title="Image with BBoxes"):
    """Create image with bounding boxes"""
    try:
        if not predictions:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(image)
            ax.set_title(title)
            ax.axis('off')
            return fig
        
        # Use method from AutoLabel if available
        if st.session_state.autolabel:
            # Create temporary entry in raw_predictions to use the method
            temp_data = {
                'image': image,
                'predictions': predictions,
                'path': st.session_state.current_image_name
            }
            st.session_state.autolabel.raw_predictions[st.session_state.current_image_name] = temp_data
            
            fig = st.session_state.autolabel.get_image_with_bboxes(
                img_name=st.session_state.current_image_name,
                show_filtered=True
            )
            
            if fig:
                fig.suptitle(title, fontsize=16)
                return fig
        
        # If method didn't work, create simple visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        for pred in predictions:
            if 'coords' in pred:
                coords = pred['coords']
                category = pred.get('category', 'Unknown')
                score = pred.get('score', 1.0)
                
                rect = plt.Rectangle(
                    (coords[0], coords[1]), 
                    coords[2] - coords[0], 
                    coords[3] - coords[1],
                    linewidth=2, 
                    edgecolor='red', 
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                ax.text(
                    coords[0], 
                    coords[1] - 5, 
                    f"{category}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.7),
                    fontsize=9,
                    color='black'
                )
        
        ax.set_title(title)
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

# Application title
st.title("🏷️ AutoLabel UI")
st.markdown("---")

# Main application structure
tab1, tab2 = st.tabs(["🚀 Initialization", "🎯 Annotation"])

with tab1:
    # Initialization section
    st.header("Initialize AutoLabel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model parameters
        st.subheader("Model Parameters")
        st.number_input("max_tokens", min_value=256, max_value=4096, value=1024, key="max_tokens", step=256)
        st.slider("temperature", min_value=0.0, max_value=2.0, value=0.75, key="temperature", step=0.05)
        st.slider("top_p", min_value=0.0, max_value=1.0, value=0.7, key="top_p", step=0.05)
        st.number_input("top_k", min_value=1, max_value=100, value=10, key="top_k", step=1)
        st.slider("repetition_penalty", min_value=0.0, max_value=2.0, value=1.0, key="repetition_penalty", step=0.1)
        
        # Data settings
        st.subheader("Data Settings")
        st.text_input("Path to images folder", value="./images", key="images_path")
        
        # Class names
        st.text_input("Class names (comma-separated)", 
                     value="cat, dog, person, car", 
                     key="class_names",
                     help="Example: cat, dog, person, car")
        
        # Classes for similar prompting
        st.text_input("Classes for similar prompting (comma-separated)", 
                     value="cat, dog", 
                     key="similar_classes",
                     help="Leave empty to disable")
    
    with col2:
        # Prompt template
        st.subheader("Prompt Template")
        default_prompt = "Detect {categories}. Output the bounding box coordinates in [x0, y0, x1, y1] format."
        st.text_area("Prompt Template", 
                    value=default_prompt,
                    key="prompt_template",
                    height=150,
                    help="Use {categories} to insert class list")
        
        # Visual prompting
        st.subheader("Visual Prompting")
        st.checkbox("Enable Visual Prompting", value=True, key="generate_visual_prompting")
        
        # Initialization button
        st.markdown("---")
        if st.button("🚀 Initialize AutoLabel", type="primary", width='stretch'):
            with st.spinner("Initializing AutoLabel..."):
                if initialize_autolabel():
                    st.success("AutoLabel successfully initialized!")
                    
                    # Show initialization info
                    if st.session_state.autolabel:
                        st.info(f"""
                        **Initialization Information:**
                        - Number of classes: {len(st.session_state.autolabel.class_names)}
                        - Classes for similar prompting: {st.session_state.autolabel.classes_for_similar_prompting}
                        - OwlViT available: {st.session_state.autolabel.owlvit_available}
                        """)

with tab2:
    if not st.session_state.autolabel:
        st.warning("Please initialize AutoLabel first in the 'Initialization' tab")
    else:
        # Image loading
        st.header("Image Loading")
        
        images_path = Path(st.session_state.images_path)
        if images_path.exists() and images_path.is_dir():
            image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + \
                         list(images_path.glob("*.png")) + list(images_path.glob("*.bmp"))
            
            if image_files:
                image_names = [f.name for f in image_files]
                selected_image = st.selectbox("Select image:", image_names)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("📥 Load Image", width='stretch'):
                        image_path = images_path / selected_image
                        with st.spinner("Loading image..."):
                            image = load_image(image_path)
                            if image:
                                st.success(f"Image loaded: {selected_image}")
                
                with col2:
                    if st.session_state.current_image:
                        # Show image
                        st.image(st.session_state.current_image, caption=f"Current image: {selected_image}", width='content')
                        
                        # Inference button
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            if st.button("🎯 Run Inference", type="primary", width='stretch'):
                                with st.spinner("Running inference..."):
                                    if run_inference():
                                        st.rerun()
                        with col_b:
                            if st.session_state.raw_predictions:
                                st.info(f"Found objects: {len(st.session_state.raw_predictions)}")
            else:
                st.error(f"No images found in folder {st.session_state.images_path}")
        else:
            st.error(f"Folder {st.session_state.images_path} does not exist")
        
        # Main annotation interface
        if st.session_state.current_image:
            st.markdown("---")
            
            # 4 images in a row
            st.subheader("Processing Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.session_state.current_image:
                    st.image(st.session_state.current_image, caption="Original Image", width='content')
            
            with col2:
                if st.session_state.raw_predictions is not None:
                    fig = get_image_with_bboxes(
                        st.session_state.current_image, 
                        st.session_state.raw_predictions,
                        "After Inference"
                    )
                    if fig:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        st.image(buf, caption=f"Inference: {len(st.session_state.raw_predictions)} objects", width='content')
                        plt.close(fig)
                else:
                    st.info("Click 'Run Inference'")
            
            with col3:
                if st.session_state.react_predictions is not None:
                    fig = get_image_with_bboxes(
                        st.session_state.current_image, 
                        st.session_state.react_predictions,
                        "After ReAct"
                    )
                    if fig:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        st.image(buf, caption=f"ReAct: {len(st.session_state.react_predictions)} objects", width='content')
                        plt.close(fig)
                else:
                    st.info("Run ReAct below")
            
            with col4:
                if st.session_state.filtered_predictions is not None:
                    fig = get_image_with_bboxes(
                        st.session_state.current_image, 
                        st.session_state.filtered_predictions,
                        "After Filtering"
                    )
                    if fig:
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        buf.seek(0)
                        st.image(buf, caption=f"Filtering: {len(st.session_state.filtered_predictions)} objects", width='content')
                        plt.close(fig)
                else:
                    st.info("Run filtering")
            
            st.markdown("---")
            
            # Two columns with tools (without Keypoint Task)
            col_filter, col_react = st.columns(2)
            
            with col_filter:
                st.subheader("🔧 Filtering")
                
                st.slider("Text Threshold", 0.0, 1.0, 0.05, 0.01, key="filter_text_threshold")
                st.checkbox("Use Similar Prompting", True, key="use_similar_prompting")
                st.slider("IoU Threshold", 0.0, 1.0, 0.9, 0.05, key="filter_iou_threshold")
                st.slider("Max Lower Bound", 0.0, 2.0, 1.0, 0.1, key="max_lower_bound")
                st.slider("Min Lower Bound", 0.0, 1.0, 0.0, 0.1, key="min_lower_bound")
                
                if st.button("🔍 Apply Filtering", width='stretch'):
                    with st.spinner("Applying filtering..."):
                        run_filter()
                        st.rerun()
            
            with col_react:
                st.subheader("🔄 ReAct")
                
                st.number_input("Number of Repeats", 1, 10, 2, key="number_repeats")
                st.slider("ReAct IoU Threshold", 0.0, 1.0, 0.9, 0.05, key="react_iou_threshold")
                
                if st.button("⚡ Run ReAct", width='stretch'):
                    with st.spinner("Running ReAct..."):
                        run_react()
                        st.rerun()
            
            # Show statistics
            st.markdown("---")
            
            if st.session_state.autolabel and st.session_state.autolabel.stats:
                st.subheader("📊 Statistics")
                
                stats = st.session_state.autolabel.stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Images", stats.get('total_images', 0))
                    st.metric("Successful Inferences", stats.get('successful_inferences', 0))
                
                with col2:
                    st.metric("Total Predictions", stats.get('total_predictions', 0))
                    st.metric("Filtered Predictions", stats.get('total_filtered_predictions', 0))
                
                with col3:
                    st.metric("Visual Prompting", stats.get('visual_prompting_predictions', 0))
                    st.metric("NMS Filtered", stats.get('nms_filtered', 0))
                
                with col4:
                    if st.session_state.raw_predictions:
                        st.metric("Current Predictions", len(st.session_state.raw_predictions))
                    if st.session_state.filtered_predictions:
                        st.metric("After Filtering", len(st.session_state.filtered_predictions))
            
            # Keypoint Task - below statistics
            st.markdown("---")
            st.subheader("📍 Keypoint Task")
            
            col_keypoint1, col_keypoint2, col_keypoint3 = st.columns(3)
            
            with col_keypoint1:
                st.selectbox(
                    "Task Type",
                    ["human_pose", "animal_pose", "face_keypoint"],
                    key="keypoint_task_type"
                )
                
                st.text_area(
                    "System Prompt",
                    value="You are a pose estimation assistant. Detect people and their keypoints accurately.",
                    height=80,
                    key="keypoint_system_prompt"
                )
            
            with col_keypoint2:
                st.text_area(
                    "Task Prompt",
                    value="Can you detect each {categories} in the image using a [x0, y0, x1, y1] box format, and then provide the coordinates of its {keypoints} as [x0, y0]? Output the answer in JSON format.",
                    height=120,
                    key="keypoint_task_prompt"
                )
            
            with col_keypoint3:
                st.markdown("<br><br>", unsafe_allow_html=True)  # Spacing
                if st.button("🎯 Run Keypoint Task", type="primary", width='stretch'):
                    with st.spinner("Running Keypoint Task..."):
                        run_keypoint_task()
                        st.rerun()
                
                # Result information
                if st.session_state.keypoint_result:
                    if st.session_state.keypoint_result.get('success'):
                        kp_result = st.session_state.keypoint_result.get('keypoint_result', {})
                        st.success(f"✅ Successfully processed: {kp_result.get('success_count', 0)} objects")
                        
                        # Additional information
                        if 'keypoint_names' in kp_result:
                            st.info(f"Keypoint names: {', '.join(kp_result['keypoint_names'][:5])}{'...' if len(kp_result['keypoint_names']) > 5 else ''}")
                    else:
                        st.error("❌ Keypoint task failed")
            
            # Display Keypoint Task result
            if st.session_state.keypoint_result and st.session_state.keypoint_result.get('success'):
                st.markdown("---")
                st.subheader("📊 Keypoint Task Result")
                
                kp_result = st.session_state.keypoint_result.get('keypoint_result', {})
                
                col_kp1, col_kp2 = st.columns([2, 1])
                
                with col_kp1:
                    # Show image with keypoints if available
                    if st.session_state.filtered_predictions and st.session_state.autolabel:
                        try:
                            fig = st.session_state.autolabel.get_image_with_bboxes(
                                img_name=st.session_state.current_image_name,
                                show_filtered=True,
                                keypoint_result=st.session_state.keypoint_result
                            )
                            if fig:
                                buf = io.BytesIO()
                                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                                buf.seek(0)
                                st.image(buf, caption="Keypoints Result", width='content')
                                plt.close(fig)
                        except Exception as e:
                            st.warning(f"Failed to display keypoints: {e}")
                
                with col_kp2:
                    # Detailed keypoint information
                    st.subheader("Details")
                    
                    if 'keypoints' in kp_result and kp_result['keypoints']:
                        st.metric("Detected Objects", kp_result.get('success_count', 0))
                        st.metric("Keypoints per Object", len(kp_result.get('keypoint_names', [])))
                        
                        # Show example keypoints for first object
                        if kp_result['keypoints']:
                            first_person_kps = kp_result['keypoints'][0]
                            visible_kps = sum(1 for kp in first_person_kps if kp[0] > 0 and kp[1] > 0)
                            st.metric("Visible Points (1st object)", f"{visible_kps}/{len(first_person_kps)}")
                    
                    # JSON result for debugging
                    with st.expander("🔍 Show JSON Result"):
                        st.json(kp_result)
            
            # Debugging and export
            st.markdown("---")
            
            with st.expander("🔍 Debugging & Export"):
                col_debug1, col_debug2 = st.columns(2)
                
                with col_debug1:
                    if st.session_state.raw_predictions:
                        st.subheader("Raw Predictions")
                        st.json(st.session_state.raw_predictions[:5])  # Show only first 5
                        
                        if st.button("📥 Export Raw Predictions", width='stretch'):
                            predictions_json = json.dumps(st.session_state.raw_predictions, indent=2)
                            st.download_button(
                                label="Download as JSON",
                                data=predictions_json,
                                file_name=f"raw_predictions_{st.session_state.current_image_name}.json",
                                mime="application/json"
                            )
                
                with col_debug2:
                    if st.session_state.filtered_predictions:
                        st.subheader("Filtered Predictions")
                        st.json(st.session_state.filtered_predictions[:5])
                        
                        if st.button("📥 Export Filtered Predictions", width='stretch'):
                            predictions_json = json.dumps(st.session_state.filtered_predictions, indent=2)
                            st.download_button(
                                label="Download as JSON",
                                data=predictions_json,
                                file_name=f"filtered_predictions_{st.session_state.current_image_name}.json",
                                mime="application/json"
                            )
                
                # Reset button
                if st.button("🔄 Reset All Results", type="secondary", width='stretch'):
                    for key in ['raw_predictions', 'filtered_predictions', 'react_predictions', 'keypoint_result']:
                        if key in st.session_state:
                            st.session_state[key] = None
                    st.rerun()

# CSS styles for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)