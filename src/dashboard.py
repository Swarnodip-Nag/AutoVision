"""
AutoVision Streamlit Dashboard
Real-time defect detection with visual explanations
"""
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os
import time
import requests
import json
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, get_transforms
from src.gradcam import GradCAM, overlay_heatmap_on_image
from src.preprocess import overlay_heatmap

# Page configuration
st.set_page_config(
    page_title="AutoVision - Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = './models/resnet18_anomaly.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CLASS_COLORS = {
    'crazing': '#FF6B6B',
    'inclusion': '#4ECDC4',
    'patches': '#45B7D1',
    'pitted_surface': '#FFA07A',
    'rolled-in_scale': '#98D8C8',
    'scratches': '#F7DC6F'
}

# Load models with caching
@st.cache_resource
def load_pytorch_model():
    """Load PyTorch model"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.info("Please train the model first by running: python src/train.py")
        st.stop()
    
    model = get_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_gradcam_model():
    """Load Grad-CAM model"""
    return GradCAM(MODEL_PATH, target_layer='layer4', device=DEVICE)

@st.cache_resource
def load_image_transforms():
    """Load image transformations"""
    return get_transforms()

def predict_image(model, transform, image):
    """
    Predict defect class for image
    
    Args:
        model: PyTorch model
        transform: Image transformations
        image: PIL Image
    
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    # Transform image
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    pred_class = CLASS_NAMES[predicted_idx.item()]
    conf_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()
    
    return pred_class, conf_score, all_probs, input_tensor

def call_api_predict(image, api_url, use_gradcam=False):
    """Call FastAPI backend for prediction"""
    try:
        # Convert image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Prepare request
        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
        
        # Choose endpoint
        endpoint = f"{api_url}/predict/gradcam" if use_gradcam else f"{api_url}/predict"
        
        # Make request
        response = requests.post(endpoint, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
            
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# Main Application
# ============================================================================

# Header
st.markdown('<div class="main-header">🔍 AutoVision - Intelligent Defect Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/inspection.png", width=100)
    st.title("⚙️ Settings")
    
    # Mode selection
    st.subheader("🎯 Operation Mode")
    mode = st.radio(
        "Select Mode:",
        ["📤 Upload Image", "📷 Real-time Camera", "🌐 API Integration"],
        help="Choose how you want to use AutoVision"
    )
    
    st.markdown("---")
    
    # Inference settings
    st.subheader("🔧 Inference Settings")
    
    use_api = st.checkbox(
        "Use FastAPI Backend",
        value=False,
        help="Use remote API instead of local model"
    )
    
    if use_api:
        api_url = st.text_input("API URL", "http://localhost:8000")
        api_status = st.empty()
        
        # Check API health
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                api_status.success("✅ API Connected")
            else:
                api_status.error("❌ API Unavailable")
        except:
            api_status.error("❌ Cannot reach API")
    
    show_gradcam = st.checkbox(
        "Show Grad-CAM Visualization",
        value=True,
        help="Display visual explanation heatmap"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for accepting predictions"
    )
    
    st.markdown("---")
    
    # System info
    st.subheader("ℹ️ System Info")
    st.info(f"""
    **Device:** {DEVICE}
    
    **Model:** ResNet-18
    
    **Classes:** {len(CLASS_NAMES)}
    
    **Input Size:** 224x224
    """)
    
    # Class legend
    st.subheader("🎨 Defect Classes")
    for class_name in CLASS_NAMES:
        color = CLASS_COLORS.get(class_name, '#FFFFFF')
        st.markdown(
            f'<div style="background-color: {color}; padding: 5px; margin: 3px; '
            f'border-radius: 5px; color: black; font-weight: bold; text-align: center;">'
            f'{class_name}</div>',
            unsafe_allow_html=True
        )

# Main content area
if not use_api:
    # Load local models
    with st.spinner("🔄 Loading models..."):
        model = load_pytorch_model()
        transform = load_image_transforms()
        if show_gradcam:
            gradcam = load_gradcam_model()

# ============================================================================
# Mode: Upload Image
# ============================================================================
if mode == "📤 Upload Image":
    st.header("📤 Upload Image for Defect Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image of a steel surface to detect defects"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # Perform inference
        all_probs = None
        with st.spinner("🔍 Analyzing image..."):
            if use_api:
                # Use API
                result = call_api_predict(image, api_url, use_gradcam=show_gradcam)
                
                if "error" in result:
                    st.error(f"❌ API Error: {result['error']}")
                else:
                    pred_class = result.get("prediction", "Unknown")
                    conf_score = result.get("confidence", 0.0)
                    all_probs = result.get("all_probabilities", {})
                    
                    # Display results
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Prediction card
                        color = CLASS_COLORS.get(pred_class, '#FFFFFF')
                        st.markdown(
                            f'<div style="background-color: {color}; padding: 20px; '
                            f'border-radius: 10px; text-align: center; margin: 10px 0;">'
                            f'<h2 style="color: black; margin: 0;">{pred_class.upper()}</h2>'
                            f'<h3 style="color: black; margin: 5px 0;">{conf_score*100:.1f}% Confidence</h3>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        # Status
                        if conf_score >= confidence_threshold:
                            st.success(f"✅ High confidence detection!")
                        else:
                            st.warning(f"⚠️ Low confidence. Manual inspection recommended.")
                    
                    # Show Grad-CAM if available
                    if show_gradcam and "gradcam_image" in result:
                        st.subheader("🔥 Grad-CAM Visualization")
                        gradcam_img = base64.b64decode(result["gradcam_image"])
                        st.image(gradcam_img, use_container_width=True)
                        st.caption(result.get("explanation", "Visual explanation of prediction"))
                    
            else:
                # Use local model
                pred_class, conf_score, all_probs, input_tensor = predict_image(model, transform, image)
                
                # Display results
                with col2:
                    st.subheader("🎯 Prediction Results")
                    
                    # Prediction card
                    color = CLASS_COLORS.get(pred_class, '#FFFFFF')
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 20px; '
                        f'border-radius: 10px; text-align: center; margin: 10px 0;">'
                        f'<h2 style="color: black; margin: 0;">{pred_class.upper()}</h2>'
                        f'<h3 style="color: black; margin: 5px 0;">{conf_score*100:.1f}% Confidence</h3>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Status
                    if conf_score >= confidence_threshold:
                        st.success(f"✅ High confidence detection!")
                    else:
                        st.warning(f"⚠️ Low confidence. Manual inspection recommended.")
                
                # Show Grad-CAM
                if show_gradcam:
                    with st.spinner("Generating visual explanation..."):
                        heatmap = gradcam.generate(input_tensor)
                        
                        # Resize image to match heatmap
                        image_resized = image.resize((224, 224))
                        image_np = np.array(image_resized)
                        
                        # Create overlay
                        overlaid = overlay_heatmap_on_image(image_np, heatmap, alpha=0.4)
                        
                        st.subheader("🔥 Grad-CAM Visualization")
                        st.image(overlaid, use_container_width=True, caption="Red regions indicate areas that influenced the prediction")
        
        # Show probability distribution
        if all_probs is not None:
            st.subheader("📊 Probability Distribution")
            
            if isinstance(all_probs, dict):
                prob_data = all_probs
            else:
                prob_data = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, all_probs)}
            
            # Create bar chart
            import pandas as pd
            df = pd.DataFrame({
                'Class': list(prob_data.keys()),
                'Probability': [v * 100 for v in prob_data.values()]
            }).sort_values('Probability', ascending=True)
            
            st.bar_chart(df.set_index('Class'))
            
            # Detailed probabilities
            with st.expander("📋 Detailed Probabilities"):
                for class_name in CLASS_NAMES:
                    prob = prob_data.get(class_name, 0.0)
                    st.write(f"**{class_name}:** {prob*100:.2f}%")

# ============================================================================
# Mode: Real-time Camera
# ============================================================================
elif mode == "📷 Real-time Camera":
    st.header("📷 Real-time Camera Feed")
    
    st.warning("⚠️ Camera mode requires webcam access. Click 'Start' to begin.")
    
    camera_enabled = st.checkbox("Enable Camera")
    
    if camera_enabled:
        # Camera feed placeholder
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        prob_placeholder = st.empty()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera!")
        else:
            st.success("✅ Camera connected")
            
            # FPS counter
            fps = 0
            frame_count = 0
            start_time = time.time()
            
            stop_button = st.button("Stop Camera")
            
            while not stop_button:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ Failed to read frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Predict
                image_pil = Image.fromarray(frame_rgb)
                pred_class, conf_score, all_probs, input_tensor = predict_image(model, transform, image_pil)
                
                # Generate Grad-CAM if enabled
                if show_gradcam:
                    heatmap = gradcam.generate(input_tensor)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    overlaid = overlay_heatmap_on_image(frame_resized, heatmap, alpha=0.3)
                    display_frame = cv2.resize(overlaid, (frame.shape[1], frame.shape[0]))
                else:
                    display_frame = frame_rgb
                
                # Add text overlay
                color = CLASS_COLORS.get(pred_class, '#FFFFFF')
                cv2.putText(display_frame, f"{pred_class}: {conf_score*100:.1f}%", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display
                frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                
                # Update results
                result_placeholder.metric(
                    label="Detected Defect",
                    value=pred_class.upper(),
                    delta=f"{conf_score*100:.1f}% confidence"
                )
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps = frame_count / (time.time() - start_time)
                
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
            st.info("Camera stopped")

# ============================================================================
# Mode: API Integration
# ============================================================================
elif mode == "🌐 API Integration":
    st.header("🌐 API Integration Guide")
    
    st.markdown("""
    ### 🚀 Using AutoVision API
    
    The AutoVision API provides REST endpoints for integrating defect detection into your applications.
    
    #### 📍 Base URL
    ```
    http://localhost:8000
    ```
    
    #### 🔗 Available Endpoints
    """)
    
    # Endpoint: Predict
    with st.expander("POST /predict - Basic Prediction", expanded=True):
        st.markdown("""
        **Description:** Upload an image and get defect classification
        
        **Request:**
        ```python
        import requests
        
        url = "http://localhost:8000/predict"
        files = {"file": open("image.jpg", "rb")}
        response = requests.post(url, files=files)
        result = response.json()
        ```
        
        **Response:**
        ```json
        {
            "success": true,
            "prediction": "crazing",
            "confidence": 0.9234,
            "class_id": 0,
            "all_probabilities": {...},
            "top_3_predictions": [...]
        }
        ```
        """)
        
        # Live test
        st.subheader("🧪 Test Endpoint")
        test_file = st.file_uploader("Upload test image", type=['jpg', 'jpeg', 'png'], key="api_test1")
        
        if test_file and st.button("Test /predict"):
            with st.spinner("Calling API..."):
                image = Image.open(test_file)
                result = call_api_predict(image, "http://localhost:8000", use_gradcam=False)
                st.json(result)
    
    # Endpoint: Predict with Grad-CAM
    with st.expander("POST /predict/gradcam - Prediction with Visual Explanation"):
        st.markdown("""
        **Description:** Get prediction with Grad-CAM heatmap overlay
        
        **Request:**
        ```python
        url = "http://localhost:8000/predict/gradcam"
        files = {"file": open("image.jpg", "rb")}
        response = requests.post(url, files=files)
        result = response.json()
        
        # Decode base64 image
        import base64
        img_data = base64.b64decode(result["gradcam_image"])
        ```
        
        **Response:**
        ```json
        {
            "success": true,
            "prediction": "inclusion",
            "confidence": 0.8863,
            "gradcam_image": "base64_encoded_image...",
            "explanation": "..."
        }
        ```
        """)
        
        # Live test
        st.subheader("🧪 Test Endpoint")
        test_file2 = st.file_uploader("Upload test image", type=['jpg', 'jpeg', 'png'], key="api_test2")
        
        if test_file2 and st.button("Test /predict/gradcam"):
            with st.spinner("Calling API..."):
                image = Image.open(test_file2)
                result = call_api_predict(image, "http://localhost:8000", use_gradcam=True)
                
                if "error" not in result:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original", use_container_width=True)
                    with col2:
                        if "gradcam_image" in result:
                            gradcam_img = base64.b64decode(result["gradcam_image"])
                            st.image(gradcam_img, caption="Grad-CAM Overlay", use_container_width=True)
                    
                    st.json(result)
                else:
                    st.error(result["error"])
    
    # Code examples
    st.markdown("---")
    st.subheader("💻 Integration Examples")
    
    tab1, tab2, tab3 = st.tabs(["Python", "cURL", "JavaScript"])
    
    with tab1:
        st.code("""
import requests

def detect_defect(image_path):
    url = "http://localhost:8000/predict/gradcam"
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Defect: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
result = detect_defect("surface_image.jpg")
        """, language="python")
    
    with tab2:
        st.code("""
curl -X POST "http://localhost:8000/predict" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@surface_image.jpg"
        """, language="bash")
    
    with tab3:
        st.code("""
async function detectDefect(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    console.log('Defect:', result.prediction);
    console.log('Confidence:', result.confidence);
    
    return result;
}
        """, language="javascript")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <strong>AutoVision</strong> - Intelligent Visual Defect Detection System<br>
    Powered by PyTorch & Streamlit | Model: ResNet-18 | Dataset: NEU-DET
</div>
""", unsafe_allow_html=True)