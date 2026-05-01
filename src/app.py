"""
AutoVision FastAPI Backend
Real-time defect detection API with explainability
"""
import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import cv2
import base64
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model, get_transforms
from src.gradcam import GradCAM
from src.preprocess import overlay_heatmap

# Configuration
MODEL_PATH = './models/resnet18_anomaly.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# Global model instances (loaded once at startup)
model = None
gradcam = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global model, gradcam, transform

    print("=" * 80)
    print("🚀 AutoVision API Starting Up...")
    print("=" * 80)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please train the model first by running: python src/train.py")
    else:
        print(f"📦 Loading PyTorch model from: {MODEL_PATH}")
        model = get_model(pretrained=False).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"✓ Model loaded successfully on {DEVICE}")

        print("🔍 Initializing Grad-CAM for explainability...")
        gradcam = GradCAM(MODEL_PATH, target_layer='layer4')
        print("✓ Grad-CAM initialized")

        transform = get_transforms()
        print("✓ Image transforms ready")

        print("\n" + "=" * 80)
        print("✅ AutoVision API Ready!")
        print("📊 Supported defect classes:", CLASS_NAMES)
        print("🌐 API Documentation: http://localhost:8000/docs")
        print("=" * 80 + "\n")

    yield

# Initialize FastAPI
app = FastAPI(
    title="AutoVision API",
    description="Intelligent Visual Defect Detection System for Manufacturing Quality Control",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AutoVision API",
        "version": "1.0.0",
        "description": "Intelligent Visual Defect Detection System",
        "status": "operational",
        "device": str(DEVICE),
        "classes": CLASS_NAMES,
        "endpoints": {
            "predict": "/predict - Upload image for defect detection",
            "predict_with_gradcam": "/predict/gradcam - Get prediction with visual explanation",
            "health": "/health - Check API health status",
            "info": "/info - Get model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gradcam_loaded": gradcam is not None,
        "device": str(DEVICE)
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "ResNet-18",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "input_size": [224, 224],
        "device": str(DEVICE),
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict defect class for uploaded image
    
    Returns:
        - prediction: Detected defect class
        - confidence: Prediction confidence (0-1)
        - class_id: Numeric class ID
        - all_probabilities: Probabilities for all classes
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        pred_class = CLASS_NAMES[predicted_idx.item()]
        conf_score = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy().tolist()
        prob_dict = {class_name: float(prob) for class_name, prob in zip(CLASS_NAMES, all_probs)}
        
        # Sort probabilities
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        return JSONResponse({
            "success": True,
            "prediction": pred_class,
            "confidence": float(conf_score),
            "class_id": int(predicted_idx.item()),
            "all_probabilities": prob_dict,
            "top_3_predictions": [
                {"class": cls, "probability": prob} 
                for cls, prob in sorted_probs[:3]
            ]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/gradcam")
async def predict_with_gradcam(
    file: UploadFile = File(...),
    return_image: bool = True
):
    """
    Predict defect class with Grad-CAM visual explanation
    
    Args:
        file: Image file
        return_image: If True, returns base64 encoded heatmap overlay image
    
    Returns:
        - prediction: Detected defect class
        - confidence: Prediction confidence
        - gradcam_heatmap: Base64 encoded heatmap overlay (if return_image=True)
        - explanation: Text explanation of the prediction
    """
    if model is None or gradcam is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        pred_class = CLASS_NAMES[predicted_idx.item()]
        conf_score = confidence.item()
        
        # Generate Grad-CAM heatmap
        print(f"Generating Grad-CAM for prediction: {pred_class}")
        heatmap = gradcam.generate(input_tensor, class_idx=predicted_idx.item())
        
        # Prepare response
        response_data = {
            "success": True,
            "prediction": pred_class,
            "confidence": float(conf_score),
            "class_id": int(predicted_idx.item()),
            "explanation": f"The model predicts '{pred_class}' with {conf_score*100:.1f}% confidence. "
                          f"The heatmap highlights the regions that contributed most to this prediction."
        }
        
        # Create overlay image if requested
        if return_image:
            # Resize original image to 224x224 to match heatmap
            image_resized = cv2.resize(image_np, (224, 224))
            
            # Create overlay
            overlaid = overlay_heatmap(image_resized, heatmap, alpha=0.4)
            
            # Convert to base64
            success, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            if success:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data["gradcam_image"] = img_base64
                response_data["image_format"] = "base64_jpeg"
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")

@app.post("/batch/predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple images
    
    Returns list of predictions for all uploaded images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for idx, file in enumerate(files):
        try:
            # Read and preprocess image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Transform image
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            results.append({
                "filename": file.filename,
                "prediction": CLASS_NAMES[predicted_idx.item()],
                "confidence": float(confidence.item()),
                "class_id": int(predicted_idx.item())
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return JSONResponse({
        "success": True,
        "total_images": len(files),
        "results": results
    })

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting AutoVision API Server...")
    print("📍 Access API at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔄 Interactive API: http://localhost:8000/redoc\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)