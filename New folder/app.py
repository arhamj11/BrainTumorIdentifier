from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from typing import List
import os

app = FastAPI()

# Add CORS middleware to allow requests from your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's default URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved model
try:
    model = tf.keras.models.load_model('brain_tumor_cnn.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

async def process_image(image_file: UploadFile) -> dict:
    try:
        # Read and validate the image
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to grayscale if it isn't already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize image to match model's expected input size
        image = image.resize((128, 128))
        
        # Convert to numpy array and preprocess
        image_array = np.array(image)
        image_array = image_array / 255.0  # Normalize pixel values
        
        # Add batch and channel dimensions
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
        
        # Make prediction
        prediction = model.predict(image_array)
        confidence = float(prediction[0][0])
        
        return {
            "hasTumor": bool(confidence > 0.5),
            "confidence": confidence,
            "fileName": image_file.filename
        }
        
    except Exception as e:
        print(f"Error processing image {image_file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "alive", "model_loaded": model is not None}

@app.post("/predict")
async def predict_images(images: List[UploadFile] = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    results = []
    for image in images:
        try:
            # Validate file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {image.filename} is not an image"
                )
            
            result = await process_image(image)
            results.append(result)
            
        except Exception as e:
            print(f"Error processing image {image.filename}: {e}")
            results.append({
                "hasTumor": False,
                "confidence": 0.0,
                "error": str(e),
                "fileName": image.filename
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)