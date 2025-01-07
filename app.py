from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your React frontend

# Load the saved model
try:
    model = tf.keras.models.load_model('brain_tumor_cnn.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def process_image(image_file):
    try:
        # Read and validate the image
        image = Image.open(io.BytesIO(image_file.read()))
        
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
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        raise ValueError(f"Error processing image: {str(e)}")

@app.route("/", methods=["GET"])
def root():
    """Health check endpoint"""
    return jsonify({"status": "alive", "model_loaded": model is not None})

@app.route("/predict", methods=["POST"])
def predict_images():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    results = []
    images = request.files.getlist("images")  # Support multiple images
    for image in images:
        try:
            # Validate file type
            if not image.content_type.startswith('image/'):
                results.append({
                    "hasTumor": False,
                    "confidence": 0.0,
                    "error": f"File {image.filename} is not an image"
                })
                continue
            
            result = process_image(image)
            result["fileName"] = image.filename
            results.append(result)
            
        except Exception as e:
            print(f"Error processing image {image.filename}: {e}")
            results.append({
                "hasTumor": False,
                "confidence": 0.0,
                "error": str(e),
                "fileName": image.filename
            })
    
    return jsonify(results)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000, debug=True)
