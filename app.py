import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import base64

app = Flask(__name__)

# --- IMPORTANT: CONFIGURE THESE ---
# Make sure this path is correct relative to app.py
MODEL_PATH = os.path.join('saved_model', 'my_emotion_model.tflite')

# --- VERY IMPORTANT: Update EMOTION_LABELS ---
# Based on our troubleshooting, your model outputs 3 classes.
# You MUST replace these with the actual 3 emotion labels your friend's model was trained on.
# Examples might be: ['Happy', 'Sad', 'Neutral'], or ['Positive', 'Negative', 'Neutral']
EMOTION_LABELS = ['Class_1', 'Class_2', 'Class_3'] # <--- *** UPDATE THESE 3 LABELS ***

# --- Model Input Configuration (Derived from original training code) ---
MODEL_INPUT_SIZE = (140, 140) # Model expects 140x140 images
MODEL_INPUT_CHANNELS = 3 # Xception base expects 3 color channels (RGB/BGR)
# --- END CONFIGURATION ---

# Initialize interpreter and details globally
interpreter = None
input_details = None
output_details = None
model_loaded_successfully = False # Flag to track model loading status

# Load the TFLite model once when the app starts
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Optional: Verify expected input shape from the loaded TFLite model
    expected_input_shape_from_tflite = input_details[0]['shape']
    print(f"TFLite Model loaded successfully! Expected input shape from TFLite: {expected_input_shape_from_tflite}")

    # You can add a check to ensure it matches your constant for safety
    if (expected_input_shape_from_tflite[1], expected_input_shape_from_tflite[2]) != MODEL_INPUT_SIZE or \
       expected_input_shape_from_tflite[3] != MODEL_INPUT_CHANNELS:
        print(f"WARNING: Model input shape mismatch! Expected {MODEL_INPUT_SIZE + (MODEL_INPUT_CHANNELS,)}, but TFLite model expects {expected_input_shape_from_tflite[1:]}. This might cause errors.")

    model_loaded_successfully = True
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    model_loaded_successfully = False

@app.route('/')
def index():
    """Serves the main page of the application."""
    return render_template('index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Endpoint to analyze an uploaded image or webcam frame."""
    if not model_loaded_successfully:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']
    # Remove the data URL header (e.g., 'data:image/jpeg;base64,')
    header, encoded = image_data.split(",", 1)

    try:
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Ensure IMREAD_COLOR for 3 channels

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # --- IMPORTANT: Pre-process the image for your model ---
        # 1. Resize to 140x140 (MODEL_INPUT_SIZE)
        resized_img = cv2.resize(img, MODEL_INPUT_SIZE) 

        # 2. Convert to float32
        input_img = resized_img.astype(np.float32)

        # 3. Normalize pixel values to -1 to 1 (as per Xception's preprocessing)
        input_img = (input_img / 127.5) - 1 

        # 4. Add batch dimension (model expects batch_size, height, width, channels)
        input_img = np.expand_dims(input_img, axis=0) # Shape will be (1, 140, 140, 3)

        # --- TFLite Prediction ---
        interpreter.set_tensor(input_details[0]['index'], input_img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # Make prediction
        emotion_index = np.argmax(predictions[0])
        
        # Ensure the index is within the bounds of your EMOTION_LABELS
        if 0 <= emotion_index < len(EMOTION_LABELS):
            predicted_emotion = EMOTION_LABELS[emotion_index]
        else:
            predicted_emotion = f"Unknown (Index: {emotion_index})"
            print(f"Warning: Predicted emotion index {emotion_index} out of bounds for EMOTION_LABELS.")

        # Also return probabilities for debugging/more info if needed
        return jsonify({
            "emotion": predicted_emotion,
            "probabilities": predictions[0].tolist()
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable, default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Listen on all network interfaces for deployment
    app.run(host='0.0.0.0', port=port, debug=False)