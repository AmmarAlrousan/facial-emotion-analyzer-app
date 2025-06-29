import os
from flask import Flask, render_template, request, jsonify
import tensorflow as tf # <-- CHANGED: Import tensorflow for TFLite Interpreter
import cv2
import numpy as np
import base64

app = Flask(__name__)

# --- IMPORTANT: CONFIGURE THESE ---
# Make sure this path is correct relative to app.py
MODEL_PATH = os.path.join('saved_model', 'my_emotion_model.tflite')
# Make sure this order matches the output classes of your trained model
EMOTION_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']
# --- END CONFIGURATION ---

# Initialize interpreter and details globally, outside of try-except for cleaner access
interpreter = None
input_details = None
output_details = None

# Load the TFLite model once when the app starts
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("TFLite Model loaded successfully!")
    # No need to assign interpreter to 'model' if you use 'interpreter' directly below
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    # interpreter, input_details, output_details remain None if loading fails

@app.route('/')
def index():
    """Serves the main page of the application."""
    return render_template('index.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Endpoint to analyze an uploaded image or webcam frame."""
    # Check if the interpreter was loaded successfully
    if interpreter is None:
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
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # --- IMPORTANT: Pre-process the image for your model ---
        # Adjust these steps (grayscale, resize, normalize, reshape)
        # to precisely match your model's input requirements (e.g., 48x48, 1 channel)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (48, 48)) # Example: resize to 48x48 pixels
        normalized_img = resized_img / 255.0 # Normalize pixel values to 0-1

        # Reshape for model input (batch_size, height, width, channels)
        # If your model expects a single grayscale channel (common for emotion detection):
        input_img = np.expand_dims(normalized_img, axis=-1) # Add channel dimension
        input_img = np.expand_dims(input_img, axis=0)       # Add batch dimension

        # Ensure the input_img has the correct data type for the TFLite model
        # Get expected input type from input_details
        input_dtype = input_details[0]['dtype']
        input_img = input_img.astype(input_dtype)

        # --- TFLite Prediction ---
        # Set the tensor
        interpreter.set_tensor(input_details[0]['index'], input_img)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])

        # --- IMPORTANT: Pre-process the image for your model ---
        # If your model expects 3 RGB channels (less common for emotion detection from grayscale):
        # input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
        # input_img = cv2.resize(input_img, (YOUR_MODEL_WIDTH, YOUR_MODEL_HEIGHT))
        # input_img = input_img / 255.0
        # input_img = np.expand_dims(input_img, axis=0)
        # --- END IMPORTANT ---


        # Make prediction
        emotion_index = np.argmax(predictions[0])
        predicted_emotion = EMOTION_LABELS[emotion_index]

        return jsonify({"emotion": predicted_emotion})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Get port from environment variable, default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Listen on all network interfaces for deployment
    app.run(host='0.0.0.0', port=port, debug=False)