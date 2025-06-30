import tensorflow as tf
import os

# Define the path to your original Keras .h5 model
keras_model_path = os.path.join('saved_model', 'my_emotion_model.h5')

# Define the output path for your TFLite model
tflite_model_path = os.path.join('saved_model', 'my_emotion_model.tflite')

print(f"Loading Keras model from: {keras_model_path}")
try:
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)
    print("Keras model loaded successfully.")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Apply optimizations for smaller size and faster inference
    # This can sometimes reduce accuracy, but is good for deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("Model conversion to TFLite successful.")

    # Save the TFLite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_model_path}")

    # You can also check the size difference
    original_size = os.path.getsize(keras_model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
    print(f"Original model size: {original_size:.2f} MB")
    print(f"TFLite model size: {tflite_size:.2f} MB")

except Exception as e:
    print(f"Error during model conversion: {e}")