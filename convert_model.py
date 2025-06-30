import tensorflow as tf
import os
import json 

keras_weights_path = os.path.join('saved_model', 'my_emotion_model.h5') 
tflite_model_path = os.path.join('saved_model', 'my_emotion_model.tflite')

# --- Model Architecture Definition (Reconstructed from your training code) ---
# Parameters from your original code
img_size = (140, 140)
# CHANGE HERE: num_classes from 7 to 3
num_classes = 3 # The .h5 file indicates 3 output classes based on the error message

def build_model_architecture(input_shape, num_classes):
    base_model = tf.keras.applications.Xception(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers[:-80]: 
        layer.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'),
        tf.keras.layers.LeakyReLU(alpha=0.1), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01), kernel_initializer='he_normal'),
        tf.keras.layers.LeakyReLU(alpha=0.1), 
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='he_normal', dtype='float32')
    ])
    
    return model

# --- End Model Architecture Definition ---


print(f"Loading Keras weights from: {keras_weights_path}")
try:
    model = build_model_architecture(img_size + (3,), num_classes)
    print("Model architecture reconstructed successfully.")
    
    model.load_weights(keras_weights_path)
    print("Model weights loaded successfully.")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("Model conversion to TFLite successful.")

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_model_path}")

    if os.path.exists(keras_weights_path) and os.path.exists(tflite_model_path):
        weights_file_size = os.path.getsize(keras_weights_path) / (1024 * 1024)
        tflite_size = os.path.getsize(tflite_model_path) / (1024 * 1024)
        print(f"Original .h5 weights file size: {weights_file_size:.2f} MB")
        print(f"TFLite model size: {tflite_size:.2f} MB")
    else:
        print("Could not get file sizes (one or both files not found).")

except Exception as e:
    print(f"Error during model loading or conversion: {e}")
    print("\nPossible reasons for error:")
    print("1. Ensure 'my_emotion_model.h5' is correctly placed in the 'saved_model' directory.")
    print("2. The .h5 file might be corrupted or not solely a weights file.")
    print("3. There might be a subtle difference in the model architecture defined above and the one in the .h5 file.")
    print("4. Your tf210_env might still have a NumPy or other dependency conflict. Verify using:")
    print("   python -c \"import tensorflow as tf; import numpy as np; print(f'TF: {tf.__version__}, NumPy: {np.__version__}')\"")