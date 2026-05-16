"""
TinyML Cosine Wave Predictor - Training Script
Assignment 3: IoT Applications Development (SWAPD 453)
Spring 2026
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

print("=" * 60)
print("TinyML Cosine Wave Predictor - Training Script")
print("=" * 60)

# ============================================================================
# 1. Data Generation
# ============================================================================
print("\n[1] Generating dataset...")

# Generate 1000 samples
n_samples = 1000
x = np.random.uniform(0, 2 * np.pi, n_samples)
y = np.cos(x) + np.random.normal(0, 0.1, n_samples)

# Shuffle the data
indices = np.random.permutation(n_samples)
x = x[indices]
y = y[indices]

# Split into 60% training, 20% validation, 20% test
train_size = int(0.6 * n_samples)
val_size = int(0.2 * n_samples)

x_train = x[:train_size]
y_train = y[:train_size]
x_val = x[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
x_test = x[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"Training samples: {len(x_train)}")
print(f"Validation samples: {len(x_val)}")
print(f"Test samples: {len(x_test)}")

# ============================================================================
# 2. Model Definition
# ============================================================================
print("\n[2] Defining model architecture...")

model = keras.Sequential([
    layers.Dense(16, activation='relu', input_shape=(1,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Linear activation for regression
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ============================================================================
# 3. Model Training
# ============================================================================
print("\n[3] Training model...")

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=32,
    verbose=1
)

# ============================================================================
# 4. Plot Training History
# ============================================================================
print("\n[4] Plotting training history...")

plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('plots/training_history.png', dpi=150)
print("Saved: plots/training_history.png")

# ============================================================================
# 5. Evaluate Float Model
# ============================================================================
print("\n[5] Evaluating float model on test set...")

test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
test_mse = test_loss

print(f"Test MSE: {test_mse:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# ============================================================================
# 6. Plot Float Model Predictions
# ============================================================================
print("\n[6] Plotting float model predictions...")

# Generate dense grid for smooth plotting
x_dense = np.linspace(0, 2 * np.pi, 200)
y_true = np.cos(x_dense)
y_pred_float = model.predict(x_dense, verbose=0).flatten()

plt.figure(figsize=(10, 6))
plt.plot(x_dense, y_true, 'b-', label='Ground Truth (cos(x))', linewidth=2)
plt.plot(x_dense, y_pred_float, 'r--', label='Float Model Prediction', linewidth=2, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Float Model: Predicted vs Ground Truth Cosine Wave')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/float_model_prediction.png', dpi=150)
print("Saved: plots/float_model_prediction.png")

# ============================================================================
# 7. Convert to TFLite (Float)
# ============================================================================
print("\n[7] Converting to TFLite (float)...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_float_model = converter.convert()

float_model_size = len(tflite_float_model)
print(f"Float TFLite model size: {float_model_size} bytes ({float_model_size / 1024:.2f} KB)")

with open('cosine_float.tflite', 'wb') as f:
    f.write(tflite_float_model)
print("Saved: cosine_float.tflite")

# ============================================================================
# 8. Full Integer Quantization
# ============================================================================
print("\n[8] Converting to TFLite with full int8 quantization...")

def representative_dataset():
    for i in range(100):
        # Sample from training distribution
        sample = x_train[i % len(x_train)].reshape(1, 1).astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_int8_model = converter.convert()

int8_model_size = len(tflite_int8_model)
print(f"Int8 TFLite model size: {int8_model_size} bytes ({int8_model_size / 1024:.2f} KB)")
print(f"Size reduction: {(1 - int8_model_size / float_model_size) * 100:.2f}%")

with open('cosine_int8.tflite', 'wb') as f:
    f.write(tflite_int8_model)
print("Saved: cosine_int8.tflite")

# ============================================================================
# 9. Evaluate Quantized Model
# ============================================================================
print("\n[9] Evaluating quantized model...")

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='cosine_int8.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input quantization: scale={input_details[0]['quantization'][0]:.6f}, zero_point={input_details[0]['quantization'][1]}")
print(f"Output quantization: scale={output_details[0]['quantization'][0]:.6f}, zero_point={output_details[0]['quantization'][1]}")

# Function to quantize and dequantize
def quantize(x, scale, zero_point):
    return np.clip(np.round(x / scale + zero_point), -128, 127).astype(np.int8)

def dequantize(x, scale, zero_point):
    return (x.astype(np.float32) - zero_point) * scale

# Evaluate on test set
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

predictions_int8 = []
for x_val in x_test:
    # Quantize input
    x_quant = quantize(x_val, input_scale, input_zero_point).reshape(1, 1)
    
    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], x_quant)
    interpreter.invoke()
    
    # Get and dequantize output
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred = dequantize(output, output_scale, output_zero_point)
    predictions_int8.append(y_pred[0][0])

predictions_int8 = np.array(predictions_int8)

# Calculate metrics
int8_mse = np.mean((predictions_int8 - y_test) ** 2)
int8_mae = np.mean(np.abs(predictions_int8 - y_test))

print(f"Int8 Model Test MSE: {int8_mse:.6f}")
print(f"Int8 Model Test MAE: {int8_mae:.6f}")

# ============================================================================
# 10. Plot Comparison: Float vs Int8 vs Ground Truth
# ============================================================================
print("\n[10] Plotting comparison of all models...")

# Generate predictions on dense grid
y_pred_int8_dense = []
for x_val in x_dense:
    x_quant = quantize(x_val, input_scale, input_zero_point).reshape(1, 1)
    interpreter.set_tensor(input_details[0]['index'], x_quant)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred = dequantize(output, output_scale, output_zero_point)
    y_pred_int8_dense.append(y_pred[0][0])

y_pred_int8_dense = np.array(y_pred_int8_dense)

plt.figure(figsize=(12, 6))
plt.plot(x_dense, y_true, 'b-', label='Ground Truth (cos(x))', linewidth=3)
plt.plot(x_dense, y_pred_float, 'r--', label='Float Model', linewidth=2, alpha=0.8)
plt.plot(x_dense, y_pred_int8_dense, 'g:', label='Int8 Quantized Model', linewidth=2, alpha=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Comparison: Float vs Int8 Quantized vs Ground Truth')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150)
print("Saved: plots/model_comparison.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nFloat Model:")
print(f"  - Test MSE: {test_mse:.6f}")
print(f"  - Test MAE: {test_mae:.6f}")
print(f"  - Model size: {float_model_size} bytes ({float_model_size / 1024:.2f} KB)")

print(f"\nInt8 Quantized Model:")
print(f"  - Test MSE: {int8_mse:.6f}")
print(f"  - Test MAE: {int8_mae:.6f}")
print(f"  - Model size: {int8_model_size} bytes ({int8_model_size / 1024:.2f} KB)")
print(f"  - Size reduction: {(1 - int8_model_size / float_model_size) * 100:.2f}%")

print(f"\nQuantization Parameters:")
print(f"  - Input scale: {input_scale:.6f}, zero_point: {input_zero_point}")
print(f"  - Output scale: {output_scale:.6f}, zero_point: {output_zero_point}")

print("\n" + "=" * 60)
print("Training complete! All files saved successfully.")
print("=" * 60)
