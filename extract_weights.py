import tensorflow as tf
import numpy as np

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path='cosine_predictor/cosine_int8.tflite')
interpreter.allocate_tensors()

# Get model details
details = interpreter.get_tensor_details()
print('Model tensors:')
for i, detail in enumerate(details):
    print(f'{i}: {detail["name"]}, shape: {detail["shape"]}, dtype: {detail["dtype"]}')
    if 'quantization' in detail:
        print(f'   Quantization: {detail["quantization"]}')
