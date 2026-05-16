import tensorflow as tf
import numpy as np

# Load the float model (easier to extract weights)
interpreter = tf.lite.Interpreter(model_path='cosine_predictor/cosine_float.tflite')
interpreter.allocate_tensors()

# Get model details
details = interpreter.get_tensor_details()

# Extract weights and biases
weights = {}
biases = {}

for detail in details:
    name = detail['name']
    try:
        tensor = interpreter.get_tensor(detail['index'])
        if 'MatMul' in name:
            weights[name] = tensor
        elif 'BiasAdd' in name or 'Add' in name:
            biases[name] = tensor
    except:
        pass

# Print extracted info
print("Weights:")
for name, w in weights.items():
    print(f"{name}: shape {w.shape}, dtype {w.dtype}")
    
print("\nBiases:")
for name, b in biases.items():
    print(f"{name}: shape {b.shape}, dtype {b.dtype}")

# Save as C header
with open('cosine_predictor/weights.h', 'w') as f:
    f.write("#ifndef WEIGHTS_H\n")
    f.write("#define WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")
    
    # Layer 1 weights (16x1)
    w1 = weights['sequential_1/dense_1/MatMul'].flatten()
    f.write(f"const float layer1_weights[{len(w1)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in w1]))
    f.write("};\n\n")
    
    # Layer 1 bias (16)
    b1 = biases['sequential_1/dense_1/Relu;sequential_1/dense_1/BiasAdd'].flatten()
    f.write(f"const float layer1_bias[{len(b1)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in b1]))
    f.write("};\n\n")
    
    # Layer 2 weights (16x16)
    w2 = weights['sequential_1/dense_1_2/MatMul'].flatten()
    f.write(f"const float layer2_weights[{len(w2)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in w2]))
    f.write("};\n\n")
    
    # Layer 2 bias (16)
    b2 = biases['sequential_1/dense_1_2/Relu;sequential_1/dense_1_2/BiasAdd'].flatten()
    f.write(f"const float layer2_bias[{len(b2)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in b2]))
    f.write("};\n\n")
    
    # Layer 3 weights (1x16)
    w3 = weights['sequential_1/dense_2_1/MatMul'].flatten()
    f.write(f"const float layer3_weights[{len(w3)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in w3]))
    f.write("};\n\n")
    
    # Layer 3 bias (1)
    b3 = biases['sequential_1/dense_2_1/Add'].flatten()
    f.write(f"const float layer3_bias[{len(b3)}] = {{")
    f.write(','.join([f"{x:.6f}f" for x in b3]))
    f.write("};\n\n")
    
    f.write("#endif // WEIGHTS_H\n")

print("\nSaved weights to cosine_predictor/weights.h")
