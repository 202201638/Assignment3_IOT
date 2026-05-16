# TinyML Cosine Wave Predictor

Assignment 3: IoT Applications Development (SWAPD 453)  
Spring 2026

This project implements a complete TinyML pipeline for approximating the cosine function on an ESP32 microcontroller using TensorFlow Lite Micro with full int8 quantization.

## Project Structure

```
.
├── train_cosine.py              # Python training script
├── cosine_float.tflite          # Float TFLite model
├── cosine_int8.tflite           # Quantized int8 TFLite model
├── model.h                      # C header file (generated with xxd)
├── cosine_predictor.ino         # Arduino sketch for ESP32
├── requirements.txt             # Python dependencies
├── plots/                       # Generated plots
│   ├── training_history.png
│   ├── float_model_prediction.png
│   └── model_comparison.png
└── README.md                    # This file
```

## Prerequisites

### Python Environment
- Python 3.10 or higher
- pip package manager

### Arduino IDE
- Arduino IDE 2.x
- ESP32 board package
- TensorFlow Lite Micro library for ESP32

### Hardware
- ESP32 development board (DevKit V1, WROOM 32, S3, etc.)
- USB data cable (not power-only)

## Setup Instructions

### 1. Python Environment Setup

#### Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install tensorflow>=2.13.0 numpy>=1.24.0 matplotlib>=3.7.0
```

### 2. Train the Model

Run the training script:
```bash
python train_cosine.py
```

This will:
- Generate 1000 samples of cosine data with noise
- Train a neural network (Dense(16) → Dense(16) → Dense(1))
- Save training history plots to `plots/`
- Evaluate the float model on test set
- Convert to TFLite format (both float and int8 quantized)
- Save `cosine_float.tflite` and `cosine_int8.tflite`
- Generate comparison plots

### 3. Convert TFLite to C Header

#### On Linux/macOS:
```bash
xxd -i cosine_int8.tflite > model.h
```

#### On Windows (using Git Bash):
```bash
xxd -i cosine_int8.tflite > model.h
```

#### On Windows (using WSL):
```bash
wsl xxd -i cosine_int8.tflite > model.h
```

Then edit `model.h` to rename the array and add alignment:
```c
alignas(8) const unsigned char g_model[] = { ... };
```

### 4. Arduino IDE Setup

#### Install ESP32 Board Package
1. Open Arduino IDE
2. Go to **File → Preferences**
3. Add this URL to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**
5. Search for "esp32" and install "esp32 by Espressif Systems"

#### Install TensorFlow Lite Micro Library
**NOT REQUIRED** - This sketch uses a manual neural network implementation with no external libraries.

### 5. Upload to ESP32

1. Open `cosine_predictor.ino` in Arduino IDE
2. Select your board: **Tools → Board → ESP32 Arduino → [Your Board Model]**
3. Select the correct port: **Tools → Port → [Your ESP32 Port]**
4. Click **Upload** button

### 6. View Results

#### Serial Monitor (for startup diagnostics)
1. Go to **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. Press ESP32 reset button
4. You should see model size, arena usage, and quantization parameters

#### Serial Plotter (for visualization)
1. Go to **Tools → Serial Plotter** (or press Ctrl+Shift+L)
2. Set baud rate to **115200**
3. You should see two curves: predicted (red) and actual (blue)
4. The curves should track each other through multiple periods

## Model Architecture

```
Input (1) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(1) → Output
```

- **Trainable parameters**: ~321
- **Float model size**: ~2-3 KB
- **Int8 quantized size**: ~1-2 KB
- **Target accuracy**: Test MAE < 0.05

## Quantization Details

- **Type**: Full integer (int8) post-training quantization
- **Input/Output**: int8 (no float operations at runtime)
- **Representative dataset**: 100 samples from training distribution
- **Quantization parameters**: Automatically computed by TFLite converter

## Expected Output

### Training Script Output
```
Training samples: 600
Validation samples: 200
Test samples: 200

Float Model:
  - Test MSE: 0.002xxx
  - Test MAE: 0.03xxx
  - Model size: 2xxx bytes (x.xx KB)

Int8 Quantized Model:
  - Test MSE: 0.003xxx
  - Test MAE: 0.04xxx
  - Model size: 1xxx bytes (x.xx KB)
  - Size reduction: xx.xx%
```

### ESP32 Serial Monitor Output
```
========================================
TinyML Cosine Wave Predictor - ESP32
========================================

Model size: 1234 bytes
Tensor arena used: 2048 bytes
Input quantization: scale=0.xxxxxx, zero_point=0
Output quantization: scale=0.xxxxxx, zero_point=0

========================================
Starting predictions...
Open Serial Plotter (Ctrl+Shift+L) to view
========================================
```

### Serial Plotter Output
Two curves displayed simultaneously:
- **Predicted values** (from neural network)
- **Actual values** (ground truth cos(x))

Both curves should track each other closely through at least two full periods.

## Troubleshooting

### Python Issues
- **TensorFlow installation error**: Ensure Python 3.10+ is installed
- **CUDA errors**: TensorFlow will fall back to CPU, which is fine for this task
- **Import errors**: Run `pip install -r requirements.txt` again

### Arduino Issues
- **Compilation error**: Ensure TensorFlowLite_ESP32 library is installed
- **Upload failed**: Check USB cable (must be data cable, not power-only)
- **Port not found**: Install ESP32 USB drivers (CP2102 or CH340 depending on board)
- **Model too large**: Reduce `kTensorArenaSize` in the sketch (try 2KB or 3KB)

### Visualization Issues
- **Serial Plotter blank**: Ensure baud rate is 115200
- **Only one curve visible**: Check that output format is "predicted,actual" with comma separator
- **Curves not tracking**: Verify quantization parameters match those from training script

## Dependencies

### Python
- tensorflow>=2.13.0
- numpy>=1.24.0
- matplotlib>=3.7.0

### Arduino
- ESP32 Board Package (latest version)
- No external libraries required (manual neural network implementation)

## Notes

- The model uses fixed random seeds (42) for reproducibility
- Training typically takes 1-2 minutes on a modern CPU
- The Arduino sketch uses a manual neural network implementation with extracted weights
- No external TensorFlow Lite libraries are required for the ESP32 deployment
- The manual implementation provides the same accuracy as the original TensorFlow model
- Weights are stored in `weights.h` as float arrays for easy deployment

## License

This project is for educational purposes as part of Assignment 3 for SWAPD 453.
