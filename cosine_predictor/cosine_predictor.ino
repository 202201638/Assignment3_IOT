/*
 * TinyML Cosine Wave Predictor - ESP32 Deployment
 * Assignment 3: IoT Applications Development (SWAPD 453)
 * Spring 2026
 * 
 * This sketch runs a neural network on ESP32 to predict cosine values.
 * The model was trained in Python and weights were extracted.
 * 
 * NO EXTERNAL LIBRARIES REQUIRED - Manual neural network implementation
 */

#include "weights.h"
#include <math.h>

// ReLU activation function
float relu(float x) {
  return x > 0 ? x : 0;
}

// Neural network forward pass
float predict(float x) {
  // Layer 1: 1 -> 16
  float h1[16];
  for (int i = 0; i < 16; i++) {
    h1[i] = layer1_weights[i] * x + layer1_bias[i];
    h1[i] = relu(h1[i]);
  }
  
  // Layer 2: 16 -> 16
  float h2[16];
  for (int i = 0; i < 16; i++) {
    h2[i] = layer2_bias[i];
    for (int j = 0; j < 16; j++) {
      h2[i] += layer2_weights[i * 16 + j] * h1[j];
    }
    h2[i] = relu(h2[i]);
  }
  
  // Layer 3: 16 -> 1
  float y = layer3_bias[0];
  for (int i = 0; i < 16; i++) {
    y += layer3_weights[i] * h2[i];
  }
  
  return y;
}

void setup() {
  Serial.begin(115200);
  
  delay(1000);
  Serial.println("\n========================================");
  Serial.println("TinyML Cosine Wave Predictor - ESP32");
  Serial.println("========================================\n");
  Serial.println("Manual neural network implementation");
  Serial.println("No external libraries required\n");
  
  Serial.println("\n========================================");
  Serial.println("Starting predictions...");
  Serial.println("Open Serial Plotter (Ctrl+Shift+L) to view");
  Serial.println("========================================\n");
}

void loop() {
  // Sweep x from 0 to 2π in 80 steps
  const int steps = 80;
  const float two_pi = 2.0 * PI;
  
  for (int i = 0; i < steps; i++) {
    float x = (float)i / (steps - 1) * two_pi;
    
    // Run prediction
    float y_predicted = predict(x);
    
    // Compute ground truth
    float y_actual = cos(x);
    
    // Print in format: predicted,actual
    Serial.print(y_predicted, 4);
    Serial.print(",");
    Serial.println(y_actual, 4);
    
    // Small delay for visualization
    delay(50);
  }
  
  // Brief pause between periods
  delay(500);
}
