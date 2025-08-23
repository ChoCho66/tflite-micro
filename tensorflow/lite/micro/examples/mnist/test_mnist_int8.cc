/*
 * Copyright (C) 2024 UpbeatTech Inc. All Rights Reserved
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX short identifier: Apache-2.0
 * ==============================================================================*/

#include <math.h>

#include <cstdint>
#include <cstdio>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/mnist/mnist_inputs/sample_image_data.h"
#include "tensorflow/lite/micro/examples/mnist/models/mnist_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

constexpr int kTensorArenaSize = 400000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

namespace {
using TestOpResolver = tflite::MicroMutableOpResolver<1>;
}

// Helper function to calculate absolute difference
float abs_diff(float a, float b) { return (a > b) ? (a - b) : (b - a); }

// Helper function to convert float to int8 quantized value
int8_t FloatToQuantized(float value, float scale, int zero_point) {
  int quantized = round(value / scale) + zero_point;
  if (quantized < -128) quantized = -128;
  if (quantized > 127) quantized = 127;
  return static_cast<int8_t>(quantized);
}

// Helper function to convert int8 quantized value back to float
float QuantizedToFloat(int8_t quantized_value, float scale, int zero_point) {
  return (quantized_value - zero_point) * scale;
}

// MNIST class names for readable output
const char* mnist_class_names[10] = {"0", "1", "2", "3", "4",
                                     "5", "6", "7", "8", "9"};

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kMnistInputSize = 784;  // 28x28

void mnist_setup(void) {
  model = tflite::GetModel(g_mnist_int8_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf(
        "Model provided is schema version %d not equal to supported version "
        "%d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();
  resolver.AddRelu();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
  printf("MNIST INT8 Model Setup Complete\n");
}

typedef struct {
  int test_case;
  int expected_class;
  int predicted_class;
  bool is_correct;
  float inference_output[10];
} MnistTestResult;

int get_predicted_mnist_class(float* inference_results, int num_classes) {
  int predicted_class = 0;
  float max_value = inference_results[0];
  for (int i = 1; i < num_classes; i++) {
    if (inference_results[i] > max_value) {
      max_value = inference_results[i];
      predicted_class = i;
    }
  }
  return predicted_class;
}

void print_mnist_test_summary(int total_tests, int correct_predictions) {
  printf("\n==================================================\n");
  printf("MNIST Test Summary:\n");
  printf("Total test cases: %d\n", total_tests);
  printf("Correct predictions: %d\n", correct_predictions);
  printf("Wrong predictions: %d\n", total_tests - correct_predictions);
  printf("Accuracy: %.2f%%\n",
         (float)correct_predictions / total_tests * 100.0f);
  printf("==================================================\n");
}

// 單一測試 (int8 model + 量化/反量化)
TfLiteStatus test_single_mnist_inference(int expected_digit) {
  printf("\n=== Testing MNIST Inference (int8) ===\n");

  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;

  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;

  // 塞入 int8 input (由 float 轉量化)
  for (int i = 0; i < kMnistInputSize; i++) {
    input->data.int8[i] =
        FloatToQuantized(g_sample_image_data[i], input_scale, input_zero_point);
  }
  printf("Input data quantized and loaded into tensor\n");

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed\n");
    return invoke_status;
  }

  // 讀取 int8 輸出並轉回 float
  float inference[10];
  for (int i = 0; i < 10; i++) {
    inference[i] =
        QuantizedToFloat(output->data.int8[i], output_scale, output_zero_point);
  }

  int predicted_class = get_predicted_mnist_class(inference, 10);

  printf("\n=== Results ===\n");
  printf("Class probabilities (dequantized):\n");
  for (int i = 0; i < 10; i++) {
    printf("  %d: %.6f", i, inference[i]);
    if (i == expected_digit) {
      printf(" <- Expected");
    }
    if (i == predicted_class) {
      printf(" <- Predicted");
    }
    printf("\n");
  }

  printf("\nPredicted digit: %d\n", predicted_class);
  if (expected_digit >= 0) {
    printf("Expected digit: %d\n", expected_digit);
    if (predicted_class == expected_digit) {
      printf("✓ PASS: Correct prediction\n");
    } else {
      printf("✗ FAIL: Incorrect prediction\n");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

int main() {
  mnist_setup();
  test_single_mnist_inference(-1);
  return 0;
}
