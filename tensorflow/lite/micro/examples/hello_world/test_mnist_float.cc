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

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/examples/hello_world/models/mnist_float_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// #include "mnist_inputs/sample0_image_data.h"
// #include "LiteRT_test.h"

constexpr int kTensorArenaSize = 400000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

namespace {
using TestOpResolver = tflite::MicroMutableOpResolver<1>;
}

// Helper function to calculate absolute difference
float abs_diff(float a, float b) { return (a > b) ? (a - b) : (b - a); }
// MNIST class names for readable output
const char* mnist_class_names[10] = {"0", "1", "2", "3", "4",
                                     "5", "6", "7", "8", "9"};

// Global variables (following your pattern)
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// MNIST constants
constexpr int kMnistInputSize = 784;  // 28x28
constexpr int kMnistOutputSize = 10;  // 10 classes

#include <cstdint>

#include "stdint.h"

constexpr int kNumMnistTestInputs = 1;
extern const float g_mnist_inputs[kNumMnistTestInputs][784];
extern const int g_mnist_expected_outputs[kNumMnistTestInputs];
#define MNIST_WIDTH 28
#define MNIST_HEIGHT 28
#define MNIST_SIZE (MNIST_WIDTH * MNIST_HEIGHT)

// Input data array (784 elements)
const float g_mnist_inputs[kNumMnistTestInputs][MNIST_SIZE] = {
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.05490196,0.58431375,0.75686276,0.019607844,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.35686275,0.8784314,0.99215686,0.99215686,0.07450981,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.10980392,0.92156863,0.99607843,0.99215686,0.99215686,0.6509804,0.07058824,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5647059,0.99215686,0.99607843,0.99215686,0.99215686,0.99215686,0.93333334,0.4509804,0.023529412,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.12156863,0.94509804,0.99215686,0.8156863,0.7254902,0.99215686,0.99215686,0.99215686,0.90588236,0.09411765,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.30980393,0.99607843,0.75686276,0.0,0.03137255,0.38431373,0.85882354,0.99607843,1.0,0.7882353,0.07058824,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3372549,0.99215686,0.3137255,0.0,0.0,0.0,0.7137255,0.99215686,0.99607843,0.7490196,0.047058824,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.6862745,0.99215686,0.60784316,0.0,0.0,0.0,0.91764706,0.99215686,0.99607843,0.5294118,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3372549,0.99215686,0.8156863,0.15686275,0.33333334,0.6509804,0.9843137,0.92941177,0.99607843,0.9254902,0.16470589,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.07058824,0.93333334,0.99215686,0.99607843,0.99215686,0.99215686,0.7254902,0.14117648,0.84705883,0.99215686,0.59607846,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.26666668,0.9411765,1.0,0.99607843,0.5686275,0.03137255,0.0,0.5254902,0.99607843,0.8745098,0.13725491,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.26666668,0.61960787,0.5568628,0.047058824,0.0,0.0,0.03529412,0.6862745,0.99215686,0.6313726,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.34509805,0.99215686,0.8862745,0.07058824,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.007843138,0.6509804,0.99215686,0.49411765,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1882353,0.9607843,0.99215686,0.14901961,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4509804,0.99607843,0.6745098,0.03529412,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.08235294,0.85490197,0.99607843,0.18039216,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.11764706,0.99607843,0.64705884,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7294118,0.95686275,0.16470589,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.05490196,0.8745098,0.30588236,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
};

void mnist_setup(void) {
  /* Map the model into a usable data structure. This doesn't involve any
   * copying or parsing, it's a very lightweight operation.
   */
  model = tflite::GetModel(g_mnist_float_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    printf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  /* This pulls in the operation implementations we need.
   * NOLINTNEXTLINE(runtime-global-variables)
   */
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();  // Likely needed instead of AveragePool2D
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddFullyConnected();
  resolver.AddRelu();  // Common activation for MNIST

  /* Build an interpreter to run the model with. */
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  /* Allocate memory from the tensor_arena for the model's tensors. */
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    printf("AllocateTensors() failed");
    return;
  }

  /* Obtain pointers to the model's input and output tensors. */
  input = interpreter->input(0);
  output = interpreter->output(0);

  /* Keep track of how many inferences we have performed. */
  inference_count = 0;
  printf("MNIST Model Setup Complete\n");
}

// Structure to store test results
typedef struct {
  int test_case;
  int expected_class;
  int predicted_class;
  bool is_correct;
  float inference_output[10];  // For MNIST: 10 classes
} MnistTestResult;

// Function to find the predicted class from inference results
int get_predicted_mnist_class(float* inference_results, int num_classes) {
  int predicted_class = 0;
  float max_value = inference_results[0];

  // 找最大值的類別
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

// 單一測試 (float model)
TfLiteStatus test_single_mnist_inference(int expected_digit,
                                         int test_case_index) {
  printf("\n=== Testing MNIST Inference (float) ===\n");

  // 塞入浮點 input
  for (int i = 0; i < kMnistInputSize; i++) {
    input->data.f[i] = g_mnist_inputs[test_case_index][i];
  }
  printf("Input data (float) loaded into tensor\n");

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    printf("Invoke failed\n");
    return invoke_status;
  }

  // 讀取 float 輸出
  float inference[10];
  for (int i = 0; i < 10; i++) {
    inference[i] = output->data.f[i];
  }

  // Get predicted class
  int predicted_class = get_predicted_mnist_class(inference, 10);

  printf("\n=== Results ===\n");
  printf("Class probabilities:\n");
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

  // 跑一筆測資 (假設第 0 筆答案是 7，你可以改掉)
  test_single_mnist_inference(-1, 0);

  return 0;
}
