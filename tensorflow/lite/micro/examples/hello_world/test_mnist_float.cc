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
#include "tensorflow/lite/micro/examples/hello_world/models/mnist_example_down_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "mnist_data.h"
#include "LiteRT_test.h"
constexpr int kTensorArenaSize = 400000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

namespace {
using TestOpResolver = tflite::MicroMutableOpResolver<1>;
}

// Helper function to calculate absolute difference
float abs_diff(float a, float b) {
    return (a > b) ? (a - b) : (b - a);
}
// MNIST class names for readable output
const char* mnist_class_names[10] = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
};

// Global variables (following your pattern)
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// MNIST constants
constexpr int kMnistInputSize = 784;  // 28x28
constexpr int kMnistOutputSize = 10;  // 10 classes

void mnist_setup(void)
{
    /* Map the model into a usable data structure. This doesn't involve any
     * copying or parsing, it's a very lightweight operation.
     */
    model = tflite::GetModel(g_mnist_example_down_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model provided is schema version %d not equal "
                    "to supported version %d.",
                    model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    /* This pulls in the operation implementations we need.
     * NOLINTNEXTLINE(runtime-global-variables)
     */
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();        // Likely needed instead of AveragePool2D
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddFullyConnected();
    resolver.AddRelu();             // Common activation for MNIST

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
    
    // Find the class with maximum confidence
    for (int i = 1; i < num_classes; i++) {
        if (inference_results[i] > max_value) {
            max_value = inference_results[i];
            predicted_class = i;
        }
    }
    return predicted_class;
}

void print_mnist_test_summary(MnistTestResult* results, int total_tests, int correct_predictions) {
    printf("\n==================================================\n");
    printf("MNIST Test Summary:\n");
    printf("Total test cases: %d\n", total_tests);
    printf("Correct predictions: %d\n", correct_predictions);
    printf("Wrong predictions: %d\n", total_tests - correct_predictions);
    printf("Accuracy: %.2f%%\n", (float)correct_predictions / total_tests * 100.0f);
    printf("==================================================\n");
}

// Helper function to convert float to int8 quantized value
int8_t FloatToQuantized(float value, float scale, int zero_point) {
   // Quantize: q = round(value / scale) + zero_point
   int quantized = round(value / scale) + zero_point;


   // Clamp to int8 range [-128, 127]
   if (quantized < -128) quantized = -128;
   if (quantized > 127) quantized = 127;


   return (int8_t)quantized;
}


// Helper function to dequantize int8 back to float
float Int8ToFloat(int8_t value, float scale, int zero_point) {
   return (static_cast<int>(value) - zero_point) * scale;
}


// Single test inference function (int8 model)
TfLiteStatus test_single_mnist_inference(int expected_digit, int test_case_index) {
   printf("\n=== Testing MNIST Inference (int8) ===\n");


   // Quantize input data
   float input_scale = input->params.scale;
   int input_zero_point = input->params.zero_point;


   for (int i = 0; i < kMnistInputSize; i++) {
       input->data.int8[i] = FloatToQuantized(g_mnist_inputs[test_case_index][i],
                                              input_scale, input_zero_point);
   }
   printf("Input data (quantized int8) loaded into tensor\n");


   // Run inference
   TfLiteStatus invoke_status = interpreter->Invoke();
   if (invoke_status != kTfLiteOk) {
       printf("Invoke failed\n");
       return invoke_status;
   }


   // Dequantize output for readability
   float output_scale = output->params.scale;
   int output_zero_point = output->params.zero_point;


   float inference[10];
   for (int i = 0; i < 10; i++) {
       inference[i] = Int8ToFloat(output->data.int8[i],
                                  output_scale, output_zero_point);
   }


   // Get predicted class
   int predicted_class = get_predicted_mnist_class(inference, 10);


   printf("\n=== Results ===\n");
   printf("Class probabilities (after dequantization):\n");
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


   inference_count++;
   return kTfLiteOk;
}
