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
#include "tensorflow/lite/micro/examples/hello_world/models/sin_int8_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// #include "tensorflow/lite/micro/micro_profiler.h"
// #include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
 
constexpr int kTensorArenaSize = 2048;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

namespace {
using TestOpResolver = tflite::MicroMutableOpResolver<1>;
}

// Helper function to calculate absolute difference
float abs_diff(float a, float b) {
    return (a > b) ? (a - b) : (b - a);
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

// Helper function to convert int8 quantized value back to float
float QuantizedToFloat(int8_t quantized_value, float scale, int zero_point) {
    // Dequantize: value = (q - zero_point) * scale
    return (quantized_value - zero_point) * scale;
}

TfLiteStatus TestSingleInference(float input_value, float tolerance = 0.15f) {
    printf("\n=== Testing input: %f ===\n", input_value);
    
    // Set up model and interpreter (using int8 quantized model)
    const tflite::Model* model = tflite::GetModel(g_sin_int8_model_data);
    TestOpResolver op_resolver;
    op_resolver.AddFullyConnected();
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return status;
    }
    
    // Get input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    // Get quantization parameters for input tensor
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    
    // Get quantization parameters for output tensor
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    printf("Input quantization - Scale: %f, Zero point: %d\n", input_scale, input_zero_point);
    printf("Output quantization - Scale: %f, Zero point: %d\n", output_scale, output_zero_point);
    
    // Convert float input to quantized int8
    int8_t quantized_input = FloatToQuantized(input_value, input_scale, input_zero_point);
    input->data.int8[0] = quantized_input;
    
    printf("Input value: %f -> Quantized: %d\n", input_value, quantized_input);
    
    // Run inference
    status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        return status;
    }
    
    // Get quantized output and convert back to float
    int8_t quantized_output = output->data.int8[0];
    float predicted = QuantizedToFloat(quantized_output, output_scale, output_zero_point);
    
    // Calculate expected value
    float expected = sin(input_value);  // The model approximates sine function
    float error = abs_diff(predicted, expected);
    
    printf("Output quantized: %d -> Predicted: %f\n", quantized_output, predicted);
    printf("Expected (sin): %f\n", expected);
    printf("Absolute error: %f\n", error);
    printf("Tolerance: %f\n", tolerance);
    
    if (error <= tolerance) {
        printf("✓ PASS: Error within tolerance\n");
        return kTfLiteOk;
    } else {
        printf("✗ FAIL: Error exceeds tolerance\n");
        return kTfLiteError;
    }
}

// Test multiple known values with int8 quantized model
TfLiteStatus TestKnownValues() {
    printf("\n=== Testing Known Values (int8 Quantized) ===\n");
    
    // Test data: input values and their expected sine values
    struct TestCase {
        float input;
        float expected_sin;
        const char* description;
    };
    
    TestCase test_cases[] = {
        {0.0f, 0.0f, "sin(0) = 0"},
        {1.57f, 1.0f, "sin(π/2) ≈ 1"},
        {3.14f, 0.0f, "sin(π) ≈ 0"},
        {1.0f, 0.841f, "sin(1) ≈ 0.841"},
        {0.5f, 0.479f, "sin(0.5) ≈ 0.479"},
        {2.0f, 0.909f, "sin(2) ≈ 0.909"}
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;
    float tolerance = 0.15f;  // Slightly higher tolerance for quantized model
    
    for (int i = 0; i < num_tests; i++) {
        printf("\n--- Test %d: %s ---\n", i + 1, test_cases[i].description);
        
        TfLiteStatus status = TestSingleInference(test_cases[i].input, tolerance);
        if (status == kTfLiteOk) {
            passed++;
        }
    }
    
    printf("\n=== Test Summary ===\n");
    printf("Total tests: %d\n", num_tests);
    printf("Passed: %d\n", passed);
    printf("Failed: %d\n", num_tests - passed);
    printf("Success rate: %.1f%%\n", (float)passed / num_tests * 100);
    
    if (passed == num_tests) {
        printf("✓ ALL TESTS PASSED\n");
        return kTfLiteOk;
    } else {
        printf("✗ SOME TESTS FAILED\n");
        return kTfLiteError;
    }
}

// // Additional function to test quantization accuracy
// void TestQuantizationAccuracy() {
//     printf("\n=== Testing Quantization Accuracy ===\n");
    
//     // Test some sample values to see quantization effects
//     float test_values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.14f};
//     int num_values = sizeof(test_values) / sizeof(test_values[0]);
    
//     // Assume typical quantization parameters (you should get these from actual model)
//     float scale = 0.02f;  // Example scale
//     int zero_point = 0;   // Example zero point
    
//     printf("Testing with scale: %f, zero_point: %d\n", scale, zero_point);
    
//     for (int i = 0; i < num_values; i++) {
//         float original = test_values[i];
//         int8_t quantized = FloatToQuantized(original, scale, zero_point);
//         float dequantized = QuantizedToFloat(quantized, scale, zero_point);
//         float quantization_error = abs_diff(original, dequantized);
        
//         printf("Original: %6.3f -> Quantized: %4d -> Dequantized: %6.3f (Error: %6.3f)\n",
//                original, quantized, dequantized, quantization_error);
//     }
// }

int main() {
    TestKnownValues();
    return 0;
}