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
#include "tensorflow/lite/micro/examples/hello_world/models/custom_example_int8_model_data.h"
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

// Runs a single inference on the int8 quantized model and returns the output value.
TfLiteStatus TestSingleInference(float x, float y, float* output_value, float tolerance = 0.15f) {
    printf("\n=== Testing input: (%f, %f) ===\n", x, y);
    
    // Set up model and interpreter (using int8 quantized model)
    const tflite::Model* model = tflite::GetModel(g_custom_example_int8_model_data);
    
    // This resolver is for a single operator. You might need to add more
    // operators depending on your new model architecture.
    TestOpResolver op_resolver;
    op_resolver.AddFullyConnected();
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return status;
    }

    // printf("Interpreter has %d inputs and %d outputs\n", 
    printf("Interpreter has %zu inputs and %zu outputs\n",
        interpreter.inputs_size(), interpreter.outputs_size());
    
    // Get pointers to the input and output tensors.
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    if (input == nullptr) {
        printf("ERROR: Input tensor is null!\n");
        return kTfLiteError;
    }

    if (output == nullptr) {
        printf("ERROR: Output tensor is null!\n");
        return kTfLiteError;
    }

    // Get quantization parameters for input tensor
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    
    // Get quantization parameters for output tensor
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    printf("Input quantization - Scale: %f, Zero point: %d\n", input_scale, input_zero_point);
    printf("Output quantization - Scale: %f, Zero point: %d\n", output_scale, output_zero_point);
    
    printf("Input tensor dims = [");
    for (int i = 0; i < input->dims->size; i++) {
        printf("%d", input->dims->data[i]);
        if (i < input->dims->size - 1) printf(", ");
    }
    printf("]\n");

    // Convert float inputs to quantized int8 values
    int8_t quantized_x = FloatToQuantized(x, input_scale, input_zero_point);
    int8_t quantized_y = FloatToQuantized(y, input_scale, input_zero_point);
    
    // ✅ Set quantized [x, y] to the input tensor
    input->data.int8[0] = quantized_x;
    input->data.int8[1] = quantized_y;
    
    printf("Input values: (%f, %f) -> Quantized: (%d, %d)\n", x, y, quantized_x, quantized_y);

    // Run inference.
    status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        return status;
    }
    
    // Get quantized output and convert back to float
    int8_t quantized_output = output->data.int8[0];
    float predicted = QuantizedToFloat(quantized_output, output_scale, output_zero_point);
    
    // Calculate expected value: z = x³ + y²
    float expected = (x * x * x) + (y * y);
    float error = abs_diff(predicted, expected);
    
    printf("Output quantized: %d -> Predicted: %f\n", quantized_output, predicted);
    // printf("Expected (x³ + y²): %f\n", expected);
    printf("Expected (x^3 + y^2): %f\n", expected);
    printf("Absolute error: %f\n", error);
    printf("Tolerance: %f\n", tolerance);
    
    *output_value = predicted;
    
    if (error <= tolerance) {
        printf("✓ PASS: Error within tolerance\n");
        return kTfLiteOk;
    } else {
        printf("✗ FAIL: Error exceeds tolerance\n");
        return kTfLiteError;
    }
}

// Test multiple known values and print the (input, output) pairs.
TfLiteStatus TestKnownValues() {

    printf("\n=== Testing Known Values for f(x,y) = x³ + y² (int8 Quantized) ===\n");
    
    // Test data for the custom function z = x³ + y².
    struct TestCase {
        float x;
        float y;
        const char* description;
    };
    
    TestCase test_cases[] = {
        {0.5f, 0.8f, "f(0.5, 0.8) = 0.5³ + 0.8² = 0.125 + 0.64 = 0.765"},
        {0.9f, -0.2f, "f(0.9, -0.2) = 0.9³ + (-0.2)² = 0.729 + 0.04 = 0.769"},
        {0.0f, 0.0f, "f(0, 0) = 0³ + 0² = 0"},
        {-0.7f, 0.6f, "f(-0.7, 0.6) = (-0.7)³ + 0.6² = -0.343 + 0.36 = 0.017"},
        {1.0f, 1.0f, "f(1, 1) = 1³ + 1² = 1 + 1 = 2"},
        {-1.0f, -1.0f, "f(-1, -1) = (-1)³ + (-1)² = -1 + 1 = 0"},
        {0.3f, -0.4f, "f(0.3, -0.4) = 0.3³ + (-0.4)² = 0.027 + 0.16 = 0.187"}
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;
    float tolerance = 0.15f;  // Slightly higher tolerance for quantized model
    
    for (int i = 0; i < num_tests; i++) {
        printf("\n--- Test %d: %s ---\n", i + 1, test_cases[i].description);
        
        float predicted_value;
        float x = test_cases[i].x;
        float y = test_cases[i].y;
        
        TfLiteStatus status = TestSingleInference(x, y, &predicted_value, tolerance);
        
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

// Additional function to test quantization accuracy for two-input function
void TestQuantizationAccuracy() {
    printf("\n=== Testing Quantization Accuracy for Two-Input Function ===\n");
    
    // Test some sample (x, y) pairs to see quantization effects
    struct TestPair {
        float x;
        float y;
    };
    
    TestPair test_pairs[] = {
        {-2.0f, -1.0f}, {0.0f, 0.0f}, {1.0f, 2.0f}, 
        {0.5f, -0.5f}, {-1.5f, 1.5f}, {2.0f, -2.0f}
    };
    
    int num_pairs = sizeof(test_pairs) / sizeof(test_pairs[0]);
    
    // Assume typical quantization parameters (you should get these from actual model)
    float scale = 0.02f;  // Example scale
    int zero_point = 0;   // Example zero point
    
    printf("Testing with scale: %f, zero_point: %d\n", scale, zero_point);
    
    for (int i = 0; i < num_pairs; i++) {
        float orig_x = test_pairs[i].x;
        float orig_y = test_pairs[i].y;
        
        int8_t quant_x = FloatToQuantized(orig_x, scale, zero_point);
        int8_t quant_y = FloatToQuantized(orig_y, scale, zero_point);
        
        float dequant_x = QuantizedToFloat(quant_x, scale, zero_point);
        float dequant_y = QuantizedToFloat(quant_y, scale, zero_point);
        
        float error_x = abs_diff(orig_x, dequant_x);
        float error_y = abs_diff(orig_y, dequant_y);
        
        printf("Original: (%6.3f, %6.3f) -> Quantized: (%4d, %4d) -> Dequantized: (%6.3f, %6.3f) (Errors: %6.3f, %6.3f)\n",
               orig_x, orig_y, quant_x, quant_y, dequant_x, dequant_y, error_x, error_y);
    }
}

int main() {
    TestKnownValues();
    return 0;
}