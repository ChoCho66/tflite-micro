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
#include "tensorflow/lite/micro/examples/hello_world/models/sin_float_model_data.h"
// #include "tensorflow/lite/micro/hello_world/models/hello_world_int8_model_data.h"
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


TfLiteStatus TestSingleInference(float input_value, float tolerance = 0.1f) {
    printf("\n=== Testing input: %f ===\n", input_value);
    
    // Set up model and interpreter
    const tflite::Model* model = tflite::GetModel(g_sin_float_model_data);
    TestOpResolver op_resolver;
    op_resolver.AddFullyConnected();
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return status;
    }
    
    // Set input
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    input->data.f[0] = input_value;
    
    // Run inference
    status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        printf("ERROR: Inference failed\n");
        return status;
    }
    
    // Get results
    float predicted = output->data.f[0];
    float expected = sin(input_value);  // The model approximates sine function
    float error = abs_diff(predicted, expected);
    
    printf("Input: %f\n", input_value);
    printf("Predicted: %f\n", predicted);
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

// Test multiple known values
TfLiteStatus TestKnownValues() {
    printf("\n=== Testing Known Values ===\n");
    
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
        {0.5f, 0.479f, "sin(0.5) ≈ 0.479"}
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    int passed = 0;
    float tolerance = 0.1f;  // 10% tolerance for embedded model
    

    
    for (int i = 0; i < num_tests; i++) {
        printf("\n--- Test %d: %s ---\n", i + 1, test_cases[i].description);
        
        TfLiteStatus status = TestSingleInference(test_cases[i].input, tolerance);
        if (status == kTfLiteOk) {
            passed++;
        }
    }

    
    if (passed == num_tests) {
        printf("ALL TESTS PASSED\n");
        return kTfLiteOk;
    } else {
        printf("TESTS FAILED\n");
        return kTfLiteError;
    }
}

int main() {
    TestKnownValues();
    return 0;
}