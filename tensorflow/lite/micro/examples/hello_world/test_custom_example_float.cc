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
#include "tensorflow/lite/micro/examples/hello_world/models/custom_example_float_model_data.h"
// #include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
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


// Runs a single inference on the model and returns the output value.
TfLiteStatus TestSingleInference(float x, float y, float* output_value, float tolerance = 0.15f) {
    const tflite::Model* model = tflite::GetModel(g_custom_example_float_model_data);
    
    TestOpResolver op_resolver;
    op_resolver.AddFullyConnected();
    
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
    
    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        return status;
    }

    TfLiteTensor* input = interpreter.input(0);
    if (input == nullptr) {
        printf("Error: Input tensor is null!\n");
        return kTfLiteError;
    }

    input->data.f[0] = x;
    input->data.f[1] = y;

    status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        return status;
    }

    TfLiteTensor* output = interpreter.output(0);
    if (output == nullptr) {
        printf("Error: Output tensor is null!\n");
        return kTfLiteError;
    }

    *output_value = output->data.f[0];

    // 誤差容忍比較
    float expected = (x * x * x) + (y * y);
    float error = abs_diff(*output_value, expected);

    printf("Input=(%f, %f), Predicted=%f, Expected=%f, Error=%f, Tolerance=%f\n", 
           x, y, *output_value, expected, error, tolerance);

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
    printf("\n=== Testing Known Values for f(x,y) = x^3 + y^2 ===\n");
    
    struct TestCase {
        float x;
        float y;
    };

    TestCase test_cases[] = {
        {0.5f, 0.8f},
        {0.9f, -0.2f},
        {0.0f, 0.0f},
        {-0.7f, 0.6f},
        {-10.0f, 20.0f}
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    bool all_tests_ok = true;
    float tolerance = 0.15f;

    for (int i = 0; i < num_tests; i++) {
        float predicted_value;
        float x = test_cases[i].x;
        float y = test_cases[i].y;
        
        TfLiteStatus status = TestSingleInference(x, y, &predicted_value, tolerance);
        
        if (status != kTfLiteOk) {
            all_tests_ok = false;
        }
    }

    printf("\nFinished testing.\n");
    return all_tests_ok ? kTfLiteOk : kTfLiteError;
}

int main() {
    TestKnownValues();
    return 0;
}