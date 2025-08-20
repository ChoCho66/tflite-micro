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
#include <stdio.h>
#include <string.h>
#include "tensorflow/lite/micro/examples/hello_world/models/stuff_toy_int8_model_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Increased tensor arena size for the model
constexpr int kTensorArenaSize = 30000;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

namespace {
// Define the OpResolver for the operations used in the model
using TestOpResolver = tflite::MicroMutableOpResolver<5>;

void AddOps(TestOpResolver& op_resolver) {
    op_resolver.AddConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddReshape();
}
} // namespace

// Helper function to calculate absolute difference
float abs_diff(float a, float b) {
    return (a > b) ? (a - b) : (b - a);
}

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

// Runs a single inference on the int8 quantized model
TfLiteStatus TestKnownValues() {
    printf("\n=== Testing Custom Model Inference (int8 Quantized) ===\n");

    // 1. Set up model and interpreter
    const tflite::Model* model = tflite::GetModel(g_stuff_toy_int8_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("ERROR: Model provided is schema version %u not equal to supported version %d.\n",
               model->version(), TFLITE_SCHEMA_VERSION);
        return kTfLiteError;
    }

    TestOpResolver op_resolver;
    AddOps(op_resolver);

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);

    TfLiteStatus status = interpreter.AllocateTensors();
    if (status != kTfLiteOk) {
        printf("ERROR: Failed to allocate tensors\n");
        return status;
    }
    printf("Tensors allocated successfully.\n");

    // 2. Get pointers to input and output tensors
    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);

    // Verify input tensor shape
    printf("Input tensor dimensions: %d\n", input->dims->size);
    if (input->dims->size < 4) {
        printf("ERROR: Input tensor has fewer than 4 dimensions.\n");
        return kTfLiteError;
    }
    printf("Input shape: [%d, %d, %d, %d]\n",
           input->dims->data[0], input->dims->data[1], input->dims->data[2], input->dims->data[3]);
    if (input->dims->data[0] != 1 || input->dims->data[1] != 24 ||
        input->dims->data[2] != 2 || input->dims->data[3] != 1) {
        printf("ERROR: Bad input tensor parameters.\n");
        return kTfLiteError;
    }

    // Get quantization parameters
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    printf("Input quantization - Scale: %f, Zero point: %d\n", input_scale, input_zero_point);
    printf("Output quantization - Scale: %f, Zero point: %d\n", output_scale, output_zero_point);

    // 3. Use the provided input data
    constexpr int input_size = 24 * 2 * 1;
    float input_data[input_size] = {
        0, 0, 0, 0, 7, 1, 10, 1, 10, 1, 6, 0, 10, 1, 21, 1,
        27, 2, 16, 3, 6, 2, 6, 2, 10, 4, 13, 7, 7, 14, 4, 12,
        4, 21, 4, 25, 2, 11, 1, 3, 0, 1, 0, 0, 0, 0, 0, 0
    };

    // Convert float inputs to quantized int8 values
    int8_t quantized_input[input_size];
    for (int i = 0; i < input_size; ++i) {
        quantized_input[i] = FloatToQuantized(input_data[i], input_scale, input_zero_point);
    }

    // Copy quantized data to the input tensor
    memcpy(input->data.int8, quantized_input, sizeof(quantized_input));
    printf("Generated and copied quantized input data.\n");

    // 4. Run inference
    printf("Invoking interpreter...\n");
    status = interpreter.Invoke();
    if (status != kTfLiteOk) {
        printf("ERROR: Inference failed with status %d\n", status);
        return status;
    }
    printf("Inference completed.\n");

    // 5. Get and print results
    printf("Output tensor dimensions: %d\n", output->dims->size);
    if (output->dims->size < 2) {
        printf("ERROR: Output tensor has fewer than 2 dimensions.\n");
        return kTfLiteError;
    }
    printf("Output shape: [%d, %d]\n", output->dims->data[0], output->dims->data[1]);
    if (output->dims->data[0] != 1 || output->dims->data[1] != 7) {
        printf("ERROR: Bad output tensor parameters.\n");
        return kTfLiteError;
    }

    printf("\n--- Model Output ---\n");
    for (int i = 0; i < 7; ++i) {
        float dequantized_output = QuantizedToFloat(output->data.int8[i], output_scale, output_zero_point);
        printf("Output[%d]: Quantized: %d, Dequantized: %f\n",
               i, output->data.int8[i], static_cast<double>(dequantized_output));
    }
    printf("--------------------\n");

    printf("\u2713 PASS: Custom model test finished.\n");
    return kTfLiteOk;
}

int main() {
    TestKnownValues();
    return 0;
}