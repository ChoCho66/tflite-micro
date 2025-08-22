# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//tensorflow/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

load("//tensorflow/lite/micro/examples/hello_world:BUILD_model_config.bzl", "get_model_info")

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

model_info = get_model_info()

tflm_cc_library(
    name = "model",
    srcs = model_info["srcs"],
    hdrs = model_info["hdrs"],
)

tflm_cc_test(
    name = "hello_world_test_float",
    srcs = [
        model_info["test_file_float"],
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/micro/examples/hello_world/mnist_inputs:mnist_inputs",
    ],
)

tflm_cc_test(
    name = "hello_world_test_int8",
    srcs = [
        model_info["test_file_int8"]
    ],
    deps = [
        ":model",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_log",
        "//tensorflow/lite/micro:micro_profiler",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:recording_allocators",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    data = [model_info["tflite_files"][0]],  # float
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("tensorflow"),
        "//python/tflite_micro:runtime",
    ],
)

py_binary(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = model_info["tflite_files"],
    deps = [
        ":evaluate",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)
