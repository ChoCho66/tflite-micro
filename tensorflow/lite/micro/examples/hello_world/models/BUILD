load("//tensorflow/lite/micro:build_def.bzl", "generate_cc_arrays")
load("//tensorflow/lite/micro/examples/hello_world:BUILD_model_config.bzl", "get_model_filenames")

package(
    default_visibility = ["//visibility:public"],
)

model_files = get_model_filenames()

exports_files(
    srcs = [
        model_files["float"]["tflite"],
        model_files["int8"]["tflite"],
    ],
    visibility = [
        "//codegen/examples/hello_world:__subpackages__",
        "//tensorflow/lite/micro/examples/hello_world:__subpackages__",
    ],
)

generate_cc_arrays(
    name = model_files["float"]["cc_name"],
    src = model_files["float"]["tflite"],
    out = model_files["float"]["cc_out"],
)

generate_cc_arrays(
    name = model_files["float"]["hdr_name"],
    src = model_files["float"]["tflite"],
    out = model_files["float"]["hdr_out"],
)

generate_cc_arrays(
    name = model_files["int8"]["cc_name"],
    src = model_files["int8"]["tflite"],
    out = model_files["int8"]["cc_out"],
)

generate_cc_arrays(
    name = model_files["int8"]["hdr_name"],
    src = model_files["int8"]["tflite"],
    out = model_files["int8"]["hdr_out"],
)
