MODEL_PREFIX = "stuff_toy"
# MODEL_PREFIX = "custom_example"
# MODEL_PREFIX = "sin"

def get_model_info(prefix = MODEL_PREFIX):
    return {
        "srcs": [
            "//tensorflow/lite/micro/examples/hello_world/models:generated_%s_float_model_cc" % prefix,
            "//tensorflow/lite/micro/examples/hello_world/models:generated_%s_int8_model_cc" % prefix,
        ],
        "hdrs": [
            "//tensorflow/lite/micro/examples/hello_world/models:generated_%s_float_model_hdr" % prefix,
            "//tensorflow/lite/micro/examples/hello_world/models:generated_%s_int8_model_hdr" % prefix,
        ],
        "tflite_files": [
            "//tensorflow/lite/micro/examples/hello_world/models:%s_float.tflite" % prefix,
            "//tensorflow/lite/micro/examples/hello_world/models:%s_int8.tflite" % prefix,
        ],
        # "test_file": "hello_world_test_%s.cc" % prefix,
        "test_file_float": "test_%s_float.cc" % prefix,
        "test_file_int8": "test_%s_int8.cc" % prefix,
    }

def get_model_filenames(prefix = MODEL_PREFIX):
    return {
        "float": {
            "tflite": "%s_float.tflite" % prefix,
            "cc_name": "generated_%s_float_model_cc" % prefix,
            "cc_out": "%s_float_model_data.cc" % prefix,
            "hdr_name": "generated_%s_float_model_hdr" % prefix,
            "hdr_out": "%s_float_model_data.h" % prefix,
        },
        "int8": {
            "tflite": "%s_int8.tflite" % prefix,
            "cc_name": "generated_%s_int8_model_cc" % prefix,
            "cc_out": "%s_int8_model_data.cc" % prefix,
            "hdr_name": "generated_%s_int8_model_hdr" % prefix,
            "hdr_out": "%s_int8_model_data.h" % prefix,
        },
    }
