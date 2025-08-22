source /home/kycho/Documents/tflite-micro-main/.tflite-micro/bin/activate
bazel run --jobs=12 tensorflow/lite/micro/examples/hello_world:hello_world_test_float

# source /home/kycho/Documents/tflite-micro-main/.tflite-micro/bin/activate
# bazel run --jobs=12 tensorflow/lite/micro/examples/hello_world:hello_world_test_int8