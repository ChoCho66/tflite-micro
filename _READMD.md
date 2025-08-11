Open tensorflow/lite/micro/examples/hello_world/README.md

`uv venv .tflite-micro --python=3.11`

`uv pip install pillow`

移除 bazel 快取 `bazel clean --expunge`

- `bazel run tensorflow/lite/micro/examples/hello_world:hello_world_test`

- `make -f tensorflow/lite/micro/tools/make/Makefile test_hello_world_test`
  - Need `uv pip install numpy`