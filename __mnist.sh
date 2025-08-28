source /home/kycho/Documents/tflite-micro/.tflite-micro/bin/activate

pushd tensorflow/lite/micro/examples/mnist/mnist_inputs > /dev/null || exit 1

    # 隨機選 0~9
    num=$(( RANDOM % 10 ))

    # mnist 會去測試 sample.png 這裡隨機挑選一個
    cp "sample${num}.png" "sample.png"

    # 執行 python 腳本
    python generate_cc_arrays.py . sample.png

popd > /dev/null

bazel run --jobs=12 tensorflow/lite/micro/examples/mnist:mnist_test_float

# source /home/kycho/Documents/tflite-micro-main/.tflite-micro/bin/activate
bazel run --jobs=12 tensorflow/lite/micro/examples/mnist:mnist_test_int8