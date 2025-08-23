source /home/kycho/Documents/tflite-micro/.tflite-micro/bin/activate

# 隨機選 0~9
num=$(( RANDOM % 10 ))

# 重新命名產生的檔案
mv "sample${num}.png" "sample.png"

# 執行 python 腳本
python generate_cc_arrays.py . sample.png
