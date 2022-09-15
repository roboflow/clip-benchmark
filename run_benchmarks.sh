#!/bin/bash
python clip_benchmark.py --device_id="cuda" --runtime="onnx"
python clip_benchmark.py --device_id="cuda" --runtime="onnx" --batch_size=4

python clip_benchmark.py --device_id="cpu" --runtime="onnx"
python clip_benchmark.py --device_id="cpu" --runtime="onnx" --batch_size=4

python clip_benchmark.py --device_id="cpu" --runtime="onnx_quant"
python clip_benchmark.py --device_id="cpu" --runtime="onnx_quant" --batch_size=4

# torch
python clip_benchmark.py --device_id="cuda" --runtime="torch"
python clip_benchmark.py --device_id="cuda" --runtime="torch" --batch_size=4

python clip_benchmark.py --device_id="cpu" --runtime="torch"
python clip_benchmark.py --device_id="cpu" --runtime="torch" --batch_size=4

python clip_benchmark.py --device_id="cpu" --runtime="torch_jit"
python clip_benchmark.py --device_id="cpu" --runtime="torch_jit" --batch_size=4