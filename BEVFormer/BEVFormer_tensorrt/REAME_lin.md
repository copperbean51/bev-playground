

# Preparation
cd TensorRT/build
cmake .. -DCMAKE_TENSORRT_PATH=/usr
make -j$(nproc)
make install
cd third_party/bev_mmdet3d
python setup.py build develop --user

```
pip install pytorch_quantization==2.1.3
```
```
ImportError: cannot import name 'iou3d_cuda' from partially initialized module 'third_party.bev_mmdet3d.ops.iou3d' (most likely due to a circular import) (/workspace/BEVFormer/BEVFormer_tensorrt/./third_party/bev_mmdet3d/ops/iou3d/__init__.py)
```
-> cd third_party/bev_mmdet3d
-> python3 setup.py build develop


# Convert tiny pytorch to onnx
python3 tools/pth2onnx.py configs/bevformer/bevformer_tiny_trt.py checkpoints/pytorch/bevformer_tiny_epoch_24.pth --opset_version 16 --cuda

## Convert tiny onnx to TensorRT
python3 tools/bevformer/onnx2trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/onnx/bevformer_tiny_epoch_24_opset16.onnx
python3 tools/bevformer/onnx2trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/onnx/bevformer_tiny_epoch_24_opset16.onnx --int8 --fp16 --calibrator entropy
python3 tools/bevformer/onnx2trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/onnx/bevformer_tiny_epoch_24_opset16.onnx --fp16

# Eval TensorRT Fp32, FP16, FP16+INT8
python3 tools/bevformer/evaluate_trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/tensorrt/bevformer_tiny_epoch_24_opset16.trt
python3 tools/bevformer/evaluate_trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/tensorrt/bevformer_tiny_epoch_24_opset16_fp16.trt
python3 tools/bevformer/evaluate_trt.py configs/bevformer/bevformer_tiny_trt.py checkpoints/tensorrt/bevformer_tiny_epoch_24_opset16_entropy_int8_fp16.trt

# Convert tiny pytorch to onnx with plugin
python3 tools/pth2onnx.py configs/bevformer/plugin/bevformer_tiny_trt_p.py checkpoints/pytorch/bevformer_tiny_epoch_24.pth --opset_version 13 --cuda --flag cp

python3 tools/bevformer/onnx2trt.py configs/bevformer/plugin/bevformer_tiny_trt_p.py checkpoints/onnx/bevformer_tiny_epoch_24_cp_opset16.onnx
python3 tools/bevformer/onnx2trt.py configs/bevformer/plugin/bevformer_tiny_trt_p.py checkpoints/onnx/bevformer_tiny_epoch_24_cp_opset16.onnx --int8 --fp16 --calibrator entropy

# Eval TensorRT Fp32, FP16, FP16+INT8
python3 tools/bevformer/evaluate_trt.py configs/bevformer/plugin/bevformer_tiny_trt_p.py checkpoints/tensorrt/bevformer_tiny_epoch_24_cp_opset16.trt
python3 tools/bevformer/evaluate_trt.py configs/bevformer/plugin/bevformer_tiny_trt_p.py checkpoints/tensorrt/bevformer_tiny_epoch_24_cp_opset16_entropy_int8_fp16.trt
