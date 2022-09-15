from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'ViT-B_32.onnx'
model_quant = 'ViT-B_32.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)