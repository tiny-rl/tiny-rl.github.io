# <center>Creating a ONNX model</center>
To continue our tiny rl journey, we will need to make this model compatible with the embedded microcontroller. This requires us to convert the [PyTorch](https://pytorch.org) model to an [ONNX](https://onnx.ai/) model. This step also include a PTQ method called [dynamic quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#dynamic-quantization). This is done to convert the weights to 8 bit integers.

:::{warning}
Doing this step after quantize aware policy distillation can result in unwanted behavior! To prevent this, set the quantization.enabled to False in the config dict.
:::

The conversion code looks like this:
```python
export_student_as_onnx(
    "/path/to/model.pth", 
    "/path/for/model.onnx", 
    TinyStudentDQN, 
    config, 
    get_environment, 
    verbose=True, 
    onnx_input_names=["input_1"], 
    onnx_output_names=["output_1"], 
    opset_version=14,
    qat=True)
```