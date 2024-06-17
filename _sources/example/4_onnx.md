# <center>Creating a ONNX model</center>
To continue our tiny rl journey, we will need to make this model compatible with the embedded microcontroller. This requires us to convert the PyTorch {cite}`ONNX` model to an ONNX {cite}`ONNX` model.  

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
    opset_version=14)
```