# <center>Convert ONNX to C code</center>
Finally we want our model to run on th ESP32.  
First step is setup your development environment. This can be done in two ways:  

- Arduino (Easy)
    - Install the [IDE](https://www.arduino.cc/en/software)
    - Add the esp32 board in the boards manager
    - Goto Tools > Board > esp32 and select ESP32 Dev Module (or other matching name with your module)
    - Ready to go!
- Espressif (Intermediate)
    - Follow the instruction on [this link](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html)

Secondly, you need to [install the library](https://github.com/kraiskil/onnx2c) that converts the onnx model to C code with the following command: 
```sh
./onnx2c [your ONNX model file] > model.c
```
This command will output model.c which has an ENTRY function. The ENTRY function is used for inference.