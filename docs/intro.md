# <center> tiny-rl: A tiny way of doing rl </center>
Tiny-rl aims to provide an end-to-end configurable and modular compression framework {cite}`ave2022quantizationaware` for reinforcement learning algorithms.  
End-to-end means that we start from a computer trained algorithm like a [DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) trained for the [Cart-Pole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and ends with a mobile neural network compatible with an embedded device such as an [ESP32](https://www.espressif.com/en/products/socs/esp32). The compression framework is based on knowledge distillation and network quantization. This means that we will first teach a smaller neural network ([the student](example/2_smaller_network.md)) the policy of a bigger policy network (the teacher) and lower the precision of the weights without compromising the accuracy.  

Tiny-rl also needs to be compatible with multiple rl frameworks. This gives the need for a modular code base. Currently only stable-baselines is supported but in the future other frameworks will follow.

:::{important}
The quantize aware training is in alpha. Currently it is done by using the [DoReFa quantizer](https://intellabs.github.io/distiller/algo_quantization.html#dorefa) of distiller from intellabs.
:::