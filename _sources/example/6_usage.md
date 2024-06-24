# <center>Usage</center>
**Arduino code**  
The Arduino code will start a serial port on the default usb and read in observation strings. These obervation strings will be converted to an actual array and inserted into the model. An argmax will then decide which action to take based on the output of the model.   
:::{warning}
When you have a continuous action space you need to remove the argmax!
:::  
:::{note}
This is a proof of concept to show that the policy network can run on an ESP32.  
The most important part of this piece of code is the inference which is done with **entry** function. The **entry** function takes two arguments:  
- state: state array in the same format as you trained your agent with.
- output: a reference to an array that is as big as raw policy network of your agent.
:::

```c
//----------
//Model code
//---------- 

void setup() {
  Serial.begin(115200);
}

void read_number() {
  read_digit = false;
  float x = std::atof(number);
  state[0][s_idx++] = std::atof(number);

  n_idx = 0;
  memset( number, '\0', sizeof(char)*FLT_STRING_SIZE );
}

int argmax(float *ptr, float *end) {
    float *first = ptr;
    float *max = ptr;
    
    while (ptr < end) 
        if (*++ptr > *max) 
            max = ptr;

    return max-first;
}

void loop() {
  if(Serial.available() > 0){
    c = Serial.read();
    if (isdigit(c) || c == '.' || c == '-') {
      read_digit = true;
      number[n_idx++] = c;
    }
    else if (c == ' ' && read_digit) {
      read_number();
    }
    else if (c == '[') {
      memset( state[0], 0, sizeof(float)*4 );
      s_idx = 0;
      sent = false;
    }
    else if (c ==']') {
      if (!sent) {
        if (read_digit){
          read_number();
        }

        float out[1][2];
        entry(state,out);  // Inference on neural network
        int N = sizeof(out) / sizeof(float);

        Serial.print(argmax(&out[0][0], &out[0][N]));
        Serial.print('\n');
        sent = true;
      }
    }
  }
}
```

**Python code**
The environment will be handled in python via a serial connection to the ESP32.

```python
import time

import serial
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

env = make_vec_env("CartPole-v1", n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs={"render_mode": "human"})

rewards_list = []
state = env.reset()
steps = 0
rewards = 0

port = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

def read_to_np():
    string = port.readline()
    return np.array([int(string)])

def write_ser(cmd):
    port.write(cmd)

while(1):
    write_ser(str(state).encode())
    
    actions = read_to_np()

    state, reward, dones, info  = env.step(actions)
    env.render()
    rewards += reward
    steps += 1

    if np.all(dones):
        print(steps)
        print(info[0]["episode"]["r"])
        print(rewards)
        print(info)
        rewards = 0
        steps = 0
```