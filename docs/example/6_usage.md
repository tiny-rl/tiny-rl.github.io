# <center>Usage</center>
**Arduino code**  
The Arduino code will start a serial port on the default usb and read in observation strings. These obervation strings will be converted to an actual array and inserted into the model. An argmax will then decide which action to take based on the output of the model. 
:::{note}
When you have a continuous action space you need to remove the argmax!
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
        entry(state,out);  
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