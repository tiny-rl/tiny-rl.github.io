# <center>Creating a ONNX model</center>
To continue our tiny rl journey, we will need to make this model compatible with the embedded microcontroller. This requires us to convert the [PyTorch](https://pytorch.org) model to an [ONNX](https://onnx.ai/) model. This step also include a PTQ method called [dynamic quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#dynamic-quantization). This is done to convert the weights to 8 bit integers.

:::{warning}
Doing this step after quantize aware policy distillation can result in unwanted behavior! To prevent this, set the quantization.enabled to False in the config dict.
:::

The conversion code looks like this:
```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy
import torch

from qpd.networks.models.student_six_model_dqn_ import StudentSixModelDQN
from run import config
from qpd.config import Config
from qpd.networks.wrapper.model_wrapper import ModelWrapper


def get_environment(config: Config):
    return make_vec_env("CartPole-v1", n_envs=config.evaluator_config.env_workers, vec_env_cls=DummyVecEnv, env_kwargs={"render_mode": "human"})

c = Config(get_environment, config)

env = get_environment(c)

checkpoint = "/path/to/dqn-CartPole-v1.zip"

model = DQN.load(checkpoint, env)
teacher = ModelWrapper.construct_wrapper(c, model)

c.init_env_model_based_params(teacher)

# Loading quantized student
student = StudentSixModelDQN(c)

student.load("/path/to/student.pth")


rewards_list = []
state = env.reset()

torch.onnx.export(student, torch.from_numpy(state), "/path/to/quantized_student.onnx", verbose=True, input_names=["input_1"], output_names=["output_1"], opset_version=14, qat=True)
```