# <center>Transfering teacher policy to student</center>
Now that we have our student, we can start transferring the policy to the student network. This process is called policy distillation. It is also possible to do the policy distillition quantize aware. Quantize aware means that during the distillation, the weights of the student will be simulated as 8 bit ints.  

:::{warning}
Doing this step after quantize aware policy distillation can result in unwanted behavior! To prevent this, set the quantization.enabled to False in the config dict.
:::
:::{Note}
You can find a default config setup an explanation [here](3_distillation_config.md)
:::

The setup code looks like this:
```python
config = {
    # Your config dict
}

def get_environment(config: Config):
    return make_vec_env("CartPole-v1", n_envs=config.evaluator_config.env_workers, vec_env_cls=DummyVecEnv)

checkpoint = "/path/to/dqn-CartPole-v1.zip"

c = Config(get_environment, config)

model = DQN.load(checkpoint, env=get_environment(c))

comp = Compressor(model, get_environment, c).student_model(TinyStudentDQN)
comp.compress()
```