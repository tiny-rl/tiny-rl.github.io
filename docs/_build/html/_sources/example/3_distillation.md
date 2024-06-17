# <center>Transfering teacher policy to student</center>
Now that we have our student, we can start transferring the policy to the student network. This process is called policy distillation. The distillation is also quantization aware. Quantize aware means that during the distillation, the weights of the student will be simulated as 8 bit ints.  

The setup code looks like this:
```python
checkpoint = load_from_hub(repo_id="sb3/dqn-CartPole-v1",filename="dqn-CartPole-v1.zip",)

c = Config(get_environment, config)

model = DQN.load(checkpoint, env=get_environment(c))

comp = Compressor(model, get_environment, c).student_model(TinyStudentDQN)
comp.compress()
```