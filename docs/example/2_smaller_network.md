# <center>Creating a smaller network</center>
First of al we want to down size our network parameters. We can create a new smaller neural network in PyTorch {cite}`pytorch` that inherits our frameworks NetworkInterface. The NetworkInterface is necessary to be compatible with our compression framework.  

:::{important}
- The input layer should have the same shape as the environments observation
- The forward should return three values (these can be None): mean_actions, action_value, action_std
- The get_action should return which action to take after the forward method is taken
:::
```python
class TinyStudentDQN(NetworkInterface):
    def __init__(self, config: Config):
        super(TinyStudentDQN, self).__init__(config)

        self.net = nn.Sequential(
            nn.Linear(self.config.observation_shape[0], 16),
            nn.ReLU(),

            nn.Linear(16, 16),
            nn.ReLU()
        )

        self.action_head = nn.Linear(16, 2)

        for module in self.children():
            if type(module) == nn.Linear:
                module.bias.data.uniform_(0.0) # type: ignore
                module.weight.data.uniform_(0, 0.01) # type: ignore

    def forward(self, observations):
        x = self.net(observations)
        self.mean_actions = self.action_head(x)

        return self.mean_actions, None, None
    
    def get_actions(self):
        actions = self.mean_actions.argmax(dim=1).reshape(-1)
        return actions.cpu().numpy()
```