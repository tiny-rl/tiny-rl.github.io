# <center>Configuring the framework</center>
The configuration is an object initialized with a dictionary. The config reference:

- *data_directory*(string): directory to save data to
- *run_name*(string): name of experiment
- *memory*(dict):
    - **Configuration of the memory that stores experience from teacher to pass to student.**
    - *device*(string): the device that is used to store the memory (cpu, cuda)
    - *size*(int): size of memory in number of cached steps.
    - *update_frequency*(int): frequency in epochs to updating the memory
    - *update_size*(int): minimum amount of steps to update the memory with.
    - *frame_stack_optimization*(bool): in case of framestacked environment there can be a optimization that only the last frame will be used for the update.
    - *check_consistency*(bool): enable consistency check of the memory
- *evaluator*(dict):
    - **Configuration of the evaluator that is responsible for collecting the data and testing the student network**
    - *device*(string): the device that used to evaluate (cpu, cuda)
    - *student_driven*(bool): enable the student to take the actions in the environment during the collecting of data of the theacher
    - *student_test_frequency*(int): frequency in epochs to test the student
    - *episodes*(int): the minimum amount of episodes for testing the student
    - *initialize*(int): amount of actions to skip in the beginning of an episode
    - *ray_workers*(int): amount of parallel evaluators
    - *deterministic*(bool): enable deterministic steps
    - *exploration_rate*(float): used in value iteration algorithm as exploration parameter (0.-1.)
- *compression*(dict):
    - **Configuration for the distillation process**
    - *device*(string): the device to use during a distillation step (cpu, cuda)
    - *checkpoint_frequency*(int): frequency in epochs to save student
    - *epochs*(int): amount of epochs to continue the distillation
    - *learning_rate*(float): the learning rate for the student network
    - *batch_size*(int): the size of experience steps to use during a single distillation step
    - **DISCRETE ACTION SPACE ONLY**:
        - *T*(float): softmax hyperparameter to scale outputs
        - *critic_importance*(float): importance of critic during distillation
    - **CONTINUOUS ACTION SPACE ONLY**
        - *distribtution*(string): take std of actions in to account during distillation (Std, Mean)
        - *loss*(string): loss function to use for distillation (KL, Huber, MSE)

```python
config = {
    "memory": {
        "size": 100000,
        "update_frequency": 1,
        "update_size": 10000,
        "device": "cpu",

        "frame_stack_optimization": False, 

        "check_consistency": True
    },
    "evaluator": {
        "student_driven": False,
        "student_test_frequency": 10,
        "episodes": 20, 
        "initialize": 0, 
        "ray_workers": 10,
        "device": "cpu",
        "deterministic": False
    },
    "compression": {
        "checkpoint_frequency": 2,
        "epochs": 600,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "device": "cuda",

        # Only used in discrete action spaces
        "T": 0.01,  # Softmax hyperparameter
        "critic_importance": 0.5,

        # Only used in continuous action spaces
        "distribution": "Std",  # Std, Mean 
        "loss": "KL"  # KL, Huber, MSE
    },
    "quantization": {
        "enabled": True,
        "bits": 8
    },
    "data_directory": "/path/for/data",
    "run_name": "test",
}

def get_environment(config: Config):
    return make_vec_env("CartPole-v1", n_envs=config.evaluator_config.env_workers, vec_env_cls=DummyVecEnv)

c = Config(get_environment, config)
```