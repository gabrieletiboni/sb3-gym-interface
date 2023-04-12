# SB3 interface for gym environments
This repo provides a ready-to-use code interface for applications using [stable-baselines3](https://stable-baselines3.readthedocs.io/) (sb3) for training RL agents in simulation on gym-registered environments. It abstracts complexity from the sb3 APIs providing useful global features under the hood, such as seamless integration with Wandb.

**Features**:
- [x] Automatic wandb recording of training return
- [x] Tracks and returns best policy overall while training
- [x] Flag for stopping training at average reward threshold
- [x] Compatibility with Domain Randomization while training (requires [random-envs](https://github.com/gabrieletiboni/random-envs) repo)
- [x] Support for vectorized gym environments
- [ ] checkpoint for resuming training
- [ ] test.py script for evaluation only

For more info refer to the official SB3 documentation at https://stable-baselines3.readthedocs.io/


## Installation
This repo is not meant to be used as a Python package. Simply clone the repo and use it a starting point for your RL project.
```
# install desired PyTorch version
# (optional) install random-envs if you wish to use Domain Randomization-compatible gym environments with this codebase (https://github.com/gabrieletiboni/random-envs)

pip install -r requirements.txt
```

## Getting Started
Basic pipeline for training an RL policy on gym-registered environments with this interface:
```
import wandb
from policy.policy import Policy

wandb.init( ... )
env = gym.make('Hopper-v3')
policy = Policy(algo='ppo', env=env, seed=42)

""" Training """
avg_return, std_return, best_policy, info = policy.train(timesteps=1000)
torch.save(best_policy, 'best_policy.pth')

""" Evaluation """
policy.load_state_dict(best_policy)
avg_return, std_return = policy.eval()

wandb.finish()
```
Check out `train.py` for a complete example, and `train_random_envs.py` for an example with domain randomization at training time (using [random-envs](https://github.com/gabrieletiboni/random-envs) repo).

## Citing
If you use this repository, please consider citing the following work which inspired the creation of this repo and made use of it throughout the experiments.
```     
@misc{tiboniadrbenchmark,
    title={Online vs. Offline Adaptive Domain Randomization Benchmark},
    author={Tiboni, Gabriele and Arndt, Karol and Averta, Giuseppe and Kyrki, Ville and Tommasi, Tatiana},
    year={2022},
    primaryClass={cs.RO},
    publisher={arXiv},
    doi={10.48550/ARXIV.2206.14661}
}
```
