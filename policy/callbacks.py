import pdb

import wandb
import numpy as np
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class WandbRecorderCallback(BaseCallback):
    """
    A custom callback that allows to print stuff on wandb after every evaluation

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, eval_freq=None, wandb_loss_suffix="", verbose=0):
        super(WandbRecorderCallback, self).__init__(verbose)

        self.wandb_loss_suffix = wandb_loss_suffix
        # self.child_eval_freq = eval_freq
        # self.n_eval_calls = 0


    def _on_step(self) -> bool:
        """
        This method is called as a child callback of the `EventCallback`),
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.

        Print stuff on wandb
        """
        last_mean_reward = self.parent.last_mean_reward
        
        # self.n_eval_calls += 1
        # current_timestep = self.n_eval_calls*self.child_eval_freq
        # current_timestep = self.parent.n_calls
        current_timestep = self.num_timesteps  # this number is multiplied by the number of parallel envs
        wandb.log({"train_mean_reward"+self.wandb_loss_suffix: last_mean_reward, "timestep": current_timestep})

        ### Plot extra metrics (assumes env of type VecEnv)
        wandb_extra_metrics = None
        if isinstance(self.training_env, VecEnv):
            if np.all(self.training_env.has_attr('wandb_extra_metrics')):
                wandb_extra_metrics = self.training_env.get_attr('wandb_extra_metrics')[0]

        if wandb_extra_metrics is not None:
            for k, v in wandb_extra_metrics.items():
                wandb.log({v: np.mean(self.training_env.get_attr(k)), "timestep": current_timestep})

        return True


"""
    Template for custom callback
"""
# class CustomCallback(BaseCallback):
#     """
#     A custom callback that derives from ``BaseCallback``.

#     :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
#     """
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         # Those variables will be accessible in the callback
#         # (they are defined in the base class)
#         # The RL model
#         # self.model = None  # type: BaseAlgorithm
#         # An alias for self.model.get_env(), the environment used for training
#         # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
#         # Number of time the callback was called
#         # self.n_calls = 0  # type: int
#         # self.num_timesteps = 0  # type: int
#         # local and global variables
#         # self.locals = None  # type: Dict[str, Any]
#         # self.globals = None  # type: Dict[str, Any]
#         # The logger object, used to report things in the terminal
#         # self.logger = None  # stable_baselines3.common.logger
#         # # Sometimes, for event callback, it is useful
#         # # to have access to the parent object
#         # self.parent = None  # type: Optional[BaseCallback]

#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass

#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.

#         This method is not actually called after every single .reset()
#         """

#         # env = self.model.get_env()

#         # pdb.set_trace()
#         # env.step_counter()
#         # self.training_env.env_method('step_counter')

#         pass

#     def _on_step(self, *args, **kwargs) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.

#         ---> For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.

#         :return: (bool) If the callback returns False, training is aborted early.
#         """



#         print('Last mean reward:', self.parent.last_mean_reward)
#         last_mean_reward = self.parent.last_mean_reward

#         return True

#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         pass

#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass