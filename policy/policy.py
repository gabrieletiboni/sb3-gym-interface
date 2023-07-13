"""
	Wrapper class for policy training and evaluation
"""
import sys
import numpy as np
import torch
import pdb
import os

from .callbacks import WandbRecorderCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.vec_env import VecEnv


class Policy:
	
	def __init__(self,
				 algo=None,
				 env=None,
				 lr=0.0003,
				 device='cpu',
				 seed=None,
				 load_from_pathname=None,
                 gradient_steps=-1):
        """
            Policy class that handles training and making all required networks.
            It provides an easy-to-use interface to sb3 APIs, with wandb support, automatic
            best model returned, and more under the hood.

            ---
            gradient_steps: number of gradient updates for SAC. -1 means as many as
                            env.num_envs
        """
		# assert isinstance(env, VecEnv)
		# else: env = make_vec_env(env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

		self.seed = seed
		self.device = device
		self.env = env
		self.algo = algo
        self.gradient_steps = gradient_steps

		if load_from_pathname is None:
			self.model = self.create_model(algo, lr=lr)
		else:
			self.model = self.load_model(algo, load_from_pathname)

		return
	
	def create_model(self, algo, lr):
		if algo == 'ppo':
			policy_kwargs = dict(activation_fn=torch.nn.Tanh,
			                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])
			model = PPO("MlpPolicy", self.env, policy_kwargs=policy_kwargs, learning_rate=lr, verbose=0, seed=self.seed, device=self.device)

		elif algo == 'sac':
			policy_kwargs = dict(activation_fn=torch.nn.Tanh,
			                     net_arch=dict(pi=[128, 128], qf=[128, 128]))
			model = SAC("MlpPolicy", self.env, policy_kwargs=policy_kwargs, learning_rate=lr, gradient_steps=self.gradient_steps, verbose=0, seed=self.seed, device=self.device)
		else:
			raise ValueError(f"RL Algo not supported: {algo}")

		return model

	def load_model(self, algo, pathname):
		if algo == 'ppo':
			model = PPO.load(pathname, env=self.env, device=self.device)
		elif algo == 'sac':
			model = SAC.load(pathname, env=self.env, device=self.device)
		else:
			raise ValueError(f"RL Algo not supported: {algo}")
		return model

	def train(self,
			  timesteps=1000,
			  stopAtRewardThreshold=False,
			  n_eval_episodes=50,
			  eval_freq=1000,
			  best_model_save_path=None,
			  return_best_model=True,
			  wandb_loss_suffix="",
			  verbose=0):
		"""Train a model

			1. Setup callbacks
			2. Train model
			3. Find best model and return it
		"""

		if self.model.get_env().env_method('get_reward_threshold')[0] is not None and stopAtRewardThreshold:
			stop_at_reward_threshold = StopTrainingOnRewardThreshold(reward_threshold=self.model.get_env().env_method('get_reward_threshold')[0], verbose=1)
		else:
			stop_at_reward_threshold = None

		wandb_recorder = WandbRecorderCallback(eval_freq=eval_freq, wandb_loss_suffix=wandb_loss_suffix) # Plot stuff on wandb
		n_eval_episodes = n_eval_episodes
		eval_callback = EvalCallback(self.env,
		                             best_model_save_path=best_model_save_path,
		                             # log_path='./logs/',
		                             eval_freq=eval_freq,
		                             n_eval_episodes=n_eval_episodes,
		                             deterministic=True,
		                             callback_after_eval=wandb_recorder,
		                             callback_on_new_best=stop_at_reward_threshold,
		                             verbose=verbose,
		                             render=False)

		self.model.learn(total_timesteps=timesteps, callback=eval_callback)

		if return_best_model:   # Find best model among last and best
			reward_final, std_reward_final = self.eval(n_eval_episodes=n_eval_episodes)

			assert os.path.exists(os.path.join(best_model_save_path, "best_model.zip")), "best_model.zip hasn't been saved because too few evaluations have been performed. Check --eval_freq and -t"
			best_model = self.load_model(self.algo, os.path.join(best_model_save_path, "best_model.zip"))
			reward_best, std_reward_best = evaluate_policy(best_model, best_model.get_env(), n_eval_episodes=n_eval_episodes)

			if reward_final > reward_best:
			    best_policy = self.state_dict()
			    best_mean_reward, best_std_reward = reward_final, std_reward_final
			    which_one = 'final'
			else:
			    best_policy = best_model.policy.state_dict()
			    best_mean_reward, best_std_reward = reward_best, std_reward_best
			    which_one = 'best'

			info = {'which_one': which_one}

			return best_mean_reward, best_std_reward, best_policy, info
		else:
			return self.eval(n_eval_episodes)

	def eval(self, n_eval_episodes=50, render=False):
		mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=n_eval_episodes, render=render)
		return mean_reward, std_reward

	def predict(self, state, deterministic=False):
	    return self.model.predict(state, deterministic=deterministic)

	def state_dict(self):
		return self.model.policy.state_dict()

	def save_state_dict(self, pathname):
		torch.save(self.state_dict(), pathname)

	def load_state_dict(self, path_or_state_dict):
		if type(path_or_state_dict) is str:
			self.model.policy.load_state_dict(torch.load(path_or_state_dict, map_location=torch.device(self.device)), strict=True)
		else:
			self.model.policy.load_state_dict(path_or_state_dict, strict=True)

	def save_full_state(self, pathname):
		self.model.save(pathname)

	def load_full_state(self, pathname):
		raise ValueError('Use the constructor with load_from_pathname parameter')
		pass