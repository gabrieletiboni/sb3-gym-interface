"""Sample script for using sb3-gym-interface to train an RL agent"""
from pprint import pprint
import argparse
import pdb
import sys
import socket
import os

import gym
import torch
import wandb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# import random_envs
# from customvecenvs.RandomVecEnv import RandomSubprocVecEnv
from utils.utils import *
from policy.policy import Policy

def main():
    assert args.env is not None
    
    if args.test_env is None:
        args.test_env = args.env

    pprint(vars(args))
    set_seed(args.seed)
    random_string = get_random_string(5)

    wandb.init(config=vars(args),
             project="<PROJECT_NAME>",
             group=('default_group' if args.group is None else args.group),
             name=args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             mode=('online' if not args.offline else 'disabled'))

    run_path = "runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    create_dirs(run_path)
    save_config(vars(args), run_path)

    env = gym.make(args.env)
    test_env = gym.make(args.test_env)

    wandb.config.path = run_path
    wandb.config.hostname = socket.gethostname()

    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=SubprocVecEnv)

    policy = Policy(algo=args.algo,
                    env=env,
                    lr=1e-3,
                    device=args.device,
                    seed=args.seed)

    print('--- POLICY TRAINING ---')
    avg_return, std_return, best_policy, info = policy.train(timesteps=args.timesteps,
                                                                   stopAtRewardThreshold=args.reward_threshold,
                                                                   n_eval_episodes=args.eval_episodes,
                                                                   eval_freq=args.eval_freq,
                                                                   best_model_save_path=run_path,
                                                                   return_best_model=True)

    policy.save_state_dict(run_path+"final_model.pth")
    policy.save_full_state(run_path+"final_full_state.zip")
    torch.save(best_policy, run_path+"overall_best.pth")
    wandb.save(run_path+"overall_best.pth")

    print('\n\nMean reward and stdev:', avg_return, std_return)
    wandb.run.summary["train_avg_return"] = avg_return
    wandb.run.summary["train_std_return"] = std_return
    wandb.run.summary["which_best_model"] = info['which_one']



    print('\n\n--- POLICY EVALUATION ---')
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=SubprocVecEnv)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed)
    policy.load_state_dict(best_policy)

    avg_return, std_return = policy.eval(n_eval_episodes=args.test_episodes)
    print('\n\nTest return (avg & std):', avg_return, std_return)
    wandb.run.summary["test_avg_return"] = avg_return
    wandb.run.summary["test_std_return"] = std_return

    wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default=None, type=str, help='Train gym env [Hopper-v3, ...]')
    parser.add_argument('--test_env', default=None, type=str, help='Test gym env')
    parser.add_argument('--group', default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo', default='sac', type=str, help='RL Algo [ppo, sac]')
    parser.add_argument('--lr', default=None, type=float, help='Learning rate')
    parser.add_argument('--now', default=1, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--timesteps', '-t', default=1000, type=int, help='Training timesteps')
    parser.add_argument('--reward_threshold', default=False, action='store_true', help='Stop at reward threshold')
    parser.add_argument('--eval_freq', default=10000, type=int, help='timesteps frequency for training evaluations')
    parser.add_argument('--eval_episodes', default=50, type=int, help='# episodes for training evaluations')
    parser.add_argument('--test_episodes', default=100, type=int, help='# episodes for test evaluations')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', default='cpu', type=str, help='<cpu,cuda>')
    parser.add_argument('--notes', default=None, type=str, help='Wandb notes')
    parser.add_argument('--offline', default=False, action='store_true', help='Offline run without wandb')

    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()