"""Train a policy with sb3 using Uniform Domain Randomization. (requires random-envs repo: https://github.com/gabrieletiboni/random-envs)

    Examples:
        (DEBUG)
            python train_random-envs.py --offline --env RandomHopper-v0 -t 1000 --eval_freq 500 --reward_threshold

        (OFFICIAL)
            python train_random-envs.py --env RandomHopper-v0 -t 5000000 --eval_freq 40000 --seed 42 --now 12 --algo ppo --reward_threshold
"""
from pprint import pprint
import argparse
import pdb
import sys
import socket
import os

import numpy as np
import gym
import torch
import wandb
from stable_baselines3.common.env_util import make_vec_env

import random_envs
from customvecenvs.RandomVecEnv import RandomSubprocVecEnv
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
             project="<PROJECT-NAME>",
             group=(args.env if args.group is None else args.group),
             name=args.algo+'_seed'+str(args.seed)+'_'+random_string,
             save_code=True,
             tags=None,
             notes=args.notes,
             mode=('online' if not args.offline else 'disabled'))

    run_path = "runs/"+str(args.env)+"/"+get_run_name(args)+"_"+random_string+"/"
    create_dirs(run_path)
    save_config(vars(args), run_path)

    wandb.config.path = run_path
    wandb.config.hostname = socket.gethostname()

    # env = gym.make(args.env)
    env = make_vec_env(args.env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv)
    test_env = gym.make(args.test_env)

    bounds_low = env.get_task()[0] / args.bound_multiplier
    bounds_high = env.get_task()[0] * args.bound_multiplier
    bounds = np.vstack((bounds_low,bounds_high)).reshape((-1,), order='F')  # alternating bounds from the low and high bounds
    env.set_dr_distribution(dr_type='uniform', distr=bounds)
    env.set_dr_training(True)

    eff_lr = get_learning_rate(args, env)  # retrieve preferred lr for current env, if exists
    policy = Policy(algo=args.algo,
                    env=env,
                    lr=eff_lr,
                    device=args.device,
                    seed=args.seed)

    print('--- Policy training start ---')
    mean_reward, std_reward, best_policy, which_one = policy.train(timesteps=args.timesteps,
                                                                   stopAtRewardThreshold=args.reward_threshold,
                                                                   n_eval_episodes=args.eval_episodes,
                                                                   eval_freq=args.eval_freq,
                                                                   best_model_save_path=run_path,
                                                                   return_best_model=True)

    env.set_dr_training(False)

    policy.save_state_dict(run_path+"final_model.pth")
    policy.save_full_state(run_path+"final_full_state.zip")
    print('--- Policy training done ----')

    print('\n\nMean reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["train_mean_reward"] = mean_reward
    wandb.run.summary["train_std_reward"] = std_reward
    wandb.run.summary["which_best_model"] = which_one

    torch.save(best_policy, run_path+"overall_best.pth")
    wandb.save(run_path+"overall_best.pth")


    """Evaluation on target domain"""
    print('\n\n--- TARGET DOMAIN EVALUATION ---')
    test_env = make_vec_env(args.test_env, n_envs=args.now, seed=args.seed, vec_env_cls=RandomSubprocVecEnv)
    policy = Policy(algo=args.algo, env=test_env, device=args.device, seed=args.seed)
    policy.load_state_dict(best_policy)

    mean_reward, std_reward = policy.eval(n_eval_episodes=args.test_episodes)
    print('Target reward and stdev:', mean_reward, std_reward)

    wandb.run.summary["target_mean_reward"] = mean_reward
    wandb.run.summary["target_std_reward"] = std_reward

    wandb.finish()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default=None, type=str, help='Train gym env')
    parser.add_argument('--test_env', default=None, type=str, help='Test gym env')
    parser.add_argument('--group', default=None, type=str, help='Wandb run group')
    parser.add_argument('--algo', default='ppo', type=str, help='RL Algo (ppo, lstmppo, sac)')
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

    parser.add_argument('--bound_multiplier', '-bm', default=1.2, type=float, help='Bound multiplier')

    # LSTM specific
    parser.add_argument('--n_lstm_layers', default=1, type=int, help='N LSTM layers')

    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    main()