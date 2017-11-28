#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from datetime import datetime
from mpi4py import MPI
import gym, logging
from belief import BeliefDriveItEnv
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))

from baselines import bench, logger
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from baselines.ppo1 import pposgd_simple
from baselines.ppo1.mlp_policy import MlpPolicy

def train(num_timesteps, seed, log_dir):
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure(dir=log_dir, format_strs=['tensorboard'])
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = BeliefDriveItEnv(time_limit=180)
    env.seed(workerseed)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    gym.logger.setLevel(logging.WARN)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95, schedule='linear'
    )
    env.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-t', '--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=datetime.now().strftime('%Y%m%d%H%M%S'))
    args = parser.parse_args()

    log_dir = osp.join(args.log_dir, args.batch_name) 
    train(num_timesteps=args.num_timesteps, seed=args.seed ,log_dir=log_dir)

    print('done.')


if __name__ == '__main__':
    main()
