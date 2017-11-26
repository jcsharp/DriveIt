#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from belief import BeliefDriveItEnv
import tensorflow as tf
from datetime import datetime
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "baselines"))

from baselines import bench, logger

def train(num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.vec_frame_stack import VecFrameStack
    from baselines.ppo2 import ppo2
    import gym
    import logging
    import multiprocessing

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def env_fn():
            env = BeliefDriveItEnv(time_limit=180)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn
    nenvs = 8
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = DriveItPolicy
    ppo2.learn(policy=policy, env=env, nsteps=4096, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=10)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--batch-name', type=str, default=datetime.now().strftime('%Y%m%d%H%M%S'))
    parser.add_argument('--log-dir', type=str, default='metrics')
    args = parser.parse_args()

    log_dir = osp.join(args.log_dir, args.batch_name) 
    logger.configure(dir=log_dir, format_strs=['tensorboard']) #format_strs=['stdout','tensorboard'])

    train(num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
