#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from datetime import datetime
from belief import BeliefDriveItEnv
from car import Car
from autopilot import ReflexPilot, SharpPilot
from PositionTracking import TruePosition
import tensorflow as tf
from policy import DriveItPolicy, LstmPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))

from baselines import bench, logger


def train(timesteps, nenvs, nframes, time_limit, seed, policy_name):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from vec_frame_stack import VecFrameStack
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

    pilots = (ReflexPilot, SharpPilot)

    def make_env(rank):
        def env_fn():
            cars = [Car.HighPerf(v_max=2.0), Car.Simple(v_max=1.0)]
            bots = [pilots[(rank + i) % len(pilots)](cars[i], cars) for i in range(1, len(cars))]
            env = BeliefDriveItEnv(cars[0], bots, time_limit, noisy=True, random_position=True, bot_speed_deviation=0.3)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)

    if policy_name == 'lstm':
        policy = LstmPolicy
        env = VecFrameStack(env, nframes)
        nsteps, nminibatches = 120, 4
    else:
        policy = DriveItPolicy
        nsteps, nminibatches = 4096, 32

    return ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
        lam=0.95, gamma=1.0, noptepochs=10, log_interval=1,
        ent_coef=0.00,
        lr=1e-4,
        cliprange=0.2,
        total_timesteps=timesteps,
        save_interval=10)


def main(name=datetime.now().strftime('%Y%m%d%H%M%S')):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-e', '--envs', help='number of environments', type=int, default=8)
    parser.add_argument('-f', '--frames', help='number of frames', type=int, default=4)
    parser.add_argument('-t', '--time-limit', type=int, default=180)
    parser.add_argument('-n', '--num-timesteps', type=int, default=int(3e7))
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=name)
    parser.add_argument('-p', '--policy-name', type=str, default='lstm')
    args = parser.parse_args()
    assert(args.envs > 1)

    log_dir = osp.join(args.log_dir, args.batch_name) 
    logger.configure(dir=log_dir, format_strs=['tensorboard']) #format_strs=['stdout','tensorboard'])

    model = train(timesteps=args.num_timesteps, nenvs=args.envs, nframes=args.frames, \
        time_limit=args.time_limit, seed=args.seed, policy_name=args.policy_name)
    
    return model


if __name__ == '__main__':
    main()
