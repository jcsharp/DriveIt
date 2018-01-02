#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from datetime import datetime
from belief import BeliefDriveItEnv
from car import Car
from autopilot import LeftLaneFollowingPilot, RightLaneFollowingPilot
from PositionTracking import TruePosition
import tensorflow as tf
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))

from baselines import bench, logger


def train(timesteps, nenvs, nframes, num_cars, time_limit, seed):
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

    pilots = (LeftLaneFollowingPilot, RightLaneFollowingPilot)

    def make_env(rank):
        def env_fn():
            cars = [Car.HighPerf(v_max=2.0)]
            for i in range(1, num_cars):
                cars.append(Car.Simple(v_max=1.2))
            bots = [pilots[(rank + i) % len(pilots)](cars[i], cars) for i in range(1, len(cars))]
            env = BeliefDriveItEnv(cars[0], bots, time_limit, noisy=True, random_position=True, bot_speed_deviation=0.15)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, nframes)
    nsteps = 32768 // nenvs

    return ppo2.learn(policy=DriveItPolicy, env=env, nsteps=nsteps, nminibatches=32,
        lam=0.95, gamma=0.995, noptepochs=10, log_interval=1,
        vf_coef=0.5, ent_coef=0.00,
        lr=1e-4, cliprange=0.2,
        total_timesteps=timesteps,
        save_interval=10)

def set_idle_priority():
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.IDLE_PRIORITY_CLASS)

def main(name=datetime.now().strftime('%Y%m%d%H%M%S')):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-e', '--envs', help='number of environments', type=int, default=32)
    parser.add_argument('-f', '--frames', help='number of frames', type=int, default=4)
    parser.add_argument('-t', '--time-limit', type=int, default=60)
    parser.add_argument('-n', '--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-c', '--num-cars', type=int, default=2)
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=name)
    args = parser.parse_args()
    assert(args.envs > 1)

    log_dir = osp.join(args.log_dir, args.batch_name) 
    logger.configure(dir=log_dir, format_strs=['tensorboard']) #format_strs=['stdout','tensorboard'])

    set_idle_priority()
    model = train(timesteps=args.num_timesteps, nenvs=args.envs, nframes=args.frames, \
        num_cars=args.num_cars, time_limit=args.time_limit, seed=args.seed)
    
    return model


if __name__ == '__main__':
    main()
