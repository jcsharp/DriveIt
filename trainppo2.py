#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from datetime import datetime
from belief import BeliefDriveItEnv
from car import Car
from autopilot import LaneFollowingPilot #, ReflexPilot, SharpPilot
# from PositionTracking import TruePosition
import tensorflow as tf
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))

from baselines import bench, logger

def load_model(model_file, checkpoint_path):
    import cloudpickle
    with open(model_file, 'rb') as f:
        make_model = cloudpickle.load(f)
    model = make_model()
    model.load(checkpoint_path)
    return model

def train(timesteps, nenvs, nframes, num_cars, time_limit, seed, model_file=None, checkpoint_path=None):
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

    if model_file is None:
        model = None
    else:
        model = load_model(model_file, checkpoint_path)

    steps_per_batch = 30000
    batch_learn_goal = 100
    max_ep_per_batch = steps_per_batch / time_limit / 60.0
    distance_growth = nenvs / max_ep_per_batch / batch_learn_goal

    pilots = (LaneFollowingPilot,) # ReflexPilot, SharpPilot)

    def make_env(rank):
        def env_fn():
            cars = [Car.HighPerf(v_max=2.0)]
            for _ in range(1, num_cars):
                cars.append(Car.Simple(v_max=1.2))
            bots = [pilots[(rank + i) % len(pilots)](cars[i], cars) for i in range(1, len(cars))]
            env = BeliefDriveItEnv(cars[0], bots, time_limit, noisy=True, random_position=True, bot_speed_deviation=0.15)
            env.set_training_mode(distance_growth)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, nframes)
    nsteps = steps_per_batch // nenvs

    return ppo2.learn(policy=DriveItPolicy, model=model, env=env, nsteps=nsteps, nminibatches=30,
        lam=0.95, gamma=0.995, noptepochs=10, log_interval=1,
        vf_coef=0.5, ent_coef=0.00,
        lr=2e-4, cliprange=0.2,
        total_timesteps=timesteps,
        save_interval=10)

def set_idle_priority():
    import psutil, os
    p = psutil.Process(os.getpid())
    p.nice(psutil.IDLE_PRIORITY_CLASS)

def main(name=datetime.now().strftime('%Y%m%d%H%M%S')):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-e', '--envs', help='number of environments', type=int, default=24)
    parser.add_argument('-f', '--frames', help='number of frames', type=int, default=4)
    parser.add_argument('-t', '--time-limit', type=float, default=4)
    parser.add_argument('-n', '--num-timesteps', type=int, default=int(5e6))
    parser.add_argument('-c', '--num-cars', type=int, default=2)
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=name)
    parser.add_argument('-ib', '--initial-batch', type=str, default='nobot')
    parser.add_argument('-ic', '--initial-checkpoint', type=str, default='00140')
    args = parser.parse_args()
    assert(args.envs > 1)

    set_idle_priority()

    log_dir = osp.join(args.log_dir, args.batch_name) 
    logger.configure(dir=log_dir, format_strs=['tensorboard']) #format_strs=['stdout','tensorboard'])

    if args.initial_checkpoint is not None:
        if args.initial_batch is None:
            args.initial_batch = args.batch_name
        model_dir = osp.join(args.log_dir, args.initial_batch)
        model_file = osp.join(model_dir, 'make_model.pkl')
        checkpoint_path = osp.join(model_dir, 'checkpoints', args.initial_checkpoint)
        model = train(timesteps=args.num_timesteps, nenvs=args.envs, nframes=args.frames, \
            num_cars=args.num_cars, time_limit=args.time_limit, seed=args.seed, \
            model_file=model_file, checkpoint_path=checkpoint_path)
    else:
        model = train(timesteps=args.num_timesteps, nenvs=args.envs, nframes=args.frames, \
            num_cars=args.num_cars, time_limit=args.time_limit, seed=args.seed)
    
    return model


if __name__ == '__main__':
    main()
