#!/usr/bin/env python
import sys
import os.path as osp
import numpy as np
import argparse
import tensorflow as tf
from belief import BeliefDriveItEnv
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))
from baselines.ppo2 import ppo2
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

nenvs, nframes = 8, 4

def load_model(model_file, checkpoint_path):
    import cloudpickle
    with open(model_file, 'rb') as f:
        make_model = cloudpickle.load(f)
    model = make_model()
    model.load(checkpoint_path)
    return model


def render_one(model, time_limit=180, seed=0):
    from vec_frame_stack_1 import VecFrameStack
    env0 = BeliefDriveItEnv(time_limit=time_limit)
    env0.seed(seed)
    env = VecFrameStack(env0, nframes)
    obs = np.zeros((nenvs,) + env.observation_space.shape)

    o = env.reset()
    steps = 0
    reward = 0
    done = False
    while not done:
        env0.render()
        steps += 1
        obs[0,:] = o
        a, _, _, _ = model.step(obs)
        o, r, done, info = env.step(a[0])
        reward += r
    
    print((steps, reward, info))


def create_venv(time_limit=180, seed=0):
    from vec_frame_stack import VecFrameStack
    def make_env(rank):
        def env_fn():
            env = BeliefDriveItEnv(time_limit=time_limit)
            env.seed(seed + rank)
            # env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            return env
        return env_fn
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, nframes)
    return env

def run_many(model, time_limit=180, seed=0):
    env = create_venv(time_limit, seed)

    o = env.reset()
    steps = 0
    reward = np.zeros(nenvs)
    done = np.zeros(nenvs, dtype=bool)
    while not done.all():
        steps += 1
        a, _, _, _ = model.step(o)
        o, r, done, info = env.step(a)
        reward += r
    
    print((steps, reward, info))
    env.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default='make_model.pkl')
    parser.add_argument('-c', '--checkpoint', type=str)
    parser.add_argument('-e', '--envs', help='number of environments', type=int, default=8)
    parser.add_argument('-f', '--frames', help='number of frames', type=int, default=4)
    parser.add_argument('-t', '--time-limit', type=int, default=180)
    parser.add_argument('-r', '--render', action='store_true', default=False)
    args = parser.parse_args()

    model_dir = osp.join(args.log_dir, args.batch_name)
    model_file = osp.join(model_dir, args.model)
    checkpoint_path = osp.join(model_dir, 'checkpoints', args.checkpoint)

    with tf.Session() as sess:
        model = load_model(model_file, checkpoint_path)
        if args.render:
            render_one(model, args.time_limit, args.seed)
        else:
            run_many(model, args.time_limit, args.seed)

        sess.close()

if __name__ == '__main__':
    main()
