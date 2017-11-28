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

def main():
    nenv = 8
    nstack = 4

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default='make_model.pkl')
    parser.add_argument('-c', '--checkpoint', type=str)
    args = parser.parse_args()

    model_dir = osp.join(args.log_dir, args.batch_name)
    model_file = osp.join(model_dir, args.model)
    checkpoint_path = osp.join(model_dir, 'checkpoints', args.checkpoint)

    env = BeliefDriveItEnv(time_limit=180)

    with tf.Session() as sess:
        import cloudpickle
        with open(model_file, 'rb') as f:
            make_model = cloudpickle.load(f)
        model = make_model()
        model.load(checkpoint_path)

        osize = env.observation_space.shape[0]
        oz = np.zeros((nenv, osize * nstack))

        obs = env.reset()
        reward, done = 0, False
        while not done:
            env.render()
            for i in range(osize):
                oz[0,i] = obs[i]
            a, v, _, _ = model.step(oz)
            o, r, done, info = env.step(a[0])
            reward += r
        
        print(reward)
        env.close()
        sess.close()

if __name__ == '__main__':
    main()
