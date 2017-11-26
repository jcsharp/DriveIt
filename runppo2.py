#!/usr/bin/env python
import sys
import os.path as osp
import argparse
from datetime import datetime
from belief import BeliefDriveItEnv
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))
from baselines.ppo2 import Model

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=datetime.now().strftime('%Y%m%d%H%M%S'))
    parser.add_argument('-c', '--checkpoint', type=str)
    args = parser.parse_args()

    checkpoint = osp.join(args.log_dir, args.batch_name, 'checkpoints', args.checkpoint)
    print(checkpoint)
    env = BeliefDriveItEnv(time_limit=180)
    model = Model(DriveItPolicy, env.observation_space.shape, env.action_space.shape, 1, 1, 1, 0, 0, None)
    model.load(checkpoint)
    o = env.reset()
    print(o)
    a, v, s, neglogp = model.act_model.step(o)
    print((a, v, s, neglogp))

if __name__ == '__main__':
    main()
