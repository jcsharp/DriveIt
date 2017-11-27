#!/usr/bin/env python
import sys
import os.path as osp
import argparse
import tensorflow as tf
from belief import BeliefDriveItEnv
from policy import DriveItPolicy

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))
from baselines.ppo2 import ppo2

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('-l', '--log-dir', type=str, default='metrics')
    parser.add_argument('-b', '--batch-name', type=str, default=None)
    parser.add_argument('-f', '--file', type=str, default='model.meta')
    args = parser.parse_args()

    model_dir = osp.join(args.log_dir, args.batch_name)
    model_file = osp.join(model_dir, args.file)
    checkpoint_path = osp.join(model_dir, 'checkpoints', '00230')

    env = BeliefDriveItEnv(time_limit=180)

    loader = tf.train.import_meta_graph(model_file)
    g = tf.get_default_graph()
    pi = g.get_tensor_by_name("pi:0")

    with tf.Session() as sess:
        #print(tf.train.latest_checkpoint(checkpoint_path))
        loader.restore(sess, checkpoint_path)

        obs = env.reset()
        print(obs)

        act = sess.run(pi, feed_dict={'Ob:0': obs})
        print(act)

if __name__ == '__main__':
    main()
