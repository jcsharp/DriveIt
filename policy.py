#!/usr/bin/env python
import sys
import os.path as osp
import numpy as np
import tensorflow as tf

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "openai"))

from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

class DriveItPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, hid_size=256, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape, name='obs')
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', nact, act=lambda x:x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, hid_size=256, reuse=False):
        nenv = nbatch // nsteps
        ob_shape = (nbatch,) + ob_space.shape
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'fc1', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'fc2', nh=hid_size, init_scale=np.sqrt(2), act=tf.tanh)
            xs = batch_to_seq(h2, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h3, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h3 = seq_to_batch(h3)
            pi = fc(h3, 'pi', nact, act=lambda x:x)
            vf = fc(h3, 'v', 1, act=lambda x:x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
