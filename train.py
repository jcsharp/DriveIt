#from DriveItMultiGym import DriveItEnv
from car import Car
from belief import BeliefDriveItEnv
from DeepQNetwork import *
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-g', '--gamma', default=0.99, type=float, help='Reward discount factor')
    parser.add_argument('-n', '--noop', default=0, type=int, help='Number of steps where the agent takes the default action')
    parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of epochs to run')
    parser.add_argument('-s', '--steps', default=10000, type=int, help='Number of steps per epoch')
    parser.add_argument('-p', '--plot', action='store_true', default=False, help='Flag for enabling Tensorboard')
    #parser.add_argument('env', default='Pong-v3', type=str, metavar='N', nargs='?', help='Gym Atari environment to run')

    args = parser.parse_args()

    env = BeliefDriveItEnv(gamma=args.gamma)
    agent = DeepQAgent((4,) + env.observation_space.shape, env.action_space.n, \
        monitor=args.plot, train_after=1000, gamma=args.gamma)

    current_step = 0
    action = 0
    max_steps = args.epoch * args.steps
    current_state = env.reset()

    while current_step < max_steps:
        if current_step >= args.noop:
            action = agent.act(current_state)
        new_state, reward, done, _ = env.step(action)
        new_state = new_state

        # Clipping reward for training stability
        reward = np.clip(reward, -1, 1)

        agent.observe(current_state, action, reward, done)
        agent.train()

        current_state = new_state

        if done:
            current_state = env.reset()

        current_step += 1
