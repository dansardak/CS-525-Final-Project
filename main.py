"""

### NOTICE ###
You DO NOT need to upload this file

"""
import argparse
from test import test
from environment import Environment
import time
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def parse():
    parser = argparse.ArgumentParser(description="DS551/CS525 RL Project3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--train_dqn_again', action='store_true', help='whether train DQN again')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--record_video', action='store_true', help='whether to record video during testing')

    parser.add_argument('--model_path', type=str, default='dqn_model.pth', help='Path to save/load the model')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor for training')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--epsilon_min', type=float, default=0.01, help='Minimum value of epsilon for epsilon-greedy strategy')
    parser.add_argument('--epsilon_decay', type=float, default=0.2, help='Decay rate of epsilon for epsilon-greedy strategy')
    parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--target_update_freq', type=int, default=10, help='Frequency of updating the target network')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train the agent')
    parser.add_argument('--folder', type=str, default='models', help='Folder to save/load the model')
    parser.add_argument('--wandb_mode', type=str, help='wandb mode')
    # parser.add_argument('--target_update_freq', type=int, default=1000, help='Frequency of updating the target network')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args, record_video=False):
    start_time = time.time()
    if args.train_dqn or args.train_dqn_again:
        env_name = args.env_name or 'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True, test=False)
        # env = gym.make("BreakoutNoFrameskip-v4")  # , render_mode="human")
        # env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
        # env = FrameStack(env, num_stack=4)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        # agent.train()
        # agent.load_model('models/dqn_3900_reward1_0.pth')
        agent.train_agent(args.num_episodes)

    if args.test_dqn:
        render_mode_value = "rgb_array" if record_video else None
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True, render_mode=render_mode_value)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100, record_video=record_video)
    print('running time:',time.time()-start_time)

if __name__ == '__main__':
    args = parse()
    run(args, record_video=args.record_video)

