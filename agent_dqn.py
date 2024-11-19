#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from dqn_model import DuelingDQN

import pandas as pd
import matplotlib.pyplot as plt

try:
    import wandb
except:
    pass

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_actions = env.action_space.n
        self.q_net = DuelingDQN(num_actions=num_actions).to(self.device)
        self.target_q_net = DuelingDQN(num_actions=num_actions).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, weight_decay=1e-4)

        self.buffer = deque(maxlen=50000)
        self.average_reward_buffer = deque(maxlen=500)
        # Prioritized replay buffer
        # self.buffer = PrioritizedReplayBuffer(capacity=10000)

        self.gamma = args.gamma
        self.epsilon_start = args.epsilon_start
        self.epsilon = args.epsilon_start
        self.epsilon_min = args.epsilon_min
        # self.epsilon_decay = args.epsilon_decay
        self.epsilon_decay_ep = int(args.num_episodes * args.epsilon_decay)
        self.epsilon_decay = (args.epsilon_start - self.epsilon_min) / self.epsilon_decay_ep
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq
        self.folder = args.folder

        self.reward_dict = {}
        self.average_reward_dict = {}
        self.loss_dict = {}


        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.steps = 0

        if args.train_dqn_again:
            print('loading trained model')
            self.q_net.load_state_dict(torch.load(args.model_path))
        
        if args.test_dqn:
            print('loading trained model')
            self.q_net.load_state_dict(torch.load(args.model_path))
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        if random.random() < self.epsilon and not test:
            return self.env.action_space.sample()
        observation = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(observation)
        # print("action", torch.argmax(q_values, dim=1).item())
        action = torch.argmax(q_values, dim=1).item()
        # action = max(0, min(action, self.env.action_space.n - 1))
        # print("Action: ", action)
        # print("Action_space: ", self.env.action_space.n)
        return action
        # return action
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        """
        self.buffer.append(*args)
        
        
    def replay_buffer(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch
        # experiences, indices, weights = self.buffer.sample(self.batch_size)
        # return experiences, indices, torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train(self):
        if len(self.buffer) < 5000:
            # print(len(self.buffer.buffer))
            # print("Buffer: ", self.buffer.buffer)
            return
        experiences = self.replay_buffer()
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        
        # print(f"States shape: {states.shape}") 
        # print(f"Actions shape: {actions.shape}") 
        # print(f"Rewards shape: {rewards.shape}") 
        # print(f"Next states shape: {next_states.shape}") 
        # print(f"Dones shape: {dones.shape}")
        
        # weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # q_values = self.q_net(states)
        # print("Q-values shape:", q_values.shape)

        curr_q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = self.q_net(next_states).argmax(dim=1)
        
        next_q_values = self.target_q_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # print(next_q_values, rewards, dones)
        # print("=============================================================================")
        
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # td_error = torch.abs(curr_q_values - target_q_values.detach()).detach().cpu().numpy()
        # prios = td_error.flatten().tolist()
        # print(curr_q_values, target_q_values)
        # print(curr_q_values.shape, target_q_values.shape)

        loss = F.huber_loss(curr_q_values, target_q_values.detach())

        # print()

        # print("Loss: ", loss)
        # prios = loss.detach().cpu().numpy() + 1e-5
        # loss = (loss * weights).mean()

        self.loss_dict[self.steps] = loss.item()
        # wandb.log({"loss": loss.item()}, step=self.steps)

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)

        # torch.nn.utils.clip_grad_value_(self.q_net.parameters(), clip_value=1.0)

        self.optimizer.step()
        return loss.item()

        
    
    def train_agent(self, num_episodes):
        wandb.init(project="MsPacman", name="ddqn", config=self.args, mode=self.args.wandb_mode)
        # wandb.log()
        # wandb.log(loss_dict: self.loss_dict)
        
        total_reward = 0
        for episode in tqdm(range(num_episodes)):
            log_data = {}
            state = self.env.reset()
            total_reward = 0
            done = False
            self.q_net.train()
            self.epiode_length = 0
            while not done:
                action = self.make_action(state, test=False)
                next_state, reward, done, _, _ = self.env.step(action)
                self.steps += 1
                # reward = -1 if done and reward == 0 else reward
                # reward = 0.1 if not done and reward == 0 else reward
                total_reward += reward
                # if done and reward == 0 or reward != 0:
                self.push((state, action, reward, next_state, done))
                # wandb.log()

                # if self.steps % 4 == 0:
                loss = self.train()

                # Decay epsilon per episode
                # self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay) if episode < self.epsilon_decay_ep else self.epsilon_min
                self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * max(0, self.epsilon_decay_ep - episode) / self.epsilon_decay_ep
                self.epiode_length += 1

                state = next_state

            # Store rewards and averages
            self.reward_dict[episode] = total_reward
            self.average_reward_buffer.append(total_reward)
            self.average_reward_dict[episode] = np.mean(list(self.average_reward_buffer))
            
            # last_500_rewards = list(self.reward_dict.values())[-500:]
            # self.average_reward_dict[episode] = np.mean(last_500_rewards)

            log_data.update({"total_reward": total_reward, "average_reward": self.average_reward_dict[episode], "epsilon": self.epsilon, "loss": loss, "episode_length": self.epiode_length})

            # Periodically update the target network
            if episode % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            print(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {self.epsilon:.3f} - Average Reward: {self.average_reward_dict[episode]:.2f}")

            # Save model checkpoint and rewards periodically
            if episode % 1000 == 0:
                filename = f"{wandb.run.dir}/dqn_{episode}_reward{self.average_reward_dict[episode]:.2f}.pth"
                torch.save(self.q_net.state_dict(), filename)
                wandb.save(f"dqn_{episode}_reward{self.average_reward_dict[episode]:.2f}.pth")

            
            if episode % 100 == 0:
                self.save_rewards()

            if len(log_data) > 0:
                wandb.log(log_data, step=episode, commit=True)
            

        torch.save(self.q_net.state_dict(), 'dqn_model.pth')
    
    def save_rewards(self):
        reward_df = pd.DataFrame.from_dict(self.reward_dict, orient='index')
        reward_df.to_csv(f"{self.folder}/reward.csv")
        plt.plot(reward_df)
        plt.savefig(f"{self.folder}/reward.png")
        plt.close()

        avg_reward_df = pd.DataFrame.from_dict(self.average_reward_dict, orient='index')
        avg_reward_df.to_csv(f"{self.folder}/average_reward.csv")
        plt.plot(avg_reward_df)
        plt.savefig(f"{self.folder}/average_reward.png")
        plt.close()

        avg_loss_df = pd.DataFrame.from_dict(self.loss_dict, orient='index')
        avg_loss_df.to_csv(f"{self.folder}/loss.csv")
        plt.plot(avg_loss_df)
        plt.savefig(f"{self.folder}/loss.png")
        plt.close()



