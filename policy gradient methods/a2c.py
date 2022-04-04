import sys
import argparse
import numpy as np
import gym
import torch
from net import NeuralNet


class A2C(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, state_dim, nA, actor_lr, baseline_lr, critic_lr, N, baseline=False, a2c=True):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce
        self.type = None  # one of: "A2C", "Baseline", "Reinforce"
        self.actor_lr = actor_lr
        self.baseline_lr = baseline_lr
        self.critic_lr = critic_lr
        self.input_dim = state_dim
        self.output_dim = nA
        self.N = N
        if baseline:
            self.type = "Baseline"
            self.policy = NeuralNet(self.input_dim, self.output_dim, torch.nn.Softmax(dim=-1))
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = actor_lr)
            self.baseline = NeuralNet(self.input_dim, 1, torch.nn.Identity())
            self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr = baseline_lr)
        elif a2c:
            self.type = "A2C"
            self.policy = NeuralNet(self.input_dim, self.output_dim, torch.nn.Softmax(dim=-1))
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = actor_lr)
            self.baseline = NeuralNet(self.input_dim, 1, torch.nn.Identity())
            self.baseline_optimizer = torch.optim.Adam(self.baseline.parameters(), lr = critic_lr)
        else:
            self.type = "Reinforce"
            self.policy = NeuralNet(self.input_dim, self.output_dim, torch.nn.Softmax(dim=-1))
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = actor_lr)
        
    def evaluate_policy(self, env):
        # Compute Accumulative trajectory reward
        _, _, rewards = self.generate_episode(env)
        return sum(rewards)

    def compute_returns(self, reward_arr, gamma):
        Gt = []
        curr = reward_arr[-1]
        for i in range(len(reward_arr)-2,-1,-1):
            Gt.append(curr)
            curr = reward_arr[i] + gamma * curr
        Gt.append(curr)
        return Gt[::-1]

    def compute_n_step_returns(self, reward_arr, state_arr, gamma, N):
        Gt = []
        T = len(reward_arr)
        for t in range(len(reward_arr)):
            v_end = self.baseline(state_arr[t+N]) if t+N < T else 0
            temp = [gamma**(k-t) * reward_arr[k] for k in range(t, min(t+N-1,T-1)+1)]
            curr = sum(temp) + gamma**N * v_end
            Gt.append(curr)
        return Gt

    def get_baseline(self, states):
        return [self.baseline(state) for state in states]

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        
        state = torch.tensor(env.reset(), dtype=torch.float32)
        
        terminated = False
        state_arr, action_arr, reward_arr = [], [], [] 
        while not terminated:
            state_arr.append(state)

            action_prob = self.policy(state)
            # print(action_prob)
            action = np.random.choice(self.output_dim, p = action_prob.detach().numpy())
            action_arr.append(action)

            state, reward, terminated, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            reward_arr.append(reward)

        return state_arr, action_arr, reward_arr


    def train(self, env, gamma=0.99, n=10):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        if self.type == "Reinforce":
            self.optimizer.zero_grad()
            states, actions, rewards = self.generate_episode(env)
            returns = self.compute_returns(rewards,gamma)
            loss = 0
            for state,action, G in zip(states,actions,returns):
                loss -= G * torch.log(self.policy(state)[action])
            loss /= len(states)
            # print("loss", loss)
            loss.backward()
            self.optimizer.step()
        elif self.type == "Baseline":
            self.optimizer.zero_grad()
            self.baseline_optimizer.zero_grad()

            states, actions, rewards = self.generate_episode(env)
            returns = self.compute_returns(rewards,gamma)
            baseline_values = self.get_baseline(states)
            loss = 0
            baseline_loss = 0
            for state,action,G,baseline in zip(states,actions,returns,baseline_values):
                loss -= (G-baseline) * torch.log(self.policy(state)[action])
                baseline_loss += (G-baseline) ** 2
            
            loss /= len(states)
            baseline_loss /= len(states)

            loss.backward(retain_graph=True)
            baseline_loss.backward()
            self.optimizer.step()
            self.baseline_optimizer.step()
        elif self.type == "A2C":
            self.optimizer.zero_grad()
            self.baseline_optimizer.zero_grad()

            states, actions, rewards = self.generate_episode(env)
            returns = self.compute_n_step_returns(rewards, states, gamma, self.N)
            baseline_values = self.get_baseline(states)
            loss = 0
            baseline_loss = 0
            for state,action,G,baseline in zip(states,actions,returns,baseline_values):
                loss -= (G-baseline) * torch.log(self.policy(state)[action])
                baseline_loss += (G-baseline) ** 2
            
            loss /= len(states)
            baseline_loss /= len(states)

            loss.backward(retain_graph=True)
            baseline_loss.backward()
            self.optimizer.step()
            self.baseline_optimizer.step()


        
        

