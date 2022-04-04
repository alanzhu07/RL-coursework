#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
import os
import torch
import collections
import tqdm
import matplotlib.pyplot as plt
from numpy import random
import copy

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, nS, nA, logdir=None):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        self.net = FullyConnectedModel(nS, nA)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.lr = lr
        self.logdir = logdir
    def save_model_weights(self):
    # Helper function to save your model / weights.
        model_file = "weights.pth"
        torch.save(self.net.state_dict(), model_file)
        return model_file

    def load_model(self, model_file):
    # Helper function to load an existing model.
        self.net.load_state_dict(torch.load(model_file))
        self.net.eval()

    def load_model_weights(self,weight_file):
    # Optional Helper function to load model weights.
        pass


class Replay_Memory():
    def __init__(self, memory_size=50000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.burn_in = burn_in
        self.mem = collections.deque(maxlen=memory_size)
        self.memory_size = memory_size
    
    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        sampled_idx = random.choice(len(self.mem), batch_size)
        return [self.mem[i] for i in sampled_idx]

    def append(self, transition):
    # Appends transition to the memory.
        if(len(self.mem) < self.memory_size):
            self.mem.append(transition)
        else:
            self.mem.popleft()
            self.mem.append(transition)


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env, nS, nA, nE, lr, gamma, eps, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.lr = lr
        self.eps = eps
        self.nE = nE
        self.nS = nS
        self.nA = nA
        self.env = env

        self.gamma = gamma
        self.Qw = QNetwork(env, lr, nS, nA)
        self.Qtar = copy.deepcopy(self.Qw)
        self.rMem = Replay_Memory()

    def epsilon_greedy_policy(self, q_values):
        if random.uniform(0,1) > self.eps:
            return np.argmax(q_values.detach().numpy())
        else:
            return random.randint(0, self.nA)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values.detach().numpy())

    def compute_returns(self, minibatch):
        # Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        returns = []
        for (curr_state, action, next_state, reward, terminated) in minibatch:
            if terminated:
                returns.append(reward)
            else:
                returns.append(reward + self.gamma*torch.max(self.Qtar.net(next_state)))
        return returns

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        self.burn_in_memory()
        count = 0
        reward_means = []
        for i in tqdm.tqdm(range(self.nE+1)):

            if i % 10 == 0:
                reward_mean, reward_sd = self.test()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(i, reward_mean, reward_sd))
                reward_means.append(reward_mean)
            if i == self.nE:
                break

            curr_state = torch.tensor(self.env.reset(), dtype=torch.float32)
            terminated = False
            reward_arr = []
            while not terminated:
                self.Qw.optimizer.zero_grad()
                action = self.epsilon_greedy_policy(self.Qw.net(curr_state))
                next_state, reward, terminated, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                reward_arr.append(reward)
                self.rMem.append([curr_state, action, next_state, reward, terminated])

                minibatch = self.rMem.sample_batch()
                returns = self.compute_returns(minibatch)
                loss = 0
                for (s, a, r, s1, term), y in zip(minibatch, returns):
                    loss += (y - self.Qw.net(s)[a])**2
                loss /= len(minibatch)
                loss.backward()
                self.Qw.optimizer.step()
                count += 1
                if count%50 == 0:
                    self.Qtar = copy.deepcopy(self.Qw)
                    self.Qtar.net.eval()
                curr_state = next_state
            # print(f'iteration {i} reward {sum(reward_arr)}')
        
        return reward_means

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.
        
        Qw = copy.deepcopy(self.Qw.net)
        test_epsiodes = 20

        G = np.zeros(test_epsiodes)
        for k in range(test_epsiodes):
            curr_state = torch.tensor(self.env.reset(), dtype=torch.float32)
            terminated = False
            reward_arr = []
            while not terminated:
                action = self.greedy_policy(Qw(curr_state))
                next_state, reward, terminated, _ = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                curr_state = next_state
                reward_arr.append(reward)
            reward = sum(reward_arr)
            G[k] = reward

        reward_mean = G.mean()
        reward_sd = G.std()
        return reward_mean, reward_sd
                

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        count=0
        curr_state = torch.tensor(self.env.reset(), dtype=torch.float32)
        new_state = None
		
        while(count < self.rMem.burn_in):
            action = random.randint(0, self.nA)
            new_state, reward, terminated, _ = self.env.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float32)
            self.rMem.append([curr_state, action, new_state, reward, terminated])
            if terminated:
                curr_state = torch.tensor(self.env.reset(), dtype=torch.float32)
            else:
                curr_state = new_state
            count+=1


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env
    env = gym.make(environment_name)
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    num_episodes = 200
    num_seeds = 5
    gamma = 0.99
    epsilon = 0.05

    l = num_episodes//10 + 1
    res = np.zeros((num_seeds, l))

    for i in tqdm.tqdm(range(num_seeds)):
        agent = DQN_Agent(env, nS, nA, num_episodes, args.lr, gamma, epsilon)
        reward_means = agent.train()
        res[i] = reward_means
    
    ks = np.arange(l)*10
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    plt.title("DQN Learning Curve", fontsize = 24)
    plt.savefig("./plots/dqn_curve.png")
    # You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)
