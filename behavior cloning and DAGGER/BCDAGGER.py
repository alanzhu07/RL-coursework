"""S22 10-403 HW3
# 10-403: Homework 3 Question 2 - Behavior Cloning & DAGGER

You will implement this assignment in this python file

You are given helper functions to plot all the required graphs
"""

from collections import OrderedDict
import json
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import tqdm

from imitation import Imitation


def generate_imitation_results(mode, expert_file, keys=[100], num_seeds=1, num_iterations=100):
	# Use a small number of training iterations
	# (e.g., 10) for debugging, and then try a larger number
	# (e.g., 100).

	# Dictionary mapping number of expert trajectories to a list of rewards.
	# Each is the result of running with a different random seed.
	# At the end of the function:
	# 	reward_data is a dictionary with keys for each # expert trajectories
	#	reward_data[i] will be a list of length num_seeds containing lists
	#	reward_data[i][j] will be a list of length num_iterations containing
	#	rewards each iteration using key i and seed j.
	reward_data = OrderedDict({key: [] for key in keys})
	accuracy_data = OrderedDict({key: [] for key in keys})
	loss_data = OrderedDict({key: [] for key in keys})

	for num_episodes in tqdm.tqdm(keys):
		for t in tqdm.tqdm(range(num_seeds)):
			print('*' * 50)
			print('num_episodes: %s; seed: %d' % (num_episodes, t))

			# Create the environment.
			env = gym.make('CartPole-v0')
			env.seed(t)  # set seed
			im = Imitation(env, num_episodes, expert_file)
			expert_reward = im.evaluate(im.expert)
			print('Expert reward: %.2f' % expert_reward)

			loss_vec = []
			acc_vec = []
			imitation_reward_vec = []
			for i in tqdm.tqdm(range(num_iterations)):
				print(f'iteration {i}')
				loss,acc = im.train(mode=mode)
				reward = im.evaluate(im.model)
				loss_vec.append(loss)
				acc_vec.append(acc)
				imitation_reward_vec.append(reward)

			reward_data[num_episodes].append(imitation_reward_vec)
			accuracy_data[num_episodes].append(acc_vec)
			loss_data[num_episodes].append(loss_vec)
	
	return reward_data, accuracy_data, loss_data, expert_reward


"""### Experiment: Student vs Expert
In the next two cells, you will compare the performance of the expert policy
to the imitation policies obtained via behavior cloning and DAGGER.
"""


def plot_student_vs_expert(mode, expert_file, keys=[100], num_seeds=1, num_iterations=100):
	assert len(keys) == 1
	reward_data, acc_data, loss_data, expert_reward = \
		generate_imitation_results(
			mode, expert_file, keys, num_seeds, num_iterations)

	plt.plot(np.arange(num_iterations), acc_data[keys[0]][0])
	plt.xlabel("# Iterations")
	plt.ylabel("Accuracy")
	plt.title(f"{mode.title()} Accuracy Plot")
	plt.savefig(f"plots/{'_'.join(mode.split())}_student_vs_expert_accuracy.png")
	plt.close()

	plt.plot(np.arange(num_iterations), loss_data[keys[0]][0])
	plt.xlabel("# Iterations")
	plt.ylabel("Loss")
	plt.title(f"{mode.title()} Loss Plot")
	plt.savefig(f"plots/{'_'.join(mode.split())}_student_vs_expert_loss.png")
	plt.close()

	plt.plot(np.arange(num_iterations), reward_data[keys[0]][0], label="Student")
	plt.axhline(expert_reward, label="Expert", c="green")
	plt.xlabel("# Iterations")
	plt.ylabel("Reward")
	plt.title(f"{mode.title()} Reward Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_student_vs_expert_rewards.png")
	plt.close()


"""Plot the reward, loss, and accuracy for each, remembering to label each line."""


def plot_compare_num_episodes(mode, expert_file, keys, num_seeds=1, num_iterations=100):
	s0 = time.time()
	reward_data, accuracy_data, loss_data, _ = \
		generate_imitation_results(
			mode, expert_file, keys, num_seeds, num_iterations)

	reward_dict = OrderedDict({key: np.mean(reward_data[key], axis=0) for key in keys})
	accuracy_dict = OrderedDict({key: np.mean(accuracy_data[key], axis=0) for key in keys})
	loss_dict = OrderedDict({key: np.mean(loss_data[key], axis=0) for key in keys})

	for k in keys:
		plt.plot(np.arange(num_iterations), accuracy_dict[k], label=f"num_episodes={k}")
	plt.xlabel("# Iterations")
	plt.ylabel("Accuracy")
	plt.title(f"{mode.title()} Accuracy Comparison Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_comparison_accuracy.png")
	plt.close()

	for k in keys:
		plt.plot(np.arange(num_iterations), loss_dict[k], label=f"num_episodes={k}")
	plt.xlabel("# Iterations")
	plt.ylabel("Loss")
	plt.title(f"{mode.title()} Loss Comparison Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_comparison_loss.png")
	plt.close()

	for k in keys:
		plt.plot(np.arange(num_iterations), reward_dict[k], label=f"num_episodes={k}")
	plt.xlabel("# Iterations")
	plt.ylabel("Reward")
	plt.title(f"{mode.title()} Reward Comparison Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_comparison_reward.png")
	plt.close()

	print('time cost', time.time() - s0)


def main():
	# Generate all plots for Problem 1

	# Pytorch Only #
	expert_file = 'expert_torch.pt'

	# Tensorflow Only #
	# expert_file = 'expert_tf.h5'

	# Switch mode
	# mode = 'behavior cloning'
	mode = 'dagger'

	# Change the list of num_episodes below for testing and different tasks
	keys = [100]  # [1, 10, 50, 100]
	num_seeds = 3  # 3
	num_iterations = 100    # Number of training iterations. Use a small number
	# (e.g., 10) for debugging, and then try a larger number
	# (e.g., 100).

	# Q2.1.1, Q2.2.1
	plot_student_vs_expert(mode, expert_file, [100],
	                       num_seeds=1, num_iterations=100)

	# Q2.1.2, Q2.2.2
	plot_compare_num_episodes(mode, expert_file, [1,10,50,100],
	                          num_seeds=3, num_iterations=100)


if __name__ == '__main__':
	main()
