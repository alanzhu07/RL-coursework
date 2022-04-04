from collections import OrderedDict 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

## Pytorch Only ##
import torch
from model_pytorch import make_model
from torch.utils.data import TensorDataset, Dataset, DataLoader


# Import make_model here from the approptiate model_*.py file
# This model should be the same as problem 2

### 2.1 Build Goal-Conditioned Task
class FourRooms:
	def __init__(self, l=5, T=30):
		'''
		FourRooms Environment for pedagogic purposes
		Each room is a l*l square gridworld, 
		connected by four narrow corridors,
		the center is at (l+1, l+1).
		There are two kinds of walls:
		- borders: x = 0 and 2*l+2 and y = 0 and 2*l+2 
		- central walls
		T: maximum horizion of one episode
			should be larger than O(4*l)
		'''
		assert l % 2 == 1 and l >= 5
		self.l = l
		self.total_l = 2 * l + 3
		self.T = T

		# create a map: zeros (walls) and ones (valid grids)
		self.map = np.ones((self.total_l, self.total_l), dtype=np.bool)
		# build walls
		self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
		self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
		self.map[l+1, l+1] = False

		# define action mapping (go right/up/left/down, counter-clockwise)
		# e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
		# hence resulting in moving right
		self.act_set = np.array([
			[1, 0], [0, 1], [-1, 0], [0, -1] 
		], dtype=np.int)
		self.action_space = spaces.Discrete(4)

		# you may use self.act_map in search algorithm 
		self.act_map = {}
		self.act_map[(1, 0)] = 0
		self.act_map[(0, 1)] = 1
		self.act_map[(-1, 0)] = 2
		self.act_map[(0, -1)] = 3



	def render_map(self):
		plt.imshow(self.map)
		plt.xlabel('y')
		plt.ylabel('x')
		plt.savefig('p2_map.png', 
					bbox_inches='tight', pad_inches=0.1, dpi=300)
		plt.show()
	
	def sample_sg(self):
		# sample s
		while True:
			s = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[s[0], s[1]]:
				break

		# sample g
		while True:
			g = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[g[0], g[1]] and \
				(s[0] != g[0] or s[1] != g[1]):
				break
		return s, g

	def reset(self, s=None, g=None):
		'''
		s: starting position, np.array((2,))
		g: goal, np.array((2,))
		return obs: np.cat(s, g)
		'''
		if s is None or g is None:
			s, g = self.sample_sg()
		else:
			assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
			assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
			assert (s[0] != g[0] or s[1] != g[1])
			assert self.map[s[0], s[1]] and self.map[g[0], g[1]]
		
		self.s = s
		self.g = g
		self.t = 1

		return self._obs()
	
	def step(self, a):
		'''
		a: action, a scalar
		return obs, reward, done, info
		- done: whether the state has reached the goal
		- info: succ if the state has reached the goal, fail otherwise 
		'''
		assert self.action_space.contains(a)

		action = self.act_set[a]
		self.s[0] += action[0]
		self.s[1] += action[1]
		#assert 0 < self.s[0] < self.total_l - 1 and 0 < self.s[1] < self.total_l - 1
		#assert 0 < self.g[0] < self.total_l - 1 and 0 < self.g[1] < self.total_l - 1
		#assert self.map[self.s[0], self.s[1]] and self.map[self.g[0], self.g[1]]
		self.t += 1
		if self.t == self.T:
			done = True
			info = 'fail'
		
		elif self.s[0] == self.g[0] and self.s[1] == self.g[1]:
			done = True
			info = 'succ'
		else:
			done = False
			info = 'fail'

		return self._obs(), 0.0, done, info

	def _obs(self):
		return np.concatenate([self.s, self.g])

# build env
l, T = 5, 30
env = FourRooms(l, T)
### Visualize the map
#env.render_map()

def plot_traj(env, ax, traj, goal=None):
	traj_map = env.map.copy().astype(np.float)
	traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
	traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
	traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
	if goal is not None:
		traj_map[goal[0], goal[1]] = 3 # goal
	ax.imshow(traj_map)
	ax.set_xlabel('y')
	ax.set_label('x')

### A uniformly random policy's trajectory
def test_step(env):
	s = np.array([1, 1])
	g = np.array([2*l+1, 2*l+1])
	s = env.reset(s, g)
	done = False
	traj = [s]
	while not done:
		s, _, done, _ = env.step(env.action_space.sample())
		traj.append(s)
	traj = np.array(traj)

	ax = plt.subplot()
	plot_traj(env, ax, traj, g)
	plt.savefig('p2_random_traj.png', 
			bbox_inches='tight', pad_inches=0.1, dpi=300)
	plt.show()

def minDistance(env, dist, visited):
	minD = 1000
	for i in range(1,12):
		for j in range(1,12):		
			if env.map[i, j] and visited[i][j] == False and dist[i][j] < minD:
					minD = dist[i][j]
					res = (i,j)
	return res

def shortest_path_expert(env):
	from queue import Queue
	N = 1000
	expert_trajs = []
	expert_actions = []
	expert_st_trajs = []
	unflat_trajs = []
	unflat_actions  = []

	for _ in range(1000):
		env.reset()
		dist = [[ 1000 for i in range(0,13)] for j in range(0,13)]
		visited = [[ False for i in range(0,13)] for j in range(0,13)]
		s = env.s
		par_act = [[ None for i in range(0,13)] for j in range(0,13)]
		dist[s[0]][s[1]] = 0
		for i in range(112):
			(x, y) = minDistance(env, dist, visited)
			xY = np.array([x,y])
			if (x,y) == tuple(env.g):
				break
			visited[x][y] = True
			adjP = [np.array([x+1,y]), np.array([x,y+1]), np.array([x-1,y]), np.array([x, y-1])]
			for p in range(4):
				a, b = adjP[p][0], adjP[p][1] 
				if env.map[x,y] and visited[a][b] == False and dist[a][b] > dist[x][y]+1:
					dist[a][b] = dist[x][y]+1
					#print(x,y,adjP[p],p)
					par_act[a][b] = (xY, p)
		
		st = env.g
		m, n = tuple(st)
		traj = np.zeros(shape=(dist[m][n]+1,2))
		traj[-1,0] = int(m)
		traj[-1,1] = int(n)
		st_traj = []
		act = []
		c = -1
		while (tuple(st)) != tuple(s):
			c -=1
			temp, pAct = par_act[st[0]][st[1]]
			parent = [int(temp[0]), int(temp[1])]
			act.append(pAct)
			st_traj.append(np.array([temp[0], temp[1], env.g[0], env.g[1]]))
			traj[c, 0] = parent[0]
			traj[c, 1] = parent[1]
			# np.append(traj[0], parent[0])
			# np.append(traj[1], parent[1])
			st = parent
		#traj = traj[::-1]
		#print(type(traj[1][0]))
		traj = traj.astype(int)
		act = act[::-1]
		st_traj = st_traj[::-1]

		expert_st_trajs.extend(st_traj)
		expert_trajs.append(traj)
		expert_actions.extend(act)
		unflat_trajs.append(st_traj)
		unflat_actions.append(act)

	return expert_st_trajs, expert_actions, unflat_trajs, unflat_actions

def action_to_one_hot(env, action):
	action_vec = np.zeros(env.action_space.n)
	action_vec[action] = 1
	return action_vec  

class GCBC:

	def __init__(self, env, expert_trajs, expert_actions):
		self.env = env
		self.expert_trajs = expert_trajs
		self.expert_actions = expert_actions
		self.transition_num = len(expert_actions)
		self.model = make_model(input_dim=4, output_dim=4)
	
	def reset_model(self):
		self.model = make_model(input_dim=4, output_dim=4)	

	def generate_behavior_cloning_data(self):
		# 3 you will use action_to_one_hot() to convert scalar to vector
		# state should include goal
		self._train_states = []
		self._train_actions = []
		#expert_st_trajs, expert_actions = shortest_path_expert(self.env)
		self._train_states = self.expert_trajs
		for i in self.expert_actions:
			self._train_actions.append(action_to_one_hot(env,i))

		self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		self._train_actions = np.array(self._train_actions) # size: (*, 4)
		
	def generate_relabel_data(self):
		# 4 apply expert relabelling trick
		self._train_states = []
		self._train_actions = []
		for i in range(len(self.expert_trajs)):
			traj = self.expert_trajs[i]
			act = self.expert_actions[i]
			traj_length = len(traj)
			for s1 in range(traj_length):
				csx, csy, cgx, cgy = traj[s1]
				action = action_to_one_hot(env, act[s1])
				for s2 in range(s1+1, traj_length):
					nsx, nsy, ngx, ngy = traj[s2]
					self._train_states.append([csx,csy,nsx,nsy])
					self._train_actions.append(action)
				self._train_states.append([csx,csy,cgx,cgy])
				self._train_actions.append(action)
		assert(len(self._train_states) == len(self._train_actions))

		self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		self._train_actions = np.array(self._train_actions) # size: (*, 4)

	def train(self, num_epochs=100, batch_size=256):
		""" 3
		Trains the model on training data generated by the expert policy.
		Args:
			num_epochs: number of epochs to train on the data generated by the expert.
			batch_size
		Return:
			loss: (float) final loss of the trained policy.
			acc: (float) final accuracy of the trained policy
		"""
		criterion = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.model.parameters())

		self._t_actions = np.argmax(self._train_actions, axis=1) 
		train_set = TensorDataset(torch.Tensor(self._train_states), torch.Tensor(self._t_actions).type(torch.long))
		train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)

		for epoch in range(num_epochs):
			running_loss = 0
			correct = 0
			for i, data in enumerate(train_loader, 0):

				x_batch, y_batch = data
				optimizer.zero_grad()
				yhat = self.model(x_batch)
				loss = criterion(yhat, y_batch)
				loss.backward()
				optimizer.step()

				correct += (torch.argmax(yhat, dim =1) == y_batch).float().sum()
				running_loss += loss.item()

			acc = correct / len(train_set)
			#print('(%d) loss= %.3f; running_loss= %.3f; accuracy = %.1f%%' % (epoch, loss, running_loss, 100 * acc))

		return loss.detach().numpy(), acc.detach().numpy()


def evaluate_gc(env, policy, n_episodes=50):
	succs = 0
	for _ in range(n_episodes):
		info = generate_gc_episode(env, policy)
		if info == 'succ':
			succs += 1
	succs = float(succs)/float(n_episodes)
	return succs


def generate_gc_episode(env, policy):
	"""Collects one rollout from the policy in an environment. The environment
	should implement the OpenAI Gym interface. A rollout ends when done=True. The
	number of states and actions should be the same, so you should not include
	the final state when done=True.
	Args:
		env: an OpenAI Gym environment.
		policy: a trained model
	Returns:
	"""
	done = False
	state = env.reset()
	state = torch.tensor(state, dtype=torch.int)
	softmax = torch.nn.Softmax(dim=-1)
	while not done:
		action = policy.model(state.float())
		action = softmax(action)
		action = np.argmax(action.detach().numpy())
		state, reward, done, info = env.step(action)
		state = torch.tensor(state, dtype=torch.int)
	return info


def run_GCBC(expert_trajs, expert_actions, mode = "vanilla"):
	gcbc = GCBC(env, expert_trajs, expert_actions)
	# mode = 'vanilla'
	# mode = 'expert_relabel'

	if mode == 'vanilla':
		gcbc.generate_behavior_cloning_data()
	else:
		gcbc.generate_relabel_data()

	#print(gcbc._train_states.shape)

	num_seeds = 2
	loss_vecs = []
	acc_vecs = []
	succ_vecs = []

	for i in range(num_seeds):
		print('*' * 50)
		print('seed: %d' % i)
		loss_vec = []
		acc_vec = []
		succ_vec = []
		gcbc.reset_model()

		for e in range(50):
			loss, acc = gcbc.train(num_epochs=20)
			succ = evaluate_gc(env, gcbc)
			loss_vec.append(loss)
			acc_vec.append(acc)
			succ_vec.append(succ)
			print(e, loss, acc, succ)
		loss_vecs.append(loss_vec)
		acc_vecs.append(acc_vec)
		succ_vecs.append(succ_vec)
	
	loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
	acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
	succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()
	
	### Plot the results
	plt.plot(np.arange(50), acc_vec, label=f"num_episodes={50}")
	plt.xlabel("# Iterations")
	plt.ylabel("Accuracy")
	plt.title(f"{mode.title()} Accuracy Comparison Plot")
	plt.legend()
	plt.savefig('p2_gcbc_%s.png' % mode, dpi=300)
	plt.close()

	plt.plot(np.arange(50), loss_vec, label=f"num_episodes={50}")
	plt.xlabel("# Iterations")
	plt.ylabel("Loss")
	plt.title(f"{mode.title()} Loss Comparison Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_comparison_loss.png")
	plt.close()

	plt.plot(np.arange(50), succ_vec, label=f"num_episodes={50}")
	plt.xlabel("# Iterations")
	plt.ylabel("success")
	plt.title(f"{mode.title()} success Comparison Plot")
	plt.legend()
	plt.savefig(f"plots/{'_'.join(mode.split())}_comparison_success.png")
	plt.close()

def generate_random_trajs():
	N = 1000
	random_trajs = []
	random_actions = []
	random_goals = []

	success = 0
	for i in range(1000):
		sub_trajs = []
		sub_actions = []
		sub_goals = []
		env.reset()
		done = False
		while not done:
			act = np.random.randint(4)
			sub_actions.append(act)
			obs, reward, done, info = env.step(act) 
			[s1,s2,g1,g2] = obs
			sub_trajs.append(obs)
			sub_goals.append(np.array([g1,g2]))
		random_trajs.append(sub_trajs)
		random_actions.append(sub_actions)
		random_goals.append(sub_goals)
		if info == "succ":
			success += 1
	print(f"success {success} rate {success/N}")
	run_GCBC(random_trajs, random_actions)

# expert_traj, expert_actions, unflat_trajs, unflat_actions = shortest_path_expert(env)

#vanilla
#run_GCBC(expert_traj, expert_actions)

#expert relabing
# run_GCBC(unflat_trajs, unflat_actions)
generate_random_trajs()