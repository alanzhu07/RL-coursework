# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import gym
import functools


def cmaes(fn, dim, num_iter=10):
  """Optimizes a given function using CMA-ES.

  Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.

  Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
  """
  # Hyperparameters
  sigma = 10
  population_size = 100
  p_keep = 0.10  # Fraction of population to keep
  noise = 0.25  # Noise added to covariance to prevent it from going to 0.
  num_keep = int(population_size*p_keep)

  # Initialize the mean and covariance
  mu = np.zeros(dim)
  cov = sigma**2 * np.eye(dim)
  
  mu_vec = []
  best_sample_vec = []
  mean_sample_vec = []
  for t in range(num_iter):
    sample = np.random.multivariate_normal(mu, cov, population_size)
    scores = np.array([fn(samp) for samp in sample])
    elites = sample[np.argsort(scores)[-num_keep:]]
    elite_scores = np.sort(scores)[-num_keep:]

    mu = np.mean(elites, axis=0)
    cov = np.cov(elites, rowvar=False) + noise*np.identity(dim)

    mu_vec.append(mu)
    best_sample_vec.append(np.max(scores))
    mean_sample_vec.append(np.mean(scores))

  return mu_vec, best_sample_vec, mean_sample_vec

def test_fn(x):
  goal = np.array([65, 49])
  return -np.sum((x - goal)**2)

mu_vec, best_sample_vec, mean_sample_vec = cmaes(test_fn, dim=2, num_iter=100)

x = np.stack(np.meshgrid(np.linspace(-10, 100, 30), np.linspace(-10, 100, 30)), axis=-1)
fn_value = [test_fn(xx) for xx in x.reshape((-1, 2))]
fn_value = np.array(fn_value).reshape((30, 30))
plt.figure(figsize=(6, 4))
plt.contourf(x[:, :, 0], x[:, :, 1], fn_value, levels=10)
plt.colorbar()
mu_vec = np.array(mu_vec)
plt.plot(mu_vec[:, 0], mu_vec[:, 1], 'b-o')
plt.plot([mu_vec[0, 0]], [mu_vec[0, 1]], 'r+', ms=20, label='initial value')
plt.plot([mu_vec[-1, 0]], [mu_vec[-1, 1]], 'g+', ms=20, label='final value')
plt.plot([65], [49], 'kx', ms=20, label='maximum')
plt.legend()
plt.show()

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, params):
  w = params[:4]
  b = params[4]
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

def rl_fn(params, env):
  assert len(params) == 5
  state = env.reset()
  terminated = False
  rewards = []
  while not terminated:
      action = _get_action(state, params)
      state, reward, terminated, _ = env.step(action)
      rewards.append(reward)
  total_rewards = sum(rewards)

  return total_rewards

env = gym.make('CartPole-v0')
params = np.array([1,0,1,0,1]) 
out = []
for i in range(1000):
  out.append(rl_fn(params,env))
print(np.average(out))

env = gym.make('CartPole-v0')
fn_with_env = functools.partial(rl_fn, env=env)
mu_vec, best_sample_vec, mean_sample_vec = cmaes(fn_with_env, dim=5, num_iter=10)

plt.plot(mean_sample_vec)
plt.plot(best_sample_vec)
plt.legend(['mean', 'best'])
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('CMA-ES')
plt.show()