import numpy as np
import matplotlib.pyplot as plt
import tqdm

def epsilon_greedy(epsilon, k=10, time_steps=1000):
    # instantiate mean reward
    true_reward = np.random.normal(1,1,k)

    # initialize estimates
    estimates = np.zeros(k)
    counts = np.zeros(k)
    expected_rewards = np.zeros(time_steps)
    rewards = np.zeros(time_steps)
    for i in range(time_steps):
        best_action = np.argmax(estimates)
        if np.random.uniform() <= 1-epsilon:
            action = best_action
        else:
            action = np.random.choice(k)
        reward = np.random.normal(true_reward[action],1)
        counts[action] += 1
        estimates[action] += (1/counts[action]) * (reward - estimates[action])

        expected_rewards[i] = true_reward[best_action] * (1-epsilon) + np.mean(true_reward) * (epsilon)
        rewards[i] = reward
    return expected_rewards

def optimistic_initialization(init, k=10, time_steps=1000):
    # instantiate mean reward
    true_reward = np.random.normal(1,1,k)

    # initialize estimates
    estimates = np.full(k, init)
    counts = np.zeros(k)
    expected_rewards = np.zeros(time_steps)
    rewards = np.zeros(time_steps)
    for i in range(time_steps):
        best_action = np.argmax(estimates)
        action = best_action
        reward = np.random.normal(true_reward[action],1)
        counts[action] += 1
        estimates[action] += (1/counts[action]) * (reward - estimates[action])

        expected_rewards[i] = true_reward[best_action]
        rewards[i] = reward
    return expected_rewards

def ucb(c, k=10, time_steps=1000):
    # instantiate mean reward
    true_reward = np.random.normal(1,1,k)

    # initialize estimates
    estimates = np.zeros(k)
    counts = np.zeros(k)
    expected_rewards = np.zeros(time_steps)
    rewards = np.zeros(time_steps)
    for i in range(time_steps):
        vals = [(estimates[action] + c * np.sqrt(np.log(i+1) / counts[action])) if counts[action] != 0 else np.inf for action in range(k)]
        best_action = np.argmax(vals)
        action = best_action
        
        reward = np.random.normal(true_reward[action],1)
        counts[action] += 1
        estimates[action] += (1/counts[action]) * (reward - estimates[action])

        expected_rewards[i] = true_reward[best_action]
        rewards[i] = reward
    return expected_rewards

def boltzmann(t, k=10, time_steps=1000):
    # instantiate mean reward
    true_reward = np.random.normal(1,1,k)

    # initialize estimates
    estimates = np.zeros(k)
    counts = np.zeros(k)
    expected_rewards = np.zeros(time_steps)
    rewards = np.zeros(time_steps)
    for i in range(time_steps):
        prob = [np.exp(t * estimates[action]) for action in range(k)]
        sample_prob = [p/sum(prob) for p in prob]
        action = np.random.choice(k, p=sample_prob)
        
        reward = np.random.normal(true_reward[action],1)
        counts[action] += 1
        estimates[action] += (1/counts[action]) * (reward - estimates[action])

        expected_rewards[i] = np.dot(sample_prob, true_reward)
        rewards[i] = reward
    return expected_rewards

if __name__ == "__main__":
    print("epsilon greedy")
    eps = [0, 0.001, 0.01, 0.1, 1.]
    for epsilon in eps:
        print("epsilon =", epsilon)
        expected_rewards = []
        for _ in tqdm.tqdm(range(1000)):
            expected_rewards.append(epsilon_greedy(epsilon))
        avg_expected = np.mean(expected_rewards, axis=0)
        plt.plot(np.arange(1,1001), avg_expected, label="epsilon={}".format(epsilon))
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Reward")
    plt.savefig("epsilon.png")
    plt.close()
    # best parameter epsilon = 0.1

    print("optimistic initialization")
    inits = [0., 1., 2., 5., 10.]
    for init in inits:
        print("init =", init)
        expected_rewards = []
        for _ in tqdm.tqdm(range(1000)):
            expected_rewards.append(optimistic_initialization(init))
        avg_expected = np.mean(expected_rewards, axis=0)
        plt.plot(np.arange(1,1001), avg_expected, label="init={}".format(init))
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Reward")
    plt.savefig("optimistic.png")
    plt.close()
    # best param init = 10

    print("ucb")
    cs = [0., 1., 2., 5.]
    for c in cs:
        print("c =", c)
        expected_rewards = []
        for _ in tqdm.tqdm(range(1000)):
            expected_rewards.append(ucb(c))
        avg_expected = np.mean(expected_rewards, axis=0)
        plt.plot(np.arange(1,1001), avg_expected, label="c={}".format(c))
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Reward")
    plt.savefig("ucb.png")
    plt.close()
    # best param c = 1

    print("boltzmann")
    ts = [1., 3., 10., 30., 100.]
    for t in ts:
        print("t =", t)
        expected_rewards = []
        for _ in tqdm.tqdm(range(1000)):
            expected_rewards.append(boltzmann(t))
        avg_expected = np.mean(expected_rewards, axis=0)
        plt.plot(np.arange(1,1001), avg_expected, label="t={}".format(t))
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Reward")
    plt.savefig("boltzmann.png")
    plt.close()
    # best param t = 3

    print("overall")
    epsilon = 0.1
    expected_rewards = []
    for _ in tqdm.tqdm(range(1000)):
        expected_rewards.append(epsilon_greedy(epsilon))
    avg_expected = np.mean(expected_rewards, axis=0)
    plt.plot(np.arange(1,1001), avg_expected, label="epsilon-greedy, epsilon={}".format(epsilon))
    
    init = 10.
    expected_rewards = []
    for _ in tqdm.tqdm(range(1000)):
        expected_rewards.append(optimistic_initialization(init))
    avg_expected = np.mean(expected_rewards, axis=0)
    plt.plot(np.arange(1,1001), avg_expected, label="optimistic initialization, init={}".format(init))

    c = 1.
    expected_rewards = []
    for _ in tqdm.tqdm(range(1000)):
        expected_rewards.append(ucb(c))
    avg_expected = np.mean(expected_rewards, axis=0)
    plt.plot(np.arange(1,1001), avg_expected, label="ucb, c={}".format(c))

    t = 3.
    expected_rewards = []
    for _ in tqdm.tqdm(range(1000)):
        expected_rewards.append(boltzmann(t))
    avg_expected = np.mean(expected_rewards, axis=0)
    plt.plot(np.arange(1,1001), avg_expected, label="boltzmann, t={}".format(t))

    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Reward")
    plt.savefig("overall.png")
    plt.close()
