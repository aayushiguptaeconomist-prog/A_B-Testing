import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0

    def pull(self):
        return np.random.random() < self.p
    
    def update(self, x):
        self.N += 1
        self.p_estimate = self.p_estimate + ((x - self.p_estimate) / self.N)

def ucb(mean, n, nj):   # n is total no. of times any bandit has been played, nj is no. of times this bandit has been played
    return mean + np.sqrt(2 * (np.log(n) / nj))     # we use this when selecting the argmax bandit

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    rewards = np.zeros([NUM_TRIALS])
    total_plays = 0

    for j in range(len(bandits)):   # this loop exists only ensure nj > 0 and avoid division by 0 in ucb function
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        # We don’t update the rewards array in the initialization loop because those pulls are not part of the learning curve we want to plot.

    for i in range(NUM_TRIALS):
        j = np.argmax([ucb(b.p_estimate, total_plays, b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        rewards[i] = x

    for b in bandits:
        print("mean estimate: ", b.p_estimate)

    print("total rewards earned: ", rewards.sum())
    print("overall rewards earned: ", rewards.sum() / NUM_TRIALS)

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.legend(['Win Rate', 'Optimal Bandit'])
    plt.savefig("win_rates_upper_confidence_bound.png")

if __name__ == "__main__":
    experiment()



