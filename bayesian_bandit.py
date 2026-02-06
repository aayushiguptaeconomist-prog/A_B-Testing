import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1 # alpha = number of wins + 1 (prior)
        self.b = 1 # beta = number of losses + 1 (prior)
        self.N = 0

    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)  # generates a sample from the beta distribution with parameters a and b
    
    def update(self, x):
        self.N += 1
        self.a += x   # a gets incremented by the reward (1 or 0) as it is the success
        self.b += (1-x)   # b gets incremented by 1 - reward (0 or 1) as it is the failure

def plot(bandits, trials):
    x = np.linspace(0, 1, 200)  # it is creating a grid of 200 evenly spaced evaluation points b/w 0 and 1. These points will be used to evaluate the curve of the beta distribution at that particular point.
    for b in bandits:   # this for loop plots all three bandits in the same graph, the rest (title etc) is common for all
        y = beta.pdf(x, b.a, b.b)   # using the parameters defined, this generates the beta distribution for each bandit
        plt.plot(x, y, label = f"real p: {b.p:.4f}, win rate: {b.a - 1}/{b.N}") # shows the true probability rounded to 4 decimal points and also the win rate (where a - 1 = # of successes)
    plt.title(f"Bandit distributions after {trials} trials")
    plt.legend()
    plt.savefig(f"posterior_dist_at_{trials}.png")

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
    sample_points = [5, 10, 20, 50, 100, 200, 500, 1000, 1500, 1999]    # these are time steps where we want to plot the posterior distributions
    rewards = np.zeros(NUM_TRIALS)

    for i in range(NUM_TRIALS):
        j = np.argmax([b.sample() for b in bandits])    # thompson sampling - draws a probability estimate from the posterior distribution

        if i in sample_points:  # plot the posterior distribution for each bandit
            plot(bandits, i)

        x = bandits[j].pull()
        bandits[j].update(x)
        rewards[i] = x

    for b in bandits:
        print("number of success: ", b.a - 1)

    print("total rewards", rewards.sum())   # rewards does not include the the initiated success values. Therefore, no need to subtract anything from it.
    print("overall win rate: ", rewards.sum() / NUM_TRIALS)
    print("number of times each bandit was selected: ", [b.N for b in bandits])

if __name__ == "__main__":
    experiment()
