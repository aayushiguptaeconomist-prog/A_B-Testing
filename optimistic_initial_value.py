import matplotlib.pyplot as plt
import numpy as np

from epsilon_greedy import experiment

# from epsilon_greedy import experiment

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]

class Bandit:
    def __init__ (self, p):
        self.p = p
        self.p_estimate = 5    # optimistic initial value is high
        self.N = 1  # if we do not initialize to 1, then the first update will erase any effect of the optimistic initial value 

    def pull(self):
        return np.random.random() < self.p  # generates rewards
        
    def update(self, x):
        self.N += 1
        self.p_estimate = self.p_estimate + ((x - self.p_estimate) / self.N)

def experiment():
        bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

        rewards = np.zeros(NUM_TRIALS)
        optimal_j = np.argmax(b.p for b in bandits)
        print("optimal j:", optimal_j)

        for i in range(NUM_TRIALS):
            j = np.argmax([b.p_estimate for b in bandits])
            # print(f"for index {i} and chosen bandit {j} : {[b.p_estimate for b in bandits]}")

            x = bandits[j].pull()
            # print(f"reward received: {x}")

            rewards[i] = x

            bandits[j].update(x)

        for b in bandits:
            print("mean estimate: ", b.p_estimate)

        print("total rewards earned: ", rewards.sum())
        print("overall rewards earned: ", rewards.sum()/ NUM_TRIALS)
        print("num of times selected each bandit: ", [b.N for b in bandits])

        cumulative_rewards = np.cumsum(rewards)
        win_rates = cumulative_rewards / (np.arange(1, NUM_TRIALS + 1))
        plt.ylim([0, 1])    # sets y-axis range from 0 to 1
        plt.plot(win_rates)
        # plt.plot(np.ones(NUM_TRIALS) * 5) # plots a horizontal line at the optimistic initial value
        plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))  # plots a horizontal line at the optimal win rate
        plt.legend(['Win Rate', 'Optimal Bandit'])
        plt.savefig("win_rates_optimistic_initial_value.png")


if __name__ == "__main__":
        experiment()

    