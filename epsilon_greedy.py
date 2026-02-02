import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10000
EPS = 0.1   # probability of exploration (random) => 0.9 is the probability of exploitation (highest win rate)
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]  # true success probabilities for each bandit (here bandit equals to a slot machine with a lever)

class Bandit:
    def __init__ (self, p):
        self.p = p  # true probability of winning
        self.p_estimate = 0 # estimated probability of winning
        self.N = 0  # number of times a bandit is pulled

    def pull(self):
        return np.random.random() < self.p  # np.random.random() generates a random no. b/w 0 and 1. Then, it returns True(1) if < p else False(0).
    
    def update(self, x):
        self.N += 1
        self.p_estimate = self.p_estimate + ((x - self.p_estimate) / self.N)    # this is basically calculating the average win rate

def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    # initialize a few variables
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0 # number of times the algorithm chose the best bandit
    optimal_j = np.argmax([b.p for b in bandits])   # index of the best bandit
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):
        # use epsilon-greedy to select the bandit

        # if else condition is just to decide whether to explore or exploit and choose a bandit
        if np.random.random() < EPS:
            num_times_explored += 1
            j = np.random.randint(len(bandits)) # select a random bandit by generating a random no. b/w 0 and len(bandits)-1

        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])
            print(f"for index {i}: {[b.p_estimate for b in bandits]}")

        # if the selected bandit is the optimal bandit, increment num_optimal
        if j == optimal_j:
            num_optimal += 1

        # pull the arm for the bandit with the largest sample mean
        x = bandits[j].pull()

        # update rewards log
        rewards[i] = x

        bandits[j].update(x)

    # print the mean estimates for each bandit
    for b in bandits:
        print("mean estimate: ", b.p_estimate)

    print("Total reward earned: ", rewards.sum())
    print("Overall win rate: ", rewards.sum() / NUM_TRIALS)
    print("Number of times explored: ", num_times_explored)
    print("Number of times exploited: ", num_times_exploited)
    print("Number of times selected optimal bandit: ", num_optimal)

    # plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)    # adding 1 to an array basically add to each element of the array which avoids division by 0 as the first index
    print("cumulative win rates: ", win_rates)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.legend(['Win Rate', 'Optimal Bandit'])
    plt.savefig("win_rates_epsilon_greedy.png")

if __name__ == "__main__":
    experiment()



