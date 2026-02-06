import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, beta

true_CTR = 0.5  # CTR = Click Through Rate
N = 501
plot_indices = [10, 21, 50, 100, 121, 250, 500]
a = 1   # beta distribution parameter with no prior information
b = 1 

rewards = np.empty(N)

for i in range(N):
    x = 1 if np.random.random() < true_CTR else 0
    rewards[i] = x  

    a += x  # update the success
    b += (1-x)  # update the failure

    if i in plot_indices:

        mean = rewards[:i+1].mean()
        std = np.sqrt((mean * (1 - mean)) / (i+1))

        x = np.linspace(0, 1, 200)
        normal = norm.pdf(x, loc = mean, scale = std)
        plt.plot(x, normal, label = "Gaussian Approximation")

        posterior = beta.pdf(x, a=a, b=b)
        plt.plot(x, posterior, label = "Beta Posterior")

        plt.legend()
        plt.title("N = %s" % (i+1))
        plt.show()

# As you move closer to N = 501, you see that the normal and beta distribution almost converge proving the Central Limit Theorem.
# This project also shocases the comparison between "Gaussian Approximation" and "Beta Posterior".
