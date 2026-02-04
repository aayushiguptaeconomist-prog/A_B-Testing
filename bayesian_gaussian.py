# DO NOT FULLY UNDERSTAND THIS METHOD. NEED TO WORK ON IT.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]    # these are means, not probabilities

class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.predicted_mean = 0     # because we have no clue about the mean, we start with 0
        self.lambda_ = 1    # we don't start with 0 because that would result in an infinite variance implying that no amount of data can teach us anything and the observations carry no information. Using 1 says that tau is noisy but informative.  
        self.tau = 1    # the reason for initializing with 1 is the same as lambda. Also, our prior is standard normal as we have mean 0, and variance 1.
        self.sum_x = 0  # it will store the sum of all samples we collect
        self.N = 0      # number of times the bandit is played

    # def pull():
        
    

