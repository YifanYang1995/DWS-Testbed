import random

import numpy as np


# The Poisson Process: Everything you need to know
# https://towardsdatascience.com/the-poisson-process-everything-you-need-to-know-322aa0ab9e9a

# rate: number of arrivals per second, time: in how many seconds
def sample_poisson(rate, time):  # Sample from a possion process
    pos_array = []
    current = 0
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if current < time:
            pos_array.append(current)
        else:
            return pos_array

def one_sample_poisson(rate, startTime):
    current = startTime
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        return current

# sample a fixed number of data from the distribution
def num_sample_poisson(rate, startTime, num):
    pos_array = []
    current = startTime
    while True:
        pos = -(np.log(1 - random.random())) / rate
        current += pos
        if len(pos_array) < num:
            pos_array.append(current)
        else:
            return pos_array

# print(sample_possion(5/3600, 3600))
# x = np.random.poisson(0.1, 100)
# print(x, sum(x))'


def sample_poisson_shape(rate, shape):
    """Sample from a Poisson process to fit a specified shape.

    Args:
        rate (float): The lambda rate parameter of the Poisson process.
        shape (tuple): The shape of the output array: Stationnary.

    Returns:
        np.ndarray: An array of samples from the Poisson process with the specified shape.
    """
    samples = np.zeros(shape)

    for i in range(shape[0]):  # First dimension
        for j in range(shape[1]):  # Second dimension
            current = 0
            pos_array = []
            while len(pos_array) < shape[2]:  # Third dimension
                pos = -(np.log(1 - np.random.random())) / rate
                current += pos
                # if current < shape[2]:
                pos_array.append(current)
            samples[i, j, :] = np.array(pos_array)

    return samples

def sample_poisson_piecewise(rate_list, shape):
    """Sample from a Poisson process with piecewise-constant rates.

    Args:
    rate_list (list of float): A list of lambdas of length K, one lambda per segment.
    shape (tuple of int): The shape of the output array, in the format (M, N, T),
    where the third dimension T will be divided into K segments.

    Returns:
    np.ndarray: An array of cumulative arrival times of shape (M, N, T). non-stationary. monotonic increasing or decreasing.
    """
    M, N, T = shape
    K = len(rate_list)

    # Divide the sequence into segments; the final segment receives the remainder.
    base_seg = T // K
    seg_sizes = [base_seg] * K
    seg_sizes[-1] += T - base_seg * K

    samples = np.zeros((M, N, T), dtype=float)

    for i in range(M):
        for j in range(N):
            current = 0.0
            pos_array = []
            # Use the corresponding arrival rate for this segment.
            for seg_idx, lam in enumerate(rate_list):
                count = seg_sizes[seg_idx]
                for _ in range(count):
                    # Exponentially distributed inter-arrival interval.
                    inter_arrival = -np.log(1 - np.random.random()) / lam
                    current += inter_arrival
                    pos_array.append(current)
            samples[i, j, :] = pos_array

    return samples
