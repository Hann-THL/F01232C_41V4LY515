import numpy as np

def min_max_scale(x, min_value, max_value, precision=5):
    # Reference: https://www.codecademy.com/articles/normalization
    return np.round((x - min_value) / (max_value - min_value), precision)

def zero_one_scale(x, precision=5):
    # Reference: https://stackoverflow.com/questions/42140347/normalize-any-value-in-range-inf-inf-to-0-1-is-it-possible
    return np.round((1 + x / (1 + np.abs(x))) * .5, precision)