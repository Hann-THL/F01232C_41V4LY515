import numpy as np

def min_max_scale(x, min_value, max_value, precision=5):
    # Reference: https://www.codecademy.com/articles/normalization
    return np.round((x - min_value) / (max_value - min_value), precision)

def zero_one_scale(x, precision=5):
    # Reference: https://stackoverflow.com/questions/42140347/normalize-any-value-in-range-inf-inf-to-0-1-is-it-possible
    return np.round((1 + x / (1 + np.abs(x))) * .5, precision)

def clipping(x, max_value):
    # Positive
    if x > round(max_value * .875, 0):
        return 1

    if x > round(max_value * .75, 0):
        return .875

    if x > round(max_value * .625, 0):
        return .75

    if x > round(max_value * .5, 0):
        return .625

    if x > round(max_value * .375, 0):
        return .5

    if x > round(max_value * .25, 0):
        return .375

    if x > round(max_value * .125, 0):
        return .25

    if x > 0:
        return .125

    # Negative
    if x < round(-max_value * .875, 0):
        return -1

    if x > round(-max_value * .75, 0):
        return -.875

    if x > round(-max_value * .625, 0):
        return -.75

    if x < round(-max_value * .5, 0):
        return -.625

    if x > round(-max_value * .375, 0):
        return -.5

    if x < round(-max_value * .25, 0):
        return -.375

    if x > round(-max_value * .125, 0):
        return -.25

    if x < 0:
        return -.125

    # Neutral
    return 0