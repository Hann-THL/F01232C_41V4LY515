import numpy as np

def min_max_scale(x, min_value, max_value, precision=5):
    # Reference: https://www.codecademy.com/articles/normalization
    return np.round((x - min_value) / (max_value - min_value), precision)

def zero_one_scale(x, precision=5):
    # Reference: https://stackoverflow.com/questions/42140347/normalize-any-value-in-range-inf-inf-to-0-1-is-it-possible
    return np.round((1 + x / (1 + np.abs(x))) * .5, precision)

def clipping(x, max_value):
    factor     = -1 if x < 0 else 1
    percentage = abs(round(x / max_value, 3))

    if percentage > .875:
        return 1 * factor

    if percentage > .75:
        return .875 * factor

    if percentage > .625:
        return .75 * factor

    if percentage > .5:
        return .625 * factor

    if percentage > .375:
        return .5 * factor

    if percentage > .25:
        return .375 * factor

    if percentage > .125:
        return .25 * factor

    if percentage > 0:
        return .125 * factor

    return 0