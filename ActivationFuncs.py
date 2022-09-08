import math
import numpy as np


def sig(z):
    return 1 / (1 + math.e ** -z)


def d_sig(activation):
    z = sig(activation) * (1 - sig(activation))
    return sig(z) * (1 - sig(z))


def tanh(z):  # distributes input z to a range from -1 to 1
    return (math.e ** z - math.e ** -z) / (math.e ** z + math.e ** -z)


def d_tanh(z):  # finds the derivative of z
    return 1 - tanh(z) ** 2


def inv_tanh(activation):  # given an activation that has been tanh'd, returns the activation before that process
    return (np.log(1 + activation) - (np.log(1 - activation))) / 2


def softplus(z):
    return math.log(1 + math.e ** z, math.e)


def d_softplus(activation):
    z = sig(activation)
    return sig(z)


def elu(z, alpha):  # Exponential Linear Unit
    if z <= 0:
        return alpha(math.e ** z - 1)
    else:
        return z


def d_elu(activation, alpha):
    if activation > 0:
        z = 1
    else:
        z = alpha * math.e ** activation
    if z > 0:
        return 1
    else:
        return alpha * math.e ** z


def relu(z):  # Rectified Linear Unit
    return max(z, 0)


def d_relu(activation):
    if activation > 0:
        return 1
    else:
        return 0  # technically, this isn't true if z = 0, but this avoids errors


def gaussian(z):
    return math.e ** (-z ** 2)


def d_gaussian(activation):
    z = -2 * activation * math.e ** (-activation ** 2)
    return -2 * z * math.e ** (-z ** 2)


def swish(z):
    return z / (1 + math.e ** -z)


def d_swish(activation):
    z = (1 + math.e ** -activation + activation * math.e ** -activation) / (1 + math.e ** -activation) ** 2
    return (1 + math.e ** -z + z * math.e ** -z) / (1 + math.e ** -z) ** 2
