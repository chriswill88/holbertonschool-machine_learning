#!/usr/bin/env python3
"""this task contains a function used in task 5"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    variable using the gradient descent with momentum optimization algorithm:

    @alpha
        is the learning rate
    @beta1
        is the momentum weight
    @var
        is a numpy.ndarray containing the variable to be updated
    @grad
        is a numpy.ndarray containing the gradient of var
    @v
        is the previous first moment of var
    """

    vdv = beta1 * v + (1 - beta1) * grad
    var = var - alpha * vdv
    return var, vdv
