#!/usr/bin/env python3
"""
Weighting functions for hierarchical suffix matching.

These functions determine how much to weight predictions from different
suffix lengths. Longer matches typically provide more specific predictions,
so they should generally receive higher weights.
"""

import numpy as np
from typing import Callable


def linear_weight(length: int) -> float:
    """
    Linear weighting: w(k) = k

    Weights increase linearly with match length. Simple and interpretable.

    Args:
        length: Suffix match length

    Returns:
        Weight value

    Example:
        >>> linear_weight(0)
        0.0
        >>> linear_weight(5)
        5.0
    """
    return float(length)


def quadratic_weight(length: int) -> float:
    """
    Quadratic weighting: w(k) = k^2

    Weights increase quadratically with match length. Gives much stronger
    preference to longer matches than linear weighting.

    Args:
        length: Suffix match length

    Returns:
        Weight value

    Example:
        >>> quadratic_weight(0)
        0.0
        >>> quadratic_weight(5)
        25.0
    """
    return float(length ** 2)


def exponential_weight(base: float = 2.0) -> Callable[[int], float]:
    """
    Exponential weighting: w(k) = base^k

    Weights increase exponentially with match length. Provides very strong
    preference for longer matches. Can grow very quickly.

    Args:
        base: Exponential base (default 2.0)

    Returns:
        Weight function

    Example:
        >>> weight_fn = exponential_weight(2.0)
        >>> weight_fn(0)
        1.0
        >>> weight_fn(5)
        32.0
    """
    def weight(length: int) -> float:
        return base ** length

    return weight


def sigmoid_weight(midpoint: int = 5, steepness: float = 1.0) -> Callable[[int], float]:
    """
    Sigmoid weighting: w(k) = 1 / (1 + exp(-steepness * (k - midpoint)))

    S-curve weighting that transitions from low to high weight around a midpoint.
    Useful when you want to strongly favor matches above a certain length threshold.

    Args:
        midpoint: Length at which weight = 0.5
        steepness: How sharp the transition is (higher = sharper)

    Returns:
        Weight function

    Example:
        >>> weight_fn = sigmoid_weight(midpoint=5, steepness=1.0)
        >>> weight_fn(0)  # doctest: +ELLIPSIS
        0.006...
        >>> weight_fn(5)  # doctest: +ELLIPSIS
        0.5...
        >>> weight_fn(10)  # doctest: +ELLIPSIS
        0.993...
    """
    def weight(length: int) -> float:
        return 1.0 / (1.0 + np.exp(-steepness * (length - midpoint)))

    return weight


# Commonly used weight functions
WEIGHT_FUNCTIONS = {
    'linear': linear_weight,
    'quadratic': quadratic_weight,
    'exponential': exponential_weight(),
    'sigmoid': sigmoid_weight(),
}


def get_weight_function(name: str) -> Callable[[int], float]:
    """
    Get a weight function by name.

    Args:
        name: Weight function name ('linear', 'quadratic', 'exponential', 'sigmoid')

    Returns:
        Weight function

    Raises:
        ValueError: If name is not recognized

    Example:
        >>> weight_fn = get_weight_function('quadratic')
        >>> weight_fn(3)
        9.0
    """
    if name not in WEIGHT_FUNCTIONS:
        raise ValueError(
            f"Unknown weight function '{name}'. "
            f"Available: {list(WEIGHT_FUNCTIONS.keys())}"
        )
    return WEIGHT_FUNCTIONS[name]
