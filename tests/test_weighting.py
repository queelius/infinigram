#!/usr/bin/env python3
"""
Tests for weighting functions used in hierarchical suffix matching.
"""

import pytest
import numpy as np

from infinigram.weighting import (
    linear_weight,
    quadratic_weight,
    exponential_weight,
    sigmoid_weight,
)


class TestLinearWeight:
    """Test linear weighting function."""

    def test_linear_weight_at_zero(self):
        """Linear weight at 0 should be 0."""
        assert linear_weight(0) == 0.0

    def test_linear_weight_at_one(self):
        """Linear weight at 1 should be 1."""
        assert linear_weight(1) == 1.0

    def test_linear_weight_scales_linearly(self):
        """Linear weight should scale linearly."""
        assert linear_weight(5) == 5.0
        assert linear_weight(10) == 10.0

    def test_linear_weight_monotonic(self):
        """Linear weight should be monotonically increasing."""
        weights = [linear_weight(i) for i in range(10)]
        assert weights == sorted(weights)


class TestQuadraticWeight:
    """Test quadratic weighting function."""

    def test_quadratic_weight_at_zero(self):
        """Quadratic weight at 0 should be 0."""
        assert quadratic_weight(0) == 0.0

    def test_quadratic_weight_at_one(self):
        """Quadratic weight at 1 should be 1."""
        assert quadratic_weight(1) == 1.0

    def test_quadratic_weight_scales_quadratically(self):
        """Quadratic weight should scale quadratically."""
        assert quadratic_weight(2) == 4.0
        assert quadratic_weight(3) == 9.0
        assert quadratic_weight(5) == 25.0

    def test_quadratic_weight_monotonic(self):
        """Quadratic weight should be monotonically increasing."""
        weights = [quadratic_weight(i) for i in range(10)]
        assert weights == sorted(weights)

    def test_quadratic_grows_faster_than_linear(self):
        """Quadratic should grow faster than linear for k > 1."""
        for k in range(2, 10):
            assert quadratic_weight(k) > linear_weight(k)


class TestExponentialWeight:
    """Test exponential weighting function."""

    def test_exponential_weight_default_base(self):
        """Test exponential with default base 2."""
        weight_fn = exponential_weight()
        assert weight_fn(0) == 1.0  # 2^0
        assert weight_fn(1) == 2.0  # 2^1
        assert weight_fn(2) == 4.0  # 2^2
        assert weight_fn(3) == 8.0  # 2^3

    def test_exponential_weight_custom_base(self):
        """Test exponential with custom base."""
        weight_fn = exponential_weight(base=3.0)
        assert weight_fn(0) == 1.0  # 3^0
        assert weight_fn(1) == 3.0  # 3^1
        assert weight_fn(2) == 9.0  # 3^2

    def test_exponential_weight_monotonic(self):
        """Exponential weight should be monotonically increasing."""
        weight_fn = exponential_weight()
        weights = [weight_fn(i) for i in range(10)]
        assert weights == sorted(weights)

    def test_exponential_grows_fastest(self):
        """Exponential should grow faster than quadratic for large k."""
        weight_fn = exponential_weight()
        for k in range(5, 10):
            assert weight_fn(k) > quadratic_weight(k)


class TestSigmoidWeight:
    """Test sigmoid weighting function."""

    def test_sigmoid_weight_shape(self):
        """Sigmoid should have S-curve shape."""
        weight_fn = sigmoid_weight(midpoint=5, steepness=1.0)

        # Should be low at start
        assert weight_fn(0) < 0.1

        # Should be around 0.5 at midpoint
        assert abs(weight_fn(5) - 0.5) < 0.1

        # Should be high at end
        assert weight_fn(10) > 0.9

    def test_sigmoid_weight_midpoint(self):
        """Sigmoid should be 0.5 at midpoint."""
        weight_fn = sigmoid_weight(midpoint=7, steepness=1.0)
        assert abs(weight_fn(7) - 0.5) < 0.01

    def test_sigmoid_weight_steepness(self):
        """Higher steepness should make transition sharper."""
        gentle = sigmoid_weight(midpoint=5, steepness=0.5)
        steep = sigmoid_weight(midpoint=5, steepness=2.0)

        # At midpoint - 2, steep should be lower
        assert steep(3) < gentle(3)

        # At midpoint + 2, steep should be higher
        assert steep(7) > gentle(7)

    def test_sigmoid_weight_bounds(self):
        """Sigmoid should be bounded between 0 and 1."""
        weight_fn = sigmoid_weight(midpoint=5, steepness=1.0)
        for k in range(20):
            w = weight_fn(k)
            assert 0.0 <= w <= 1.0

    def test_sigmoid_weight_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        weight_fn = sigmoid_weight(midpoint=5, steepness=1.0)
        weights = [weight_fn(i) for i in range(20)]
        assert weights == sorted(weights)


class TestWeightComparison:
    """Compare different weight functions."""

    def test_all_weights_at_zero(self):
        """All weight functions should handle k=0."""
        assert linear_weight(0) == 0.0
        assert quadratic_weight(0) == 0.0
        assert exponential_weight()(0) == 1.0  # Exponential is different
        assert sigmoid_weight()(0) < 0.1  # Sigmoid starts low

    def test_all_weights_monotonic(self):
        """All weight functions should be monotonically increasing."""
        for weight_fn in [
            linear_weight,
            quadratic_weight,
            exponential_weight(),
            sigmoid_weight(),
        ]:
            weights = [weight_fn(i) for i in range(10)]
            assert weights == sorted(weights), f"Failed for {weight_fn}"

    def test_weight_ranking_at_large_k(self):
        """At large k, exponential > quadratic > linear."""
        k = 8
        assert exponential_weight()(k) > quadratic_weight(k) > linear_weight(k)


class TestGetWeightFunction:
    """Test get_weight_function utility."""

    def test_get_weight_function_linear(self):
        """Test getting linear weight function by name."""
        from infinigram.weighting import get_weight_function
        fn = get_weight_function("linear")
        assert fn(5) == 5.0

    def test_get_weight_function_quadratic(self):
        """Test getting quadratic weight function by name."""
        from infinigram.weighting import get_weight_function
        fn = get_weight_function("quadratic")
        assert fn(3) == 9.0

    def test_get_weight_function_exponential(self):
        """Test getting exponential weight function by name."""
        from infinigram.weighting import get_weight_function
        fn = get_weight_function("exponential")
        assert fn(2) == 4.0  # 2^2

    def test_get_weight_function_invalid_name_raises(self):
        """Test that invalid weight function name raises ValueError."""
        from infinigram.weighting import get_weight_function
        with pytest.raises(ValueError, match="Unknown weight function"):
            get_weight_function("nonexistent")

    def test_get_weight_function_error_lists_available(self):
        """Test error message lists available functions."""
        from infinigram.weighting import get_weight_function
        try:
            get_weight_function("bad_function_name")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_msg = str(e).lower()
            assert "linear" in error_msg, "Error should list available functions"
            assert "quadratic" in error_msg, "Error should list available functions"

    def test_get_weight_function_case_sensitive(self):
        """Test weight function names are case-sensitive."""
        from infinigram.weighting import get_weight_function
        with pytest.raises(ValueError):
            get_weight_function("Linear")  # Should be lowercase
