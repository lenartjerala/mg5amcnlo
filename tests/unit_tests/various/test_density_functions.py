################################################################################
#
# Regression tests for the triangular density-matrix convention shared by the
# generated Fortran GET_INTER routines and Density_functions.py.
#
################################################################################
from __future__ import division

import os
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
        os.pardir,
        os.pardir,
    )
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import madgraph.various.Density_functions as density_functions  # noqa: E402
import madgraph.various.fxfx_ewsudakov as fx  # noqa: E402


def _upper_triangle(matrix):
    return [
        matrix[i, j]
        for i in range(matrix.shape[0])
        for j in range(i, matrix.shape[1])
    ]


class TestFortranTriangleConvention(unittest.TestCase):
    def test_upper_triangle_is_not_conjugated(self):
        """GET_INTER stores B[i,j]=M_i*conj(M_j) for i<=j."""
        rng = np.random.default_rng(20260728)
        amplitudes = rng.normal(size=9) + 1j * rng.normal(size=9)
        expected = np.outer(amplitudes, np.conj(amplitudes))

        reconstructed = np.asarray(
            density_functions.DensityMatrixObservables(
                _upper_triangle(expected)
            ).square_matrix()
        )

        np.testing.assert_allclose(reconstructed, expected, rtol=1e-13, atol=1e-13)
        self.assertGreater(
            np.max(np.abs(reconstructed - np.conj(expected))),
            1.0e-6,
        )

    def test_complex_sudakov_weight_uses_delta_not_conjugate_delta(self):
        """The fixed B/C reconstruction closes the complex-delta contraction."""
        rng = np.random.default_rng(17)
        prod_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
        decay_amp = rng.normal(size=9) + 1j * rng.normal(size=9)
        b_true = np.outer(prod_amp, np.conj(prod_amp))
        c_true = np.outer(decay_amp, np.conj(decay_amp))
        # Keep the contraction safely away from zero.
        c_true += 0.25 * np.trace(c_true).real / 9.0 * np.eye(9)
        delta = 0.08 * rng.normal(size=9) + 0.05j * rng.normal(size=9)

        b_reconstructed = np.asarray(
            density_functions.DensityMatrixObservables(
                _upper_triangle(b_true)
            ).square_matrix()
        )
        c_reconstructed = np.asarray(
            density_functions.DensityMatrixObservables(
                _upper_triangle(c_true)
            ).square_matrix()
        )

        def weight(b_matrix, c_matrix, correction):
            b = fx.DensityMatrix.from_full_matrix(b_matrix, [24, -24])
            c = fx.DensityMatrix.from_full_matrix(c_matrix, [24, -24])
            return b.apply_sudakov(correction).contract(c) / b.contract(c)

        expected = weight(b_true, c_true, delta)
        reconstructed = weight(b_reconstructed, c_reconstructed, delta)
        old_conjugated_result = weight(
            np.conj(b_true), np.conj(c_true), delta
        )
        conjugated_delta_result = weight(
            b_true, c_true, np.conj(delta)
        )
        real_delta_result = weight(b_true, c_true, np.real(delta))
        old_real_delta_result = weight(
            np.conj(b_true), np.conj(c_true), np.real(delta)
        )

        self.assertAlmostEqual(reconstructed, expected, places=13)
        self.assertGreater(abs(old_conjugated_result - expected), 1.0e-7)
        self.assertAlmostEqual(
            old_conjugated_result, conjugated_delta_result, places=13
        )
        self.assertAlmostEqual(
            old_real_delta_result, real_delta_result, places=13
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
