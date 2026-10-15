import numpy as np
from cryobalance.core import generate_symmetry_matrices, compute_metrics_and_weights

def test_symmetry():
    c3 = generate_symmetry_matrices("C3")
    assert c3.shape == (3, 3, 3)

def test_metrics():
    np.random.seed(42)
    phi = np.random.uniform(0, 2 * np.pi, 1000)
    theta = np.arccos(np.random.uniform(-1, 1, 1000))
    w, m = compute_metrics_and_weights(phi, theta, l_max=6, sym="C1")
    assert m.anisotropy_index < 0.25
    assert np.isfinite(w).all()
