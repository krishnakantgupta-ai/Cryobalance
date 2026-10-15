#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

try:
    from scipy.special import sph_harm_y
    def eval_sph_harm(m, l, phi, theta):
        return sph_harm_y(l, m, theta, phi)
except ImportError:
    from scipy.special import sph_harm
    def eval_sph_harm(m, l, phi, theta):
        return sph_harm(m, l, phi, theta)

@dataclass
class OrientationMetrics:
    anisotropy_index: float
    sampling_entropy: float
    spherical_power: Dict[int, float]
    effective_particle_count: float

def generate_symmetry_matrices(sym: str) -> np.ndarray:
    sym = sym.upper()
    matrices = [np.eye(3)]
    if sym.startswith("C"):
        order = int(sym[1:])
        for k in range(1, order):
            ang = 2.0 * np.pi * k / order
            c, s = np.cos(ang), np.sin(ang)
            matrices.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
    elif sym.startswith("D"):
        order = int(sym[1:])
        for k in range(1, order):
            ang = 2.0 * np.pi * k / order
            c, s = np.cos(ang), np.sin(ang)
            matrices.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))
        rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for r in list(matrices):
            matrices.append(rx @ r)
    return np.stack(matrices, axis=0)

def angles_to_unit_vectors(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def unit_vectors_to_angles(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    theta = np.arccos(np.clip(vecs[:, 2], -1.0, 1.0))
    phi = np.arctan2(vecs[:, 1], vecs[:, 0]) % (2.0 * np.pi)
    return phi, theta

def expand_by_symmetry(phi: np.ndarray, theta: np.ndarray, sym: str = "C1") -> Tuple[np.ndarray, np.ndarray]:
    if sym.upper() == "C1":
        return phi, theta
    r_sym = generate_symmetry_matrices(sym)
    vecs = angles_to_unit_vectors(phi, theta)
    expanded = np.einsum("gij,nj->gni", r_sym, vecs).reshape(-1, 3)
    return unit_vectors_to_angles(expanded)

def fit_spherical_harmonics(phi: np.ndarray, theta: np.ndarray, l_max: int = 12):
    n = len(phi)
    coeffs = {}
    power_spec = {}
    for l in range(l_max + 1):
        sl = 0.0
        for m in range(-l, l + 1):
            ylm = eval_sph_harm(m, l, phi, theta)
            clm = (4.0 * np.pi / n) * np.sum(ylm)
            coeffs[(l, m)] = clm
            sl += np.abs(clm) ** 2
        sl /= (2 * l + 1)
        power_spec[l] = float(sl)
    return coeffs, power_spec

def evaluate_density_field(phi: np.ndarray, theta: np.ndarray, coeffs: Dict[Tuple[int, int], complex]) -> np.ndarray:
    density = np.zeros(len(phi), dtype=np.float64)
    for (l, m), clm in coeffs.items():
        ylm = eval_sph_harm(m, l, phi, theta)
        density += np.real(clm * np.conj(ylm))
    return density

def compute_metrics_and_weights(phi: np.ndarray, theta: np.ndarray, l_max: int = 12, sym: str = "C1", alpha: float = 0.02):
    n_orig = len(phi)
    phi_exp, theta_exp = expand_by_symmetry(phi, theta, sym=sym)
    coeffs, power_spec = fit_spherical_harmonics(phi_exp, theta_exp, l_max=l_max)
    f_est = evaluate_density_field(phi, theta, coeffs)
    f_max = np.max(f_est)
    f_clamped = np.maximum(f_est, 1e-4 * f_max)
    s0 = power_spec[0]
    anisotropic_power = sum((2 * l + 1) * power_spec[l] for l in range(1, l_max + 1))
    anisotropy_index = float(anisotropic_power / s0) if s0 > 0 else 0.0
    p = f_clamped / np.sum(f_clamped)
    entropy = -float(np.sum(p * np.log(p + 1e-12))) / np.log(n_orig)
    weights = 1.0 / (f_clamped + alpha * f_max)
    weights /= np.mean(weights)
    ess = float((np.sum(weights) ** 2) / np.sum(weights ** 2))
    return weights, OrientationMetrics(
        anisotropy_index=anisotropy_index,
        sampling_entropy=entropy,
        spherical_power=power_spec,
        effective_particle_count=ess
    )
