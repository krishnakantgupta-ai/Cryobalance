# Cryobalance

[![PyPI version](https://img.shields.io/pypi/v/cryobalance.svg)](https://pypi.org/project/cryobalance/)
[![Python versions](https://img.shields.io/pypi/pyversions/cryobalance.svg)](https://pypi.org/project/cryobalance/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)

**Cryobalance** is a lightweight, high-throughput Python package for continuous orientation density estimation, orientation anisotropy diagnostics, and regularized inverse-density balancing in single-particle cryo-electron microscopy (cryo-EM).

By expanding discrete Euler projection angles into a real spherical harmonic basis on the unit sphere ($S^2$), Cryobalance balances reciprocal-space central slice sampling without heuristic particle deletion or data loss.

---

## Key Features

- **Continuous Orientation Profiling:** Decomposes Euler viewing distributions onto an orthonormal spherical harmonic basis up to degree $L_{\max}$.
- **Point-Group Symmetry Support:** Accounts for cyclic ($C_n$) and dihedral ($D_n$) point-group symmetry operations prior to density fitting.
- **Quantitative Diagnostics:** Computes the rotationally invariant **Anisotropy Index ($A$)**, normalized spherical Shannon entropy ($\tilde{H}$), and Kish's **Effective Sample Size ($\text{ESS}$)**.
- **Tikhonov-Regularized Balancing:** Assigns continuous inverse-density weights to mitigate preferred orientation artifacts while preventing numerical divergence in sparsely sampled angular domains.
- **Seamless RELION Integration:** Appends weights directly into the `rlnParticleWeight` header column of output `.star` files for direct hand-off to `relion_reconstruct` or 3D refinement jobs.
- **Fast Execution:** Vectorized implementation processes 60,000+ particles in under 10 seconds with $<120\text{ MB}$ memory consumption.

---

## Installation

### Via PyPI
```bash
pip install cryobalance
