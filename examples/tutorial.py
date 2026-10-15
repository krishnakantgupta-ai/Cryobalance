"""
Cryobalance Quickstart Tutorial
===============================
This example shows how to inspect orientation bias and generate
a balanced STAR file for RELION back-projection.
"""

import numpy as np
import starfile
from cryobalance.core import compute_metrics_and_weights

def main():
    input_file = "examples/data/sample_particles.star"
    output_file = "examples/balanced_output.star"

    print(f"1. Loading orientation metadata from: {input_file}")
    data = starfile.read(input_file)
    df = data["particles"] if isinstance(data, dict) else data

    # Extract Euler angles in radians
    phi = np.deg2rad(df["rlnAngleRot"].to_numpy()) % (2 * np.pi)
    theta = np.clip(np.deg2rad(df["rlnAngleTilt"].to_numpy()), 0, np.pi)

    print("2. Running spherical harmonic decomposition (L_max = 12, Symmetry = C3)...")
    weights, metrics = compute_metrics_and_weights(
        phi=phi,
        theta=theta,
        l_max=12,
        sym="C3",
        alpha=0.02
    )

    print("\n--- Diagnostic Results ---")
    print(f"Anisotropy Index (A):         {metrics.anisotropy_index:.4f}")
    print(f"Normalized Sampling Entropy:  {metrics.sampling_entropy:.4f}")
    print(f"Effective Sample Size (ESS):  {metrics.effective_particle_count:,.0f} / {len(df):,}")

    print("\n3. Appending weights and writing output STAR file...")
    df["rlnParticleWeight"] = np.round(weights, 5)
    
    if isinstance(data, dict):
        data["particles"] = df
        starfile.write(data, output_file, overwrite=True)
    else:
        starfile.write(df, output_file, overwrite=True)

    print(f"[✓] Tutorial completed. Balanced file saved to: {output_file}")
    print("Ready for RELION back-projection: relion_reconstruct --i examples/balanced_output.star ...")

if __name__ == "__main__":
    main()