import click
import numpy as np
import starfile
from .core import compute_metrics_and_weights

def load_angles(star_path):
    data = starfile.read(star_path)
    df = data["particles"] if isinstance(data, dict) and "particles" in data else (list(data.values())[-1] if isinstance(data, dict) else data)
    phi = np.deg2rad(df["rlnAngleRot"].to_numpy()) % (2.0 * np.pi)
    theta = np.clip(np.deg2rad(df["rlnAngleTilt"].to_numpy()), 0.0, np.pi)
    return data, df, phi, theta

@click.command()
@click.argument("input_star", type=click.Path(exists=True))
@click.argument("output_star", type=click.Path())
@click.option("-l", "--lmax", default=12, type=int)
@click.option("-s", "--symmetry", default="C1", type=str)
@click.option("-a", "--alpha", default=0.02, type=float)
def main(input_star, output_star, lmax, symmetry, alpha):
    data, df, phi, theta = load_angles(input_star)
    weights, metrics = compute_metrics_and_weights(phi, theta, l_max=lmax, sym=symmetry, alpha=alpha)
    click.echo(f"Anisotropy Index: {metrics.anisotropy_index:.4f}")
    click.echo(f"Effective Sample Size: {metrics.effective_particle_count:,.0f} / {len(df):,}")
    df_out = df.copy()
    df_out["rlnParticleWeight"] = np.round(weights, 5)
    if isinstance(data, dict):
        key = "particles" if "particles" in data else list(data.keys())[-1]
        data[key] = df_out
        starfile.write(data, output_star, overwrite=True)
    else:
        starfile.write(df_out, output_star, overwrite=True)
    click.echo(f"Saved: {output_star}")

if __name__ == "__main__":
    main()
