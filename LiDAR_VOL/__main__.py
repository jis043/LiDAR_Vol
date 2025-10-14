import numpy as np
import os
from typing import Literal, Optional
import laz2dem as laz

def create_cube_pointcloud(
    output_path: str,
    cube_size: float = 10.0,
    spacing: float = 0.10,
    noise_sigma: float = 0.005,
    file_format: Literal["xyz", "ply", "las", "laz"] = "xyz",
    include_bottom: bool = False,
):
    """
    Generate a 3D cube point cloud (default: 10×10×10 m) representing its outer surfaces.
    By default the bottom face is excluded (simulating a drone scanning from above).

    Parameters
    ----------
    output_path : str
        Output file path (without extension or with desired extension).
    cube_size : float
        Cube edge length in meters.
    spacing : float
        Point spacing between grid samples.
    noise_sigma : float
        Standard deviation of Gaussian noise (meters).
    file_format : {"xyz", "ply", "las", "laz"}
        Output format. If las/laz chosen, requires `laspy` installed.
    include_bottom : bool
        If True, include the bottom (z=0) face.

    Returns
    -------
    dict : paths and point counts
    """
    rng = np.random.default_rng(42)

    def add_face(X, Y, Z, coords):
        Xn = X + rng.normal(0, noise_sigma, size=X.shape)
        Yn = Y + rng.normal(0, noise_sigma, size=Y.shape)
        Zn = Z + rng.normal(0, noise_sigma, size=Z.shape)
        coords.append((Xn.ravel(), Yn.ravel(), Zn.ravel()))

    grid = np.arange(0.0, cube_size + 1e-9, spacing)
    coords = []

    # Faces
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    Z = np.full_like(X, cube_size)        # Top (z=10)
    add_face(X, Y, Z, coords)

    if include_bottom:
        Z = np.zeros_like(X)              # Bottom (z=0)
        add_face(X, Y, Z, coords)

    # x=0 and x=10
    Y, Z = np.meshgrid(grid, grid, indexing="xy")
    X = np.zeros_like(Y)
    add_face(X, Y, Z, coords)
    X = np.full_like(Y, cube_size)
    add_face(X, Y, Z, coords)

    # y=0 and y=10
    X, Z = np.meshgrid(grid, grid, indexing="xy")
    Y = np.zeros_like(X)
    add_face(X, Y, Z, coords)
    Y = np.full_like(X, cube_size)
    add_face(X, Y, Z, coords)

    xs = np.concatenate([c[0] for c in coords])
    ys = np.concatenate([c[1] for c in coords])
    zs = np.concatenate([c[2] for c in coords])
    n_points = xs.size

    intensity = (200 + 100 * rng.random(n_points)).astype(np.uint16)
    classification = np.full(n_points, 6, dtype=np.uint8)

    # Infer extension if not provided
    ext = file_format.lower()
    if not output_path.endswith(f".{ext}"):
        output_path += f".{ext}"

    # Write
    if ext in ("xyz",):
        with open(output_path, "w") as f:
            for i in range(n_points):
                f.write(f"{xs[i]:.6f} {ys[i]:.6f} {zs[i]:.6f} {intensity[i]} {classification[i]}\n")

    elif ext == "ply":
        with open(output_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property ushort intensity\nproperty uchar classification\nend_header\n")
            for i in range(n_points):
                f.write(f"{xs[i]:.6f} {ys[i]:.6f} {zs[i]:.6f} {intensity[i]} {classification[i]}\n")

    elif ext in ("las", "laz"):
        try:
            import laspy
            header = laspy.LasHeader(point_format=1, version="1.2")
            header.offsets = np.array([0.0, 0.0, 0.0])
            header.scales = np.array([0.001, 0.001, 0.001])
            las = laspy.LasData(header)
            las.x, las.y, las.z = xs, ys, zs
            las.intensity = intensity
            las.classification = classification
            las.write(output_path)
        except ImportError:
            raise RuntimeError("laspy is not installed. Install it via `pip install laspy[lazrs]`.")

    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    return {"path": output_path, "points": n_points}


# Example usage:
# result = create_cube_pointcloud("cube_10m_5faces", file_format="xyz", include_bottom=False)
# print(result)
if __name__ == "__main__":
    create_cube_pointcloud(r"./cube_10m_5faces", file_format="laz", spacing=0.05)
    laz.laz_to_dem_pure(r"./cube_10m_5faces.laz")
    laz.calculate_volume_from_dem(r"./dem.tif", base_elevation=0.0)

