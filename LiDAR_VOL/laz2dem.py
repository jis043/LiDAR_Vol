import laspy, numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin

def laz_to_dem_pure(laz_path, dem_path="dem.tif", resolution=0.25):
    las = laspy.read(laz_path)
    xs, ys, zs = las.x, las.y, las.z
    xi = np.arange(xs.min(), xs.max(), resolution)
    yi = np.arange(ys.min(), ys.max(), resolution)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xs, ys), zs, (X, Y), method="linear")

    transform = from_origin(xs.min(), ys.max(), resolution, resolution)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4978",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(np.nan_to_num(Z, nan=-9999).astype("float32"), 1)
    print(f"DEM saved: {dem_path}")
    return Z, transform


def calculate_volume_from_dem(dem_path: str, base_elevation: float = 0.0):
    """
    Calculate the total volume above a base elevation from a DEM GeoTIFF.

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF (.tif)
    base_elevation : float, optional
        Base elevation in meters. Defaults to 0.0 m.

    Returns
    -------
    float
        Volume in cubic meters
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)  # first band
        res_x, res_y = src.res
        cell_area = res_x * res_y

    # Mask out invalid or NoData cells
    nodata = src.nodata
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    # Height above base
    height_above = np.clip(dem - base_elevation, 0, None)

    # Compute total volume (m³)
    volume_m3 = np.nansum(height_above) * cell_area

    print(f"✅ Base elevation: {base_elevation:.2f} m")
    print(f"✅ Cell size: {res_x:.3f} × {res_y:.3f} m (area {cell_area:.3f} m²)")
    print(f"📦 Estimated volume above base = {volume_m3:.2f} m³")

    return volume_m3