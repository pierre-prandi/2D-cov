import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def show_time_series(x: np.ndarray, ar: np.ndarray, figname: str) -> None:
    fig, ax = plt.subplots()
    (nx, ny, nt) = np.shape(ar)

    assert nt == len(x)
    ax.grid()
    for ix in range(nx):
        for iy in range(ny):
            ax.scatter(
                x / 365.25 + 1950.0,
                ar[ix, iy, :],
                color="lightgray",
                marker="o",
                edgecolor="black",
            )

    fig.savefig(figname)


def show_time_series_with_pred(
    x: np.ndarray, ar: np.ndarray, x_pred, y_pred, figname: str
) -> None:
    fig, ax = plt.subplots()
    (nx, ny, nt) = np.shape(ar)

    assert nt == len(x)
    ax.grid()
    for ix in range(nx):
        for iy in range(ny):
            ax.scatter(
                x / 365.25 + 1950.0,
                ar[ix, iy, :],
                color="lightgray",
                marker="o",
                edgecolor="black",
            )
    ax.plot(
        x_pred / 365.25 + 1950.0,
        y_pred,
        color="red",
        linewidth=3,
        marker="o",
        mec="white",
        mew=1,
    )

    fig.savefig(figname)


def show_area(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray, figname: str
) -> None:
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_axes(
        [0.05, 0.1, 0.9, 0.8],
        projection=ccrs.PlateCarree(central_longitude=lon.mean()),
    )
    ax.add_feature(cfeature.LAND, color="#d9d9d9")
    lon_offset = 40
    lat_offset = 30
    ax.set_extent(
        [
            lon.min() - lon_offset,
            lon.max() + lon_offset,
            lat.min() - lat_offset,
            lat.max() + lat_offset,
        ],
        crs=ccrs.PlateCarree(),
    )

    ax.pcolormesh(lon, lat, values.T, transform=ccrs.PlateCarree())

    fig.savefig(figname)


def show_covariance(omega, figname):
    """ show the error covariance """
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_axes(
        [0.05, 0.1, 0.9, 0.8],
    )
    ax.imshow(omega)
    fig.savefig(figname)