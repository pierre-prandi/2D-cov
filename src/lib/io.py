from typing import List
import xarray as xr
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class dataset:
    def __init__(self, filename: str, coords: List[float]) -> None:
        self.filename = filename
        self.coords = coords
        self._dset = (
            xr.load_dataset(self.filename)
            .set_coords(["longitude", "latitude", "time"])
            .sea_level_anomaly
        )
        self.select()
        self._dset.values[self._dset.values>1e10] = np.nan
        self._shape = self.values.shape

    def select(self) -> None:
        """ perform coords based selection """
        [ll_lon, ll_lat, ur_lon, ur_lat] = self.coords
        selection = self._dset.where(
            (self._dset.longitude >= ll_lon)
            & (self._dset.longitude < ur_lon)
            & (self._dset.latitude >= ll_lat)
            & (self._dset.latitude < ur_lat),
            drop=True,
        )
        self._dset = selection

    def create_mask(self):
        number_of_nans = np.sum(
            np.isnan(self.values),
            axis=-1
        )
        self._valid_mask = number_of_nans==0

    @property
    def valid(self):
        return self._valid_mask
    
    @property
    def invalid(self):
        return ~self.valid
        
    @property 
    def shape(self):
        return self._shape
        
    def drop_incomplete(self) -> None:
        """ drop incomplete time series """
        # TODO better selection of incomplete time series
        complete = self._dset.where(self._dset < 1e10, drop=True)
        self._dset = complete

    @property
    def nelem(self):
        dims = np.array([self._dset.sizes[k] for k in self._dset.sizes.keys()])
        return np.prod(dims)

    @property
    def longitude(self):
        return self._dset.longitude.values

    @property
    def latitude(self):
        return self._dset.latitude.values

    @property
    def time(self):
        return self._dset.time.values

    @property
    def values(self):
        return self._dset.values
    
    def map(self, time_step=0):
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_axes(
            [0.05, 0.1, 0.9, 0.8],
            projection=ccrs.PlateCarree(central_longitude=self.longitude.mean()),
        )
        ax.add_feature(cfeature.LAND, color="#d9d9d9")
        ax.set_global()

        ax.pcolormesh(self.longitude, self.latitude, self.values[:,:,time_step].T, transform=ccrs.PlateCarree())

        return fig, ax
