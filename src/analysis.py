from lib.io import dataset
from lib.model import Model
from lib.covariance import NoiseCovariance2D
from lib.utils import (
    show_area,
    show_time_series,
    show_time_series_with_pred,
    show_covariance,
)

fname = "/Users/pierre/Dev/2d_cov_4_sea_level/data/77853.nc"
coords = [0, -50, 360, 50]
# open the dataset
dset = dataset(fname, coords)
print(dset.nelem)

# selection the area of interest
dset.select()
print(dset.nelem)

# drop incomplete time series
dset.drop_incomplete()

# plot the original time series
show_time_series(dset.time, dset.values, "time_series.png")

# show the area of interest
show_area(dset.longitude, dset.latitude, dset.values[:, :, 0], "zoi.png")

# build the model
model = Model(1)
model.design(dset.longitude, dset.latitude, dset.time, dset.values)
model.fit()
params = model.parameters()
print(params)

# plot the original time series and the model prediction
show_time_series_with_pred(
    dset.time,
    dset.values,
    dset.time,
    model.predict(dset.time),
    "time_series_with_pred.png",
)

# build the covariance
noise = NoiseCovariance2D(10, 10, 1000)
omega = noise.cov(dset.longitude, dset.latitude, dset.time)
params_var = model.parameters_variance(omega)
print(params_var)

show_covariance(omega.T, "covariance.png")
