# Horizon package
## Installation
- from this folder, run `pip install .`  

No need to install `requirements.txt` as this is included in the package dependancies.

## Modules description
### load
Various functions to download elevation data.
### camera
Manage open3d camera to interactively visualize terrain or generate images.
### project
Utilities functions to project and transform coordinates: EPSG, cartesian to spherical, extract skyline, compute distances.
### signal
Process signals (skylines): normalize, smooth, compute correlation score, estimate azimuth, plot skylines.
### image
Process images, extract skylines from images
### adjustement
Train a model to adjust sub signals so they better fit their reference signal. Not necessary now that a slope_square_diff is used instead of classical square_diff for correlation.