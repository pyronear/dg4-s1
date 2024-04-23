# Horizon package
## Installation
- check that build is installed: `pip install --upgrade build`
- from this folder, run `python -m build`
- then install the package with `pip install dist/horizon-0.0.0-py3-none-any.whl` or any `.whl` file generated  

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