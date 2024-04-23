# Data
This folder gather all the data used in this project. Most of the files are not provided here due to their size, but can be easily downloaded or generated. In addition, some datasets were only used for exploration and are not required to use the packages.  

| Folder name  | Source | Description |
|---|---|---|
| pyronear_cam_data  | Pyronear  | Images and metadata from cameras used for fire detection at different viewpoints in France |
| alertwildfire | [ALERTWildfire](https://www.alertwildfire.org/) | Images and metadata from cameras used for fire detection at different viewpoints in USA |
| geopose3k| [GeoPose3K](https://cphoto.fit.vutbr.cz/geoPose3K/)| 3k mountain landscape photos, with camera coordinates and orientations, depth estimations, terrain illuminations, and more |
| terrain | [elevation](https://pypi.org/project/elevation/) or [IGN](https://geoservices.ign.fr/lidarhd)| Digital elevation model (DEM) coming from lidar measurements. Downloaded by zone of interest and transformed into point cloud files (.xyz)|
| horizon | generated | Horizons (skylines) extracted from DEM or from images as numpy arrays |
| models | generated | Trained pytorch models |
| open3d_parameters | generated | camera parameters for Open3d |
