# Smoke localization

## Lidar data
Go to [`./lidar`](./lidar)  
Install [`requirements.txt`](./lidar/requirements.txt) in your environment:  
`pip install requirements.txt`  

### Download the data (once)
- either from [IGN geoservice](https://geoservices.ign.fr/lidarhd) where you can select the tiles you want
- or directly from the notebook [`visualize_terrain.ipynb`](./lidar/visualize_terrain.ipynb), where a preselection of tiles is available in `liste_dalle_*.txt`

### Load data
Once the lidar data is loaded and downsampled, it is saved into `.pcd` files, where it can be quickly loaded next times.

## Explore
Open [`visualize_terrain.ipynb`](./lidar/visualize_terrain.ipynb) and follow the steps.
