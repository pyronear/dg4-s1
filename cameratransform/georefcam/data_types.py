import pandera as pa

from nptyping import Int, Float, NDArray, Shape
from pandera.typing import DataFrame, Index, Series

Coord1DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [z]"], Float]
Coord2DIntPoints = NDArray[Shape["* n_points, [x, y]"], Int]
Coord3DFloatGrid = NDArray[Shape["* n_rows, * n_cols, [x, y, z]"], Float]
Coord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z]"], Float]
RayCoord3DFloatPoints = NDArray[Shape["* n_points, [x, y, z], [origin, destination]"], Float]


class DfRaySchema(pa.SchemaModel):

    ori_x: Series[float]
    ori_y: Series[float]
    ori_z: Series[float]
    dest_x: Series[float]
    dest_y: Series[float]
    dest_z: Series[float]
    inter_x: Series[float]
    inter_y: Series[float]
    inter_z: Series[float]
    n_tri: Series[int] = pa.Field(nullable=True)
    dist_o: Series[float]
    n_ray: Index[int]


DfRayInstance = DataFrame[DfRaySchema]
