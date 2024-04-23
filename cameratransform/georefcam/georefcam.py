import numpy as np
import pandas as pd
import signal

from .camera_model import AbstractCameraModel
from .dem import AbstractDEM
from .data_types import Coord3DFloatPoints, DfRayInstance, RayCoord3DFloatPoints
from .logger import logger
from nptyping import assert_isinstance, Int, NDArray, Shape
from pyproj import Transformer
from typing import Literal


class GeoRefCam:
    """Represents a georeferenced camera.

    This class combines a camera model and the DEM of the area within which the
    camera is located. It can be used to project pixel coordinate points into
    real-world coordinate rays, and find the intersection between these rays and
    the surrounding DEM.

    Parameters
    ----------
    camera_model: AbstractCameraModel
        The camera model used to represent the georeferenced camera.
    dem: AbstractDEM
        The DEM inside which the georeferenced camera lives.

    Attributes
    ----------
    camera_model: AbstractCameraModel
        The camera model used to represent the georeferenced camera.
    dem: AbstractDEM
        The DEM inside which the georeferenced camera lives.
    """

    def __init__(self, camera_model: AbstractCameraModel, dem: AbstractDEM):

        self.camera_model = camera_model
        self.dem = dem

    def project_points_from_cam_to_dem_crs(self, points: Coord3DFloatPoints) -> Coord3DFloatPoints:
        """Projects a collection of points from the camera's to the DEM's CRS.

        Parameters
        ----------
        points : Coord3DFloatPoints
            A collection of point coordinates in the camera's CRS.

        Returns
        -------
        proj_points : Coord3DFloatPoints
            The collection of point coordinates projected in the DEM's CRS.

        """

        proj_points = project_points_to_crs(points, self.camera_model.crs, self.dem.crs)
        return proj_points

    def project_rays_from_cam_to_dem_crs(self, rays: RayCoord3DFloatPoints) -> RayCoord3DFloatPoints:
        """Projects a collection of rays from the camera's to the DEM's CRS.

        Parameters
        ----------
        rays : RayCoord3DFloatPoints
            A collection of ray coordinates in the camera's CRS.

        Returns
        -------
        proj_rays : RayCoord3DFloatPoints
            The collection of ray coordinates projected in the DEM's CRS.
        """

        origins, destinations = rays[:, :, 0], rays[:, :, 1]
        proj_raypoints = self.project_points_from_cam_to_dem_crs(np.vstack((origins, destinations)))
        proj_rays = np.dstack((proj_raypoints[:len(origins)], proj_raypoints[len(origins):]))
        return proj_rays

    def cast_rays(
            self,
            rays: RayCoord3DFloatPoints,
            check_crs: bool = True,
            min_d_intersection_m: float = 0,
            return_df_ray: bool = False,
            cast_method: Literal["vec"] | Literal["seq"] | Literal["auto"] = "seq",
            max_cp_time_s: int | None = None,
    ) -> (Coord3DFloatPoints | None, NDArray[Shape["* n_rays"], Int] | None, DfRayInstance | None):
        """Computes the intersection points between a collection of rays and the class DEM.

        Each ray is cast over the DEM in order to find their intersection point.

        If `min_d_intersection_m` is not 0, the intersection points found too
        close to the camera are discarded.

        If multiple intersection points are found for a single ray, the one
        that is closest to the camera is retained.

        Parameters
        ----------
        rays : RayCoord3DFloatPoints
            The collection of rays to cast.
        check_crs : bool, default: True
            If True, the ray coordinates will be considered to be expressed in
            the camera model's CRS. If the latter is different from the DEM's
            CRS, the coordinates will be projected into the DEM's CRS.
        min_d_intersection_m : float, default: 0
            The minimal distance required, in m, between the camera and an
            intersection point for the latter to be considered valid.
        return_df_ray : bool, default: False
            If True, the dataframe containing the projected ray information will
            be returned.
        cast_method : Literal["vec"] | Literal["seq"] | Literal["auto"], default: "seq"
            The method to use for casting multiple rays:
            - "seq" : the intersections are calculated sequentially using
              `_cast_rays_seq()`. Typically faster for less than 1000 rays.
            - "vec" : the intersections are calculated in a vectorized way using
              `_cast_rays_vec()`. Typically faster for more than 1000 rays.
            - "auto" : the method is automatically selected based on the number
              rays to intersect.
        max_cp_time_s : int | None, default: None
            The maximal allowed amount of runtime, in s, to compute all the
            intersections. If this time is exceeded, the function stops and
            returns a tuple of None. If set to None, no runtime restriction is
            set.

        Returns
        -------
        filtered_inter_points : Coord3DFloatPoints | None
            The coordinates of the filtered intersection points.
        filtered_inter_triangles : NDArray[Shape["* n_rays"], Int] | None
            The number of the intersected mesh triangle for each intersection
            point.
        filtered_df_ray : DfRayInstance | None
            The dataframe containing the projected ray information.
        """

        assert_isinstance(rays, RayCoord3DFloatPoints)
        if check_crs and self.camera_model.crs != self.dem.crs:
            rays = self.project_rays_from_cam_to_dem_crs(rays)

        origins, destinations = rays[:, :, 0], rays[:, :, 1]

        df_ray = pd.DataFrame(
            data=np.hstack((origins, destinations)),
            columns=["ori_x", "ori_y", "ori_z", "dest_x", "dest_y", "dest_z"]
        )
        df_ray["n_ray"] = range(len(df_ray))

        if cast_method == "auto":
            if len(df_ray) > 1000:
                cast_method = "vec"
            else:
                cast_method = "seq"

        if max_cp_time_s is not None:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_cp_time_s)

        try:
            df_inter = getattr(self, f"_cast_rays_{cast_method}")(df_ray)
        except:
            logger.error(f"The ray casting operation reached the timeout threshold of {max_cp_time_s} s")
            return None, None, None

        filtered_inter_points, filtered_inter_triangles, filtered_df_ray = self._filter_intersections(df_inter, df_ray, min_d_intersection_m)

        if return_df_ray:
            return filtered_inter_points, filtered_inter_triangles, filtered_df_ray
        else:
            return filtered_inter_points, filtered_inter_triangles, None

    def _cast_rays_vec(self, df_ray: DfRayInstance) -> DfRayInstance:
        """Casts a collection of rays over the DEM in a vectorized way.

        This method uses PyVista's `pv.PolyData.multi_ray_trace()` method in
        order to accelerate the computation for large numbers of rays. Its
        overhead time makes it inefficient for small numbers of rays (typically
        less than 1000).

        Parameters
        ----------
        df_ray : DfRayInstance
            The dataframe storing the collection of rays to cast.

        Returns
        -------
        df_inter : DfRayInstance
            The dataframe storing the computed intersection data.
        """

        inter_points, parent_rays, inter_triangles = self.dem.mesh.multi_ray_trace(
            df_ray[["ori_x", "ori_y", "ori_z"]],
            df_ray[["dest_x", "dest_y", "dest_z"]],
            retry=False,
            first_point=False
        )
        df_inter = pd.DataFrame(
            data=np.hstack((inter_points, parent_rays[:, np.newaxis], inter_triangles[:, np.newaxis])),
            columns=["inter_x", "inter_y", "inter_z", "n_ray", "n_tri"]
        )
        df_inter = df_inter.astype({"n_ray": int, "n_tri": "Int64"})  # the type assigned to 'n_tri' works as a nullable integer type

        return df_inter

    def _cast_rays_seq(self, df_ray: DfRayInstance) -> DfRayInstance:
        """Casts a collection of rays over the DEM in a sequential manner.

        This method uses PyVista's `pv.PolyData.ray_trace()` method in order to
        quickly compute intersections for small numbers of rays.

        Parameters
        ----------
        df_ray : DfRayInstance
            The dataframe storing the collection of rays to cast.

        Returns
        -------
        df_inter : DfRayInstance
            The dataframe storing the computed intersection data.
        """

        inter_points, parent_rays, inter_triangles = [], [], []
        for ray in df_ray.itertuples():
            parent_ray = ray.n_ray
            origin = (ray.ori_x, ray.ori_y, ray.ori_z)
            destination = (ray.dest_x, ray.dest_y, ray.dest_z)
            inter_point, inter_triangle = self.dem.mesh.ray_trace(origin, destination)
            for point, triangle in zip(inter_point, inter_triangle):
                inter_points.append(point)
                parent_rays.append([parent_ray])
                inter_triangles.append([triangle])

        inter_points, parent_rays, inter_triangles = np.array(inter_points), np.array(parent_rays), np.array(inter_triangles)
        df_inter = pd.DataFrame(
            data=np.hstack((inter_points, parent_rays, inter_triangles)),
            columns=["inter_x", "inter_y", "inter_z", "n_ray", "n_tri"]
        )
        df_inter = df_inter.astype({"n_ray": int, "n_tri": "Int64"})  # the type assigned to 'n_tri' works as a nullable integer type

        return df_inter

    @staticmethod
    def _filter_intersections(
            df_inter: DfRayInstance, df_ray: DfRayInstance, min_d_intersection_m: int
    ) -> (Coord3DFloatPoints, NDArray[Shape["* n_rays"], Int], DfRayInstance):
        """Filters the intersection points found by the casting method.

        Two filtering criteria are applied in the following order:
        - If the distance between an intersection point and the camera is
          smaller than `min_d_intersection_m`, it is discarded
        - If several candidate intersection points remain for a given ray, the
          one that is closest to the camera is selected

        Parameters
        ----------
        df_inter : DfRayInstance
            The dataframe storing the (unfiltered) computed intersection data.
        df_ray : DfRayInstance
            The dataframe storing the collection of rays to cast.
        min_d_intersection_m : int
            The minimal distance required, in m, between the camera and an
            intersection point for the latter to be considered valid.

        Returns
        -------
        filtered_inter_points : Coord3DFloatPoints
            The coordinates of the filtered intersection points.
        filtered_inter_triangles : NDArray[Shape["* n_rays"], Int]
            The number of the intersected mesh triangle for each intersection
            point.
        filtered_df_ray : DfRayInstance
            The dataframe containing the projected ray information.
        """

        df_inter[["ori_x", "ori_y", "ori_z"]] = df_ray.loc[df_inter.n_ray, ["ori_x", "ori_y", "ori_z"]].values
        df_inter["dist_o"] = np.linalg.norm(df_inter[["inter_x", "inter_y", "inter_z"]].values - df_inter[["ori_x", "ori_y", "ori_z"]].values, axis=1)
        df_inter = df_inter[df_inter["dist_o"] > min_d_intersection_m]

        gp_ray = df_inter.groupby("n_ray")
        df_inter = df_inter.loc[gp_ray.dist_o.idxmin()]
        df_ray = df_ray.merge(df_inter[["inter_x", "inter_y", "inter_z", "n_ray", "n_tri", "dist_o"]], on="n_ray", how="left")
        df_ray = df_ray.sort_values("n_ray").set_index("n_ray")

        filtered_inter_points = df_ray[["inter_x", "inter_y", "inter_z"]].values
        filtered_inter_triangles = df_ray.n_tri.values

        return filtered_inter_points, filtered_inter_triangles, df_ray


def project_points_to_crs(points: Coord3DFloatPoints, from_crs: str, to_crs: str) -> Coord3DFloatPoints:
    """Projects a collection of points from one CRS to another.

    Parameters
    ----------
    points : Coord3DFloatPoints
        The points to project.
    from_crs : str
        The source CRS.
    to_crs : str
        The target CRS.

    Returns
    -------
    proj_points : Coord3DFloatPoints
        The projected points.
    """

    transformer = Transformer.from_crs(from_crs, to_crs)
    if points.ndim == 2:
        xx, yy, zz = [points[:, i] for i in range(3)]
    else:
        xx, yy, zz = points
    pr_xx, pr_yy, pr_zz = transformer.transform(xx, yy, zz)
    proj_points = np.vstack((pr_xx, pr_yy, pr_zz)).T

    return proj_points


def timeout_handler(signum, frame):
    """Raises a TimeoutError upon calling."""
    raise TimeoutError
