import cameratransform as ct
import numpy as np

from .data_types import Coord2DIntPoints, RayCoord3DFloatPoints
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute


class AbstractCameraModel(metaclass=ABCMeta):
    """An abstract class representing the common API shared by camera models.

    This abstract class codifies the public methods (and their respective
    signatures) that must be exposed by camera model classes. Such classes must
    inherit from this abstract class.
    """

    crs: str = abstract_attribute()

    @abstractmethod
    def project_pixel_points_to_world_rays(self, pixels: Coord2DIntPoints) -> RayCoord3DFloatPoints:
        """Projects a collection of pixel coordinates to real-world rays.

        Each projected ray is represented by two points.

        Parameters
        ----------
        pixels : Coord2DIntPoints
            The collection of pixel coordinates to project.

        Returns
        -------
        proj_rays : RayCoord3DFloatPoints
            The collection of projected rays matching the pixel coordinates.
        """
        pass


class CTCameraModel(AbstractCameraModel):
    """A model of geolocated, oriented camera leveraging the `cameratransform` package.

    It can be used to project image pixel coordinates into real-world ray
    coordinates.

    Parameters
    ----------
    device_params : dict[str, float | tuple[int, int]]
        The device parameters, in a format accepted by
        `ct.RectiLinearProjection()`.
    ypr_orientation_params : dict[str, float]
        The yaw-pitch-roll parameters, in a format accepted by
        `ct.SpatialOrientationYawPitchRoll()`.
    location_params : dict[str, float]
        The latitude-longitude-elevation parameters, in a format accepted by
        `ct.Camera.setGPSpos()`.

    Attributes
    ----------
    device_params : dict[str, float | tuple[int, int]]
        The device parameters, in a format accepted by
        `ct.RectiLinearProjection()`.
    ypr_orientation_params : dict[str, float]
        The yaw-pitch-roll parameters, in a format accepted by
        `ct.SpatialOrientationYawPitchRoll()`.
    location_params : dict[str, float]
        The latitude-longitude-elevation parameters, in a format accepted by
        `ct.Camera.setGPSpos()`.
    crs : str
        The CRS of the system in string format, set to "WGS84".
    """

    crs = "WGS84"

    def __init__(
            self,
            device_params: dict[str, float | tuple[int, int]],
            ypr_orientation_params: dict[str, float],
            location_params: dict[str, float]
    ) -> None:

        self.device_params, self.ypr_orientation_params, self.location_params = device_params, ypr_orientation_params, location_params
        self.cam_model = ct.Camera(
            ct.RectilinearProjection(**device_params),
            ct.SpatialOrientationYawPitchRoll(**ypr_orientation_params)
        )
        self.cam_model.setGPSpos(location_params["lat"], location_params["lon"], location_params["alt"])

    def project_pixel_points_to_world_rays(self, pixels: Coord2DIntPoints) -> RayCoord3DFloatPoints:
        """Projects a collection of pixel coordinates to real-world rays.

        Each projected ray is represented by two points:
        - the source point, i.e. the camera coordinates
        - the target point, i.e. the intersection between the ray and the
          sea-level plane.

        If the projection of a point failed, all the coordinates of the
        corresponding ray will be set to NaN.

        Parameters
        ----------
        pixels : Coord2DIntPoints
            The collection of pixel coordinates to project.

        Returns
        -------
        proj_rays : RayCoord3DFloatPoints
            The collection of projected rays matching the pixel coordinates.
        """

        proj_pixels = self.cam_model.gpsFromImage(pixels)
        campoint_world = np.array([self.location_params["lat"], self.location_params["lon"], self.location_params["alt"]])
        campoint_world_broadcasted = np.broadcast_to(campoint_world, proj_pixels.shape)
        proj_rays = np.dstack((campoint_world_broadcasted, proj_pixels))

        return proj_rays

