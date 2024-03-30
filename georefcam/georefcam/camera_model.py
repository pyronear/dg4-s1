import cameratransform as ct
import numpy as np

from .data_types import Coord2DIntPoints, RayCoord3DFloatPoints
from.logger import logger
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute
from pyproj import CRS, Transformer
from rasterio.crs import CRS as rioCRS


class AbstractCameraModel(metaclass=ABCMeta):
    """An abstract class representing the common API shared by camera models.

    This abstract class codifies the public methods (and their respective
    signatures) that must be exposed by camera model classes. Such classes must
    inherit from this abstract class.
    """

    crs: str | CRS | rioCRS = abstract_attribute()

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


class SimpleCameraModel(AbstractCameraModel):
    """A model of geolocated, oriented camera.

    It can be used to project image pixel coordinates into real-world ray
    coordinates. It has no support for lens distortion corrections.

    Parameters
    ----------
    image_res : tuple[int, int]
        The (width, height) image resolution.
    view_x_deg : float
        The horizontal angle of view.
    view_y_deg : float
        The vertical angle of view.
    yaw_deg : float
        The yaw (i.e. azimuth) of the camera.
    pitch_deg : float
        The pitch of the camera.
    roll_deg : float
        The roll of the camera.
    lat : float
        The latitude of the camera location.
    lon : float
        The longitude of the camera location.
    alt_m : float
        The altitude of the camera location.
    crs : str | CRS | rioCRS
        The CRS of the system.

    Attributes
    ----------
    image_res : tuple[int, int]
        The (width, height) image resolution.
    view_x_deg : float
        The horizontal angle of view.
    view_y_deg : float
        The vertical angle of view.
    yaw_deg : float
        The yaw (i.e. azimuth) of the camera.
    pitch_deg : float
        The pitch of the camera.
    roll_deg : float
        The roll of the camera.
    lat : float
        The latitude of the camera location.
    lon : float
        The longitude of the camera location.
    alt_m : float
        The altitude of the camera location.
    crs : CRS | rioCRS
        The CRS of the system.
    tf_wgs84_to_local : Transformer
        The transform between the "WGS84" and the local CRS.
    tf_local_to_wgs84 : Transformer
        The transform between the local CRS and the "WGS84".
    proj_lat : float
        The latitude of the camera location projected into the local CRS.
    proj_lon : float
        The longitude of the camera location projected into the local CRS.

    """

    def __init__(
            self,
            image_res: tuple[int, int],
            view_x_deg: float,
            view_y_deg: float,
            yaw_deg: float,
            pitch_deg: float,
            roll_deg: float,
            lat: float,
            lon: float,
            alt_m: float,
            crs: str | CRS | rioCRS,
    ) -> None:

        self.image_res, self.view_x_deg, self.view_y_deg = image_res, view_x_deg, view_y_deg
        self.yaw_deg, self.pitch_deg, self.roll_deg = yaw_deg, pitch_deg, roll_deg
        self.lat, self.lon, self.alt_m = lat, lon, alt_m

        if isinstance(crs, str):
            self.crs = CRS(crs)
        else:
            self.crs = crs
        if not self.crs.is_projected:
            raise ValueError("The CRS must be a projected one.")

        self.tf_wgs84_to_local = Transformer.from_crs("wgs84", self.crs)
        self.tf_local_to_wgs84 = Transformer.from_crs(self.crs, "wgs84")
        self.proj_lat, self.proj_lon = self.tf_wgs84_to_local.transform(self.lat, self.lon)

    def project_pixel_points_to_world_rays(self, pixels: Coord2DIntPoints) -> RayCoord3DFloatPoints:
        """Projects a collection of pixel coordinates to real-world rays.

        Each projected ray is represented by two points:
        - the source point, i.e. the camera coordinates
        - the target point, i.e. the end of the ray (set to a 100km length)

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

        x, y = pixels[:, 0], pixels[:, 1]
        w, h = self.image_res
        point_yaw = self.yaw_deg + self.view_x_deg * (x - w / 2) / w
        point_pitch = self.pitch_deg + self.view_y_deg * (y - h / 2) / h

        ray_length_m = 100e3
        ray_endpoint_world = np.vstack((
            ray_length_m * np.sin(np.radians(point_yaw)),
            ray_length_m * np.cos(np.radians(point_yaw)),
            self.alt_m - ray_length_m * np.sin(np.radians(point_pitch))
        )).T

        campoint_world = np.array([self.proj_lat, self.proj_lon, self.alt_m])
        campoint_world_broadcasted = np.broadcast_to(campoint_world, ray_endpoint_world.shape)

        proj_rays = np.dstack((campoint_world_broadcasted, ray_endpoint_world))

        return proj_rays


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

