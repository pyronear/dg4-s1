import elevation as eio
import numpy as np
import pandas as pd
import pyvista as pv
import rioxarray

from .data_types import Coord1DFloatGrid, Coord3DFloatGrid
from .logger import logger
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute
from pandera.typing import Series
from pathlib import Path
from pyproj import CRS
from rasterio.crs import CRS as rioCRS


class AbstractDEM(metaclass=ABCMeta):
    """An abstract class representing the common API shared by DEM classes.

    This abstract class codifies the public methods (and their respective
    signatures) that must be exposed by DEM classes. Such classes must inherit
    from this abstract class.

    Attributes
    ----------
    crs : str | CRS | rioCRS
        The CRS of the system.
    pcd : Coord3DFloatGrid | None
        The point cloud representing the DEM of the file.
    mesh : pv.PolyData | None
        The triangular mesh associated with the point cloud.
    """

    crs: str | CRS | rioCRS | None = abstract_attribute()
    pcd: Coord3DFloatGrid = abstract_attribute()
    mesh: pv.PolyData | None = abstract_attribute()

    @abstractmethod
    def build_pcd(self, sample_step: int) -> None:
        """Builds the 3D-point cloud corresponding to the DEM.

        Parameters
        ----------
        sample_step : int
            The sampling step, used to subsample the mapping before creating the
            grid. For example, sample_step = 10 will sample 1 in 10 points.
        """
        pass

    @abstractmethod
    def build_mesh(self) -> None:
        """Builds the triangular mesh of a cloud point using a Delaunay2D."""
        pass


class ASCIIGridDEM(AbstractDEM):
    """Represents the DEM extracted from an ASCII grid file (.asc).

    The class instanciation requires to provide an ASCII grid filepath, and
    eventually the corresponding CRS. One can then use the class methods to
    build the corresponding point cloud and triangular mesh.

    Parameters
    ----------
    filepath : Path | str
        The filepath of the target DEM to represent.
    crs : str | None, default: None
        The CRS used to represent the coordinates of the DEM points.

    Attributes
    ----------
    filepath : Path | str
        The filepath of the target DEM to represent.
    crs : str | None
        The CRS used to represent the coordinates of the DEM points.
    header : Series[float]
        The header read in the file.
    alts : Coord1DFloatGrid
        The elevation table read in the file.
    pcd : Coord3DFloatGrid | None
        The point cloud representing the DEM of the file.
    mesh : pv.PolyData | None
        The triangular mesh associated with the point cloud.
    """

    def __init__(self, filepath: Path | str, crs: str | CRS | None = None) -> None:

        self.filepath, self.crs = filepath, crs
        self.header, self.alts = self._build_header_and_alts_table()
        self.pcd, self.mesh = None, None

    def build_pcd(self, sample_step: int) -> None:
        """Builds the 3D-point cloud corresponding to an .asc file.

        Parameters
        ----------
        sample_step : int
            The sampling step, used to subsample the mapping before creating the
            grid. For example, sample_step = 10 will sample 1 in 10 points.
        """

        x_grid = (self.header.xllcorner + np.arange(0, self.header.ncols) * self.header.cellsize)[::sample_step]
        y_grid = np.flip(self.header.yllcorner + np.arange(0, self.header.nrows) * self.header.cellsize)[::sample_step]  # flip nécessaire pour avoir y croissants ascendants
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        self.pcd = np.dstack((xx_grid, yy_grid, self.alts[::sample_step, ::sample_step]))

    def build_mesh(self) -> None:
        """Builds the triangular mesh of a cloud point using a Delaunay2D."""

        if self.pcd is None:
            raise ValueError("Building a mesh requires first to build a point cloud with the 'build_pcd' method.")
        pv_pcd = pv.StructuredGrid(*[self.pcd[:, :, i] for i in range(3)])
        self.mesh = pv_pcd.cast_to_poly_points().delaunay_2d()

    def _build_header_and_alts_table(self) -> (Series[float], Coord1DFloatGrid):
        """Reads the header and altimetry table from an IGN tile file (.asc).

        Returns
        -------
        header : Series[float]
            The file's header containing the following values: ncols, nrows,
            xllcorner, yllcorner, cellsize, NODATA_value.
        alts : Coord1DFloatGrid
            The altimetry table for the tile.
        """

        n_lines = 0
        header_lines = []
        with open(self.filepath, "r") as f:
            while n_lines < 6:
                header_lines.append(f.readline().split())
                n_lines += 1
        header = pd.DataFrame(header_lines).set_index(0).squeeze().astype(float)
        alts = pd.read_csv(self.filepath, sep=" ", skiprows=6, header=None).drop(0, axis=1).replace(header.NODATA_value, np.nan).values
        return header, alts


class EioDEM(AbstractDEM):
    """Represents the DEM extracted using the 'elevation' package.

    The class only requires to provide the bounds of the area to cover. One can
    then use the class methods to build the corresponding point cloud and
    triangular mesh.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        The [xmin, ymin, xmax, ymax] bounds delimiting the region to be covered
        by the DEM.
    product : str, default: "SRTM3"
        The data source to be used for generating the DEM.
    filepath : Path | str | None, default: None
        The filepath where the loaded data will be stored as a GeoTIFF.

    Attributes
    ----------
    bounds : tuple[float, float, float, float]
        The [xmin, ymin, xmax, ymax] bounds delimiting the region to be covered
        by the DEM.
    crs : rioCRS
        The CRS used to represent the coordinates of the DEM points.
    product : str
        The data source to be used for generating the DEM.
    filepath : Path
        The filepath where the loaded data will be stored as a GeoTIFF.
    pcd : Coord3DFloatGrid | None
        The point cloud representing the DEM of the file.
    mesh : pv.PolyData | None
        The triangular mesh associated with the point cloud.
    """

    def __init__(
            self,
            bounds: tuple[float, float, float, float],
            product: str = "SRTM3",
            filepath: Path | str | None = None,
    ) -> None:

        self.bounds, self.product = bounds, product

        if filepath is None:
            self.filepath = Path.cwd() / f"{product}_{'_'.join([f'{bound:.5f}' for bound in bounds])}.tiff"
        elif isinstance(filepath, Path):
            self.filepath = filepath
        else:
            self.filepath = Path(filepath)

        self.pcd, self.mesh = None, None

        if not self.filepath.is_file():
            eio.clip(bounds=bounds, output=self.filepath, product=self.product)
            logger.info(f"DEM saved at '{self.filepath}'")

        lat_center, lon_center = np.mean([self.bounds[1], self.bounds[3]]), np.mean([self.bounds[0], self.bounds[2]])
        self.crs = rioCRS.from_dict(proj="tmerc", ellps="WGS84", lat_0=lat_center, lon_0=lon_center)

    def build_pcd(self, sample_step: int) -> None:
        """Builds the 3D-point cloud.

        Parameters
        ----------
        sample_step : int
            The sampling step, used to subsample the grid. For example,
            sample_step = 10 will sample 1 in 10 points.
        """

        dem = rioxarray.open_rasterio(self.filepath)
        dem = dem.rio.reproject(self.crs)
        dem = dem.rio.interpolate_na()

        dem_bounds = dem.rio.bounds()
        x_grid = np.linspace(dem_bounds[0], dem_bounds[2], num=dem.rio.shape[0])[::sample_step]
        y_grid = np.linspace(dem_bounds[1], dem_bounds[3], num=dem.rio.shape[1])[::sample_step]
        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        alts = np.flip(dem.to_numpy().squeeze(0).transpose(1, 0), 1)[::sample_step, ::sample_step]
        self.pcd = np.dstack((xx_grid, yy_grid, alts))

    def build_mesh(self) -> None:
        """Builds the triangular mesh of a cloud point using a Delaunay2D."""

        if self.pcd is None:
            raise ValueError("Building a mesh requires first to build a point cloud with the 'build_pcd' method.")

        grid = pv.ImageData(
            dimensions=np.array([self.pcd.shape[0], self.pcd.shape[1], 1])+1,
            spacing=(self.pcd[1, 0, 1] - self.pcd[0, 0, 1], self.pcd[0, 1, 0] - self.pcd[0, 0, 0], 0),
            origin=(self.pcd[0, 0, 0], self.pcd[0, 0, 1], 0)
        )
        grid.cell_data["alt_m"] = self.pcd[:, :, 2].flatten(order='F')
        self.mesh = grid.ctp().warp_by_scalar("alt_m").extract_surface()
