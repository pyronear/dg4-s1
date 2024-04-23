import numpy as np
import pandas as pd
import pyvista as pv

from .data_types import Coord1DFloatGrid
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute
from pandera.typing import Series
from pathlib import Path
from pyproj import CRS


class AbstractDEM(metaclass=ABCMeta):
    """An abstract class representing the common API shared by DEM classes.

    This abstract class codifies the public methods (and their respective
    signatures) that must be exposed by DEM classes. Such classes must inherit
    from this abstract class.
    """

    crs: str = abstract_attribute()
    mesh: pv.PolyData | None = abstract_attribute()
    pcd = abstract_attribute()

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
