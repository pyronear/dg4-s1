import numpy as np
from pyproj import Transformer
import matplotlib.pyplot as plt

def to_lambert93(lat, lon, altitude=None):
    '''project a point from latitude longitude (ESPG:4326) to Lambert 93 coordinates (ESPG:2154)

    Args:
        lat (float): latitude
        lon (float): longitude
        altitude (float, optional): altitude

    Returns:
        np.array: [x,y,z] or [x, y] if no altitude
    '''
    projector = Transformer.from_crs("EPSG:4326", "EPSG:2154")
    point = np.array(projector.transform(lat,lon))
    if altitude:
        point = np.append(point, altitude)
    return point

def array_to_lambert93(array):
    projector = Transformer.from_crs("EPSG:4326", "EPSG:2154")
    def row_to_lambert_93(row):
        point = np.array(projector.transform(row[0],row[1]))
        if len(row) == 3:
            point = np.append(point, row[2])
        return point
    projected = np.apply_along_axis(row_to_lambert_93, 1, array)
    return projected


def to_lat_lon(x, y, z):
    '''project a point from Lambert 93 (ESPG:2154) to latitude longitude coordinates (ESPG:4326) 

    Args:
        x (float): x
        y (float): y
        z (float): altitude

    Returns:
        np.array: [lat,lon,z]
    '''
    projector = Transformer.from_crs("EPSG:2154", "EPSG:4326")
    point = np.array(projector.transform(x,y)+(z,))
    return point

def array_cartesian_to_spherical(points, view_point):
    '''Convert each cartesian points [x,y,z] of an array to spherical coordinates [r,theta,phi] (physics convention).

    Args:
        points (np.array (n,3)): list of points
        view_point (list or np.array (1,3)): viewpoint to consider as origin

    Return:
        np.array: array of spherical coordinates
    '''
    def cartesian_to_spherical(point):
        '''Convert a point [x,y,z] to spherical coordinates [r,theta,phi] (physics convention).
        The origin is the view_point.

        Args:
            point (list or np.array (1,3)): [x,y,z]

        Returns:
            list: [r,theta,phi]
        '''
        x0, y0, z0 = view_point
        x, y, z = point
        ydif = y-y0
        zdif = z-z0
        rho2 = np.square(x-x0)+np.square(ydif)
        r = np.sqrt(rho2+np.square(zdif))
        theta = np.arcsin(zdif/r)
        if x==x0 and y==y0:
            phi = 0
        elif x>=x0:
            phi = np.arcsin(ydif/np.sqrt(rho2))
        else: #x<x0
            phi = -np.arcsin(ydif/np.sqrt(rho2))+np.pi
        return [r, theta, 2*np.pi-phi]
    spherical = np.apply_along_axis(cartesian_to_spherical, 1, np.asarray(points))
    return spherical

def spherical_to_cartesian(point, replaceZ=False):
    '''Convert a point from spherical coordinates [r,theta,phi] to cartesian [x,y,z]

    Args:
        point (list or np.array (1,3)): the point in spherical coordinates (angles in radian)
        getZ (bool or int, optional): whether or not to compute the z dimension. Defaults to False.

    Returns:
        list: [x,y] or [x,y,z]
    '''
    r, theta, phi = point
    rho = r*np.cos(theta)
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    if replaceZ:
        return [x,y,replaceZ]
    else:
        z = r*np.sin(theta)
        return [x, y, z]

def get_deg_angles(spherical, decimals=0):
    '''Get angles in degree from spherical coordinates. Rounds only phi.

    Args:
        spherical (np.array (3,n)): array of spherical coordinates [[r,theta,phi],...]
        decimals (int, optional): decimals to keep when rounding phi. Defaults to 0.

    Returns:
        np.array (2,n): array of angles in degree [[theta, phi],...]
    '''
    # convert angles to degree (drop r)
    deg = np.rad2deg(spherical[:,1:])
    # round phi
    deg[:,1] = np.round(deg[:,1], decimals)
    return deg

def delete_max(array):
    '''Get the maximum value of an array and remove it

    Args:
        array (np.array): the array to compute max from

    Returns:
        (int, np.array): max value, array without max
    '''
    if len(array)==1:
        return array[0], array
    imax = np.argmax(array)
    m = array[imax]
    new_array = np.delete(array, imax)
    return m, new_array

def get_skyline(angles, threshold=1e-1, savepath=False):
    '''Extract skyline from spherical coordinates points

    Args:
        angles (np.array (3,n)): array of spherical coordinates angles in degree [[theta,phi],...]
        threshold (float, optional): outliers distants by more than this value are removed. Defaults to 1e-4.
        savepath (bool or str, optional): whether or not to save the skyline and at which path. defaults to False. 

    Returns:
        np.array (360,): skyline, i.e. max elevation angle for each azimuth angle
    '''
    # sort 
    angles = angles[angles[:, 1].argsort()]
    phi_values = np.unique(angles[:, 1], return_index=True)
    # group by phi 
    grouped_by_phi = np.split(angles[:,0], phi_values[1][1:])

    # get max theta for each phi
    skyline = np.empty(shape=(360,))
    for i, thetas in enumerate(grouped_by_phi):
        # remove outliers (max values too far away from second max)
        m, t = delete_max(thetas)
        while m-np.max(t) > threshold:
            thetas = t
            m, t = delete_max(thetas)
        skyline[i%360] = np.max(thetas)

    # save the skyline if requested
    if savepath:
        np.save(savepath+'.npy', skyline)
    return skyline

def skyline_to_cartesian(spherical, angles, skyline, view_point, max_z):
    '''Transform the skyline to cartesian coordinates, so it can be diplayed with the terrain

    Args:
        spherical (np.array (n,3)): spherical coordinates of the terrain points
        angles (np.array (n,2)): angles only in degree of the terrain points
        skyline (np.array (360,)): max elevation angle for each azimuth
        view_point (list or np.array (1,3)): viewpoint considered as origin for spherical coordinates
        max_z (float): maximum altitude in the point cloud

    Returns:
        np.array (n,3): array of coordinates [x,y,z] for each point of the skyline
    '''
    skyline_points = np.empty(shape=(360,3))
    # for each angle pair
    for phi, theta in enumerate(skyline):
        # get the index corresponding to that angle combination (spherical and angles have the same order)
        i = np.argwhere((angles[:,0]==theta) & ((angles[:,1]-90)%360==phi))[0][0]
        # convert the point to cartesian coordinates
        cart_point = spherical_to_cartesian(spherical[i], replaceZ=max_z)
        # translate cartesian conversion to viewpoint
        skyline_points[phi] = np.add(cart_point, view_point)
    return skyline_points

def unproject(u,v,w,param):
    '''Get real world coordinates (x,y,z) from image coordinates (u,v,depth)

    Args:
        u (int): image horizontal coordinate
        v (int): image vertical coordinates (pixel)
        w (float): depth value at (u,v)
        param (open3d.): camera parameters

    Returns:
        list (3): world coordinates [x,y,z]
    '''
    # project image coordinates to camera coordinates
    ray = np.linalg.inv(param.intrinsic.intrinsic_matrix).dot([u,v,1])
    # normalize and multiply by depth
    ray *= w/np.linalg.norm(ray)
    # project camera coordinates to world coordinates
    x,y,z,_ = np.linalg.inv(param.extrinsic).dot(np.append(ray,1))
    return [x,y,z]

def distance_points_refpoint(points, refpoint):
    '''Compute the distance of each points with a reference point

    Args:
        points (array (N,2)): list of points
        refpoint (array (2,)): reference point

    Returns:
        np.array (N,): list of distances
    '''
    sq = np.square(np.subtract(points, refpoint))
    dist = np.sqrt(np.sum(sq, axis=1))
    return dist

def distance_points_points(points1, points2):
    '''Compute each distances between two list of points.

    Args:
        points1 (np.array (N,2)): list of points
        points2 (np.array (M,2)): list of points

    Returns:
        np.array (N,M): upper triangular matrix with distances
    '''
    dist_matrix = np.full((len(points1), len(points2)), 0.0)
    for i, point in enumerate(points2):
        dist_matrix[i,i:] = distance_points_refpoint(points1[i:], point)
    return dist_matrix

def closest_point(terrain_points, view_point):
    '''return the point index in terrain_point that is the closest to the viewpoint

    Args:
        terrain_points (np.array (N,2)): list of points that makes the terrain
        view_point (np.array (2,)): point

    Returns:
        int: index of the closest point
    '''
    dist = distance_points_refpoint(terrain_points, view_point)
    closest = np.argmin(dist)
    return closest