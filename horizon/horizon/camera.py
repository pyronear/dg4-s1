import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import cv2

WIDTH = 1280
HEIGHT = 720

def get_focal_length_px(f, diag, width=WIDTH, height=HEIGHT):
    '''Compute focal lengths in pixels from camera sensor specifications

    Args:
        f (float): focal length in mm
        diag (float): sensor diagonal, for example 1/2.49
        width (int, optional): image width in px
        height (int, optional): image height in px

    Returns:
        (float, float): focal lengths for x and y, in pixels
    '''
    diag *= 16 # in mm
    a = np.arctan(height/width)
    sensorWidth = diag * np.cos(a)
    sensorHeight = diag * np.sin(a)
    fx = f*width/sensorWidth
    fy = f*height/sensorHeight
    return fx, fy

def visualize(objects, parameters, width=WIDTH, height=HEIGHT):
    '''Create a window to visualize a point cloud given camera parameters

    Args:
        objects (list of open3d.geometry...): list of point cloud, or mesh to visualize
        parameters (open3d.camera.PinholeCameraParameters): camera parameters
        width (int, optional): window width. Defaults to WIDTH.
        height (int, optional): window height. Defaults to HEIGHT.
    '''
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for o in objects:
        vis.add_geometry(o)
    vis.get_view_control().convert_from_pinhole_camera_parameters(parameters, True)
    vis.run()
    vis.destroy_window()

def callbacks(parameters, view_point):
    '''Set camera actions and return callbacks dict

    Args:
        parameters (open3d.camera.PinholeCameraParameters): virtual camera parameters
        view_point (np.array): view point coordinates [x,y,z]

    Returns:
        dict: dict of keys to callback functions
    '''
    def rotate_view_right(vis):
        '''Camera rotation around local up axis, clockwise
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_rotate(4.0, 0.0)
        return False

    def rotate_view_left(vis):
        '''Camera rotation around local up axis, counterclockwise
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_rotate(-4.0, 0.0)
        return False

    def translate_view(vis):
        '''Camera translation towards local front axis
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.camera_local_translate(10,0,0)
        return False

    def correct_up(vis):
        '''Set the local up axis to be colinear to global z axis
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.set_up(np.array([0,0,1]))
        return False

    def set_view(vis):
        '''Set the camera position to view point
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(parameters, True)
        print('field of view:', ctr.get_field_of_view())
        return False
    
    def set_fov(vis):
        '''Increase the camera field of view
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=1)
        return False
    
    def set_fov_back(vis):
        '''Decrease the camera field of view
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.change_field_of_view(step=-1)
        return False

    def look_at_view_point(vis):
        '''Set the local front axis to look at viewpoint
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        ctr = vis.get_view_control()
        ctr.set_lookat(view_point)
        return False
    
    def save_depth(vis):
        '''Save the current depth map and camera parameters
        Args:
            vis (o3d.visualization.Visualizer): visualizer
        Returns:
            bool: False
        '''
        datadir = '../../data/open3d_parameters/'
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(datadir+'test_parameters.json', param)
        depths = np.asarray(vis.capture_depth_float_buffer())
        np.save(datadir+'test_depth.npy', depths)

        return False

    key_to_callback = {}
    key_to_callback[gui.KeyName.RIGHT] = rotate_view_right
    key_to_callback[gui.KeyName.LEFT] = rotate_view_left
    key_to_callback[ord("T")] = translate_view
    key_to_callback[ord("U")] = correct_up
    key_to_callback[ord("V")] = set_view
    key_to_callback[ord("L")] = look_at_view_point
    key_to_callback[ord("D")] = save_depth
    key_to_callback[ord("F")] = set_fov
    key_to_callback[ord("G")] = set_fov_back

    return key_to_callback

def save_skyline_with_terrain(pc, skyline_points, image_path):
    '''Render an orthographic projection (no perspective) of the terrain from above, and add the skyline. Save it as image

    Args:
        pc (open3d.geometry.PointCloud): the terrain point cloud
        skyline_points (np.array (360,3)): skyline cartesian coordinates
        image_path (str): path and name where to save the image 
    '''
    # connect each point with the next one (except last with first)
    line_indices = [[i, i + 1] for i in range(0, 360)]
    line_indices[-1][1] = 0
    # create open3d lines from skyline points
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(skyline_points)
    lines.lines = o3d.utility.Vector2iVector(line_indices)
    #colors = [[1.0, 0.2, 0.7] for _ in range(0, 360)]
    #lines.colors = o3d.utility.Vector3dVector(colors)

    # Generate a render
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=500, height=500, visible=False)
    vis.add_geometry(pc)
    vis.add_geometry(lines)
    # fov < 5 is rendered as orthographic projection
    vis.get_view_control().change_field_of_view(-60)
    vis.get_view_control().set_zoom(0.55)
    vis.poll_events()
    vis.update_renderer()
    # save the image
    vis.capture_screen_image(image_path)
    print('Image saved at',image_path)
    vis.destroy_window()

def get_extrinsic(azimuth, view_point):
    '''Compute the camera extrinsic matrix (convert from camera coordinates to real world)

    Args:
        azimuth (int): angle in degrees
        view_point (np.array (3,)): x,y,z viewpoint coordinates 

    Returns:
        np.array (4,4): extrinsic matrix (as defined by open3d)
    '''
    '''
    First get the transformation matrix from real world to camera 
    (usually called extrinsic matrix, but for open3d it's its inverse)
    | R(3,3) t(1,3) |
    | 0(3,1)    1   |
    with R the rotation matrix, 
        can be expressed with camera axis (front, right, up):
        | right.T  |
        | -up.T    |
        | -front.T |
        for example, the default R is:
        | 1  0  0 |
        | 0  0 -1 |
        | 0 -1  0 | 
    and t, the translation vector [tx, ty, tz].T, in our case the viewpoint coordinates
    '''
    # get R directly from a rotation vector
    # the only rotation is around the up axis, the value is the azimuth (in radians)
    rvec = np.array([0, 0, 1]) * azimuth * np.pi/180
    R, _ = cv2.Rodrigues(rvec)
    # some transformations to adapt to the open3d conventions
    R[1,:] *=-1
    R[2,:] *=-1
    R[:,[1,2]] = R[:,[2,1]] # invert Y and Z columns
    # append R columns with 0 and t with 1, and stack them to get the 4*4 extrinsic matrix
    ext = np.column_stack((np.row_stack((R, [0,0,0])),np.append(view_point,[1])))
    # invert to get open3d extrinsic matrix
    ext = np.linalg.inv(ext)
    return ext

def get_depths(pc, parameters, savepath=False, name='test', height=HEIGHT, width=WIDTH):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pc)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(parameters, True)
    depths = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    # if savepath:
    #     o3d.io.write_pinhole_camera_parameters(savepath+name+'.json', parameters)
    #     np.save(savepath+name+'_depth.npy', depths)
    vis.poll_events()
    vis.update_renderer()
    vis.destroy_window()
    return depths