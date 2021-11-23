import os
import logging

import numpy as np
import quaternion
import trimesh
import pyrender
import open3d as o3d
from tqdm import tqdm
import imageio
from PIL import Image
try:
    import pyexr
    logging.debug('OpenEXR support: ACTIVE')
except ImportError:
    pyexr = None
    logging.debug('OpenEXR support: NONE')

# temporary
import pybullet
from . import sim

from . import util
from . import io
from . import mesh_processing


def _check_pyexr():
    if pyexr is None:
        raise ImportError('pyexr pacakge is missing. It depends on OpenEXR, for which some prerequisites must be ' +
                          'installed on the system. Please see https://stackoverflow.com/a/68102521/1264582 for ' +
                          'additional info on how to install this.')


class Camera:
    """
    holds intrinsic and extrinsic parameters, initialises with some Kinect-like intrinsics.
    """
    def __init__(self):
        self.resolution = [640, 480]  # w x h
        self.intrinsic_parameters = {
            'fx': 572.41140,
            'fy': 573.57043,
            'cx': 325.26110,
            'cy': 242.04899
        }
        self.pose = np.eye(4)

    def set_resolution(self, width: int, height: int):
        self.resolution = [width, height]

    def set_intrinsic_parameters(self, fx=None, fy=None, cx=None, cy=None):
        """
        overwrites only the given parameters, the others stay the same

        :param fx: focal length x
        :param fy: focal length y
        :param cx: principal point x
        :param cy: principal point y
        """
        if fx is not None:
            self.intrinsic_parameters['fx'] = fx
        if fy is not None:
            self.intrinsic_parameters['fy'] = fy
        if cx is not None:
            self.intrinsic_parameters['cx'] = cx
        if cy is not None:
            self.intrinsic_parameters['cy'] = cy

    def get_o3d_intrinsics(self):
        """
        :return: intrinsic parameters (incl. resolution) as instance of o3d.camera.PinholeCameraIntrinsic()
        """
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.resolution[0]),
            height=int(self.resolution[1]),
            fx=self.intrinsic_parameters['fx'],
            fy=self.intrinsic_parameters['fy'],
            cx=self.intrinsic_parameters['cx'],
            cy=self.intrinsic_parameters['cy']
        )
        return o3d_intrinsics

    def set_extrinsic_parameters(self, camera_pose):
        """
        sets the pose of the camera

        :param camera_pose: np 4x4 homogenous tf matrix
        """
        self.pose = camera_pose


class CameraPoseGenerator:
    """
    A class that offers various ways to generate camera poses. All poses look towards the center point. Their distance
    is between ``cam_distance_min`` and ``cam_distance_max``. ``rand_seed`` can be set to get reproducible results.
    You can also choose the hemispheres from which you want to get poses.

    :param cam_distance_min: float, minimum distance of camera poses (to origin)
    :param cam_distance_max: float, maximum distance of camera poses (to origin)
    :param upper_hemisphere: bool, whether to sample poses from upper hemisphere
    :param lower_hemisphere: bool, whether to sample poses from lower hemisphere
    :param center_point: (3,) array or list, the center point around which to construct camera poses
    :param rand_seed: int, provide seed for the rng for reproducible results, use None (default) for random seed
    """
    def __init__(self, cam_distance_min=0.6, cam_distance_max=0.9, upper_hemisphere=True,
                 lower_hemisphere=False, center_point=None, rand_seed=None):
        self.cam_distance_min = cam_distance_min
        self.cam_distance_max = cam_distance_max
        self.upper_hemisphere = upper_hemisphere
        self.lower_hemisphere = lower_hemisphere
        self.center_point = center_point
        self.rand_seed = rand_seed

    def _random_distance(self, rng=None):
        """
        Uses rng to produce a distance between `self.cam_distance_min` and `self.cam_distance_max`

        :param rng: a random number generator.

        :return: a random distance between `self.cam_distance_min` and `self.cam_distance_max`
        """
        if rng is None:
            rng = np.random.default_rng()

        return rng.random() * (self.cam_distance_max - self.cam_distance_min) + self.cam_distance_min

    def _check_hemisphere_setting(self):
        if not (self.upper_hemisphere or self.lower_hemisphere):
            raise ValueError('bad configuration: need at least one hemisphere to generate camera poses.')

    def _apply_offset(self, poses):
        if self.center_point is not None:
            # shift all poses towards center point
            offset = np.array(self.center_point).flatten()
            poses[:, 0:3, 3] += offset

    def random(self, n=64):
        """
        Samples random poses according to the specs given by the object.

        :return: (n, 4, 4) ndarray with transform matrices
        """
        self._check_hemisphere_setting()

        # hemi factor will be 1 if only upper, and -1 if only sample from lower (and 0 if both)
        hemi_factor = 1*self.upper_hemisphere - 1*self.lower_hemisphere

        rng = np.random.default_rng(self.rand_seed)
        poses = np.empty((n, 4, 4))

        for i in range(n):
            unit_vec = util.generate_random_unit_vector()
            # account for hemisphere sampling and just project to other hemisphere if necessary
            if unit_vec[2] * hemi_factor < 0:
                unit_vec[2] *= -1

            # apply distance and convert to matrix oriented towards origin
            cam_pos = unit_vec * self._random_distance(rng)
            poses[i] = util.look_at(cam_pos, target=[0, 0, 0], up=[0, 0, 1], flip=True)

        self._apply_offset(poses)
        return poses

    def _number_of_icosphere_poses(self, subdivisions, in_plane_rotations, scales):
        n_vertices = 4**subdivisions * 10 + 2

        if self.upper_hemisphere + self.lower_hemisphere == 1:
            # only one of both
            n_ground = 4 * 2**subdivisions
            n_vertices = (n_vertices - n_ground) / 2
        elif not (self.upper_hemisphere or self.lower_hemisphere):
            # none of both (configuration not useful but possible)
            return 0

        return int(n_vertices * scales * in_plane_rotations)

    def icosphere(self, subdivisions=2, in_plane_rotations=12, scales=2, random_distances=True):
        """
        The vertices of an icosphere will be used as camera positions, oriented such that they look towards the origin.
        In-plane rotations and distances will be applied accordingly. This is similar to what has been used e.g. for
        LINEMOD template matching (Hinterstoisser et al., ACCV, 2012).

        :param subdivisions: number of subdivisions of the icosahedron, determines number of view points
        :param in_plane_rotations: number of in-plane rotations for every view point
        :param scales: number of scales for every in-plane rotation
        :param random_distances: if True, will use random distances, if false, will evenly space distances between
                                 object's `cam_distance_min` and `cam_distance_max`

        :return: (n, 4, 4) ndarray of transform matrices
        """
        self._check_hemisphere_setting()

        # the vertices of the icosphere are our camera positions
        cam_positions = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1).vertices

        # only keep positions of appropriate hemisphere
        # it is debatable whether we want to keep the points on xy plane or not...
        if not self.lower_hemisphere:
            cam_positions = cam_positions[cam_positions[:, 2] > 0]
        if not self.upper_hemisphere:
            cam_positions = cam_positions[cam_positions[:, 2] < 0]

        # prepare distances (only used if not random)
        scale_distances = np.linspace(self.cam_distance_min, self.cam_distance_max, num=scales)

        # prepare in-plane rotations
        # this matrix will need to be applied in each step to rotate a bit more
        in_plane_rotation_matrix = np.eye(4)
        if in_plane_rotations > 1:
            angle = 2*np.pi / in_plane_rotations
            in_plane_rotation_matrix[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)],
                                                         [np.sin(angle), np.cos(angle)]])

        rng = np.random.default_rng(self.rand_seed)
        poses = np.empty((self._number_of_icosphere_poses(subdivisions, in_plane_rotations, scales), 4, 4))
        pose_idx = 0

        for cam_pos in cam_positions.reshape(-1, 3):
            # get the pose of the camera
            base_pose = util.look_at(cam_pos, target=[0, 0, 0], up=[0, 0, 1], flip=True)

            for i in range(in_plane_rotations):

                for s in range(scales):
                    # scale position using distances (orientation not affected)
                    if random_distances:
                        d = self._random_distance(rng)
                    else:
                        d = scale_distances[s]
                    poses[pose_idx] = np.array(base_pose, copy=True)
                    poses[pose_idx, 0:3, 3] = poses[pose_idx, 0:3, 3] * d
                    pose_idx += 1

                # now apply the in-plane rotation
                base_pose = base_pose @ in_plane_rotation_matrix

        self._apply_offset(poses)
        return poses


class MeshRenderer:
    """
    This class can render images of individual meshes and store them to files.
    Default intrinsic parameters will be set which resemble a Kinect and can be overridden using
    `set_camera_parameters()` function.

    :param output_dir: directory where to put files (ignored for thumbnail-rendering)
    :param camera: burg.render.Camera that holds relevant intrinsic parameters
    """
    def __init__(self, output_dir='../data/output/', camera=None):
        self.output_dir = output_dir
        io.make_sure_directory_exists(self.output_dir)

        if camera is None:
            self.camera = Camera()
        else:
            self.camera = camera

        self.cam_info_fn = 'CameraInfo'

    @staticmethod
    def _default_depth_fn_func(i):
        return f'depth{i:04d}'

    @staticmethod
    def _default_color_fn_func(i):
        return f'image{i:04d}'

    @staticmethod
    def _check_fn_types_are_supported(color_fn_type, depth_fn_type):
        supported_color_fn_types = ['png']
        supported_depth_fn_types = ['tum', 'exr', 'npy-pc']

        if color_fn_type is not None and color_fn_type not in supported_color_fn_types:
            raise ValueError(f'color file type must be one of {supported_color_fn_types} or None')
        if depth_fn_type is not None and depth_fn_type not in supported_depth_fn_types:
            raise ValueError(f'depth file type must be one of {supported_depth_fn_types} or None')
        if depth_fn_type == 'exr':
            _check_pyexr()  # automatically raises Exception if not satisfied
        return True

    def _setup_scene(self, mesh, camera_poses, ambient_light):
        mesh = mesh_processing.as_trimesh(mesh)

        # let's determine camera's znear and zfar as limits for rendering, with some safety margin (factor 2)
        # assuming we look onto origin
        magnitudes = np.linalg.norm(camera_poses[:, 0:3, 3], axis=-1)
        lower_bound = np.min(magnitudes) / 2
        upper_bound = np.max(magnitudes) * 2

        intrinsics = self.camera.intrinsic_parameters
        cam_node = pyrender.Node(
            camera=pyrender.IntrinsicsCamera(
                intrinsics['fx'],
                intrinsics['fy'],
                intrinsics['cx'],
                intrinsics['cy'],
                znear=lower_bound, zfar=upper_bound))

        # setup the pyrender scene
        render_scene = pyrender.Scene(ambient_light=ambient_light)
        render_scene.add(pyrender.Mesh.from_trimesh(mesh))
        render_scene.add_node(cam_node)
        return render_scene, cam_node

    def render_depth(self, mesh, camera_poses, sub_dir='', depth_fn_type='tum', depth_fn_func=None):
        """shorthand for MeshRenderer.render(), but renders depth images only"""
        self.render(mesh, camera_poses, sub_dir=sub_dir, depth_fn_type=depth_fn_type, depth_fn_func=depth_fn_func,
                    color_fn_type=None)

    def render_color(self, mesh, camera_poses, sub_dir='', color_fn_type='png', color_fn_func=None):
        """shorthand for MeshRenderer.render(), but renders color images only"""
        self.render(mesh, camera_poses, sub_dir=sub_dir, color_fn_type=color_fn_type, color_fn_func=color_fn_func,
                    depth_fn_type=None)

    def render(self, mesh, camera_poses, sub_dir='', depth_fn_type='tum', depth_fn_func=None,
               color_fn_type='png', color_fn_func=None):
        """
        Renders depth images of the given mesh, from the given camera poses.
        Uses the configuration of the MeshRenderer object.

        :param mesh: o3d.geometry.TriangleMesh or trimesh.Trimesh - the mesh which shall be rendered
        :param camera_poses: (4, 4) or (n, 4, 4) ndarray with poses
        :param sub_dir: directory where to produce output files (will be relative to the object's `output_dir`)
        :param depth_fn_type: file format to store rendered depth, defaults to 'tum' (png), 'exr' also possible, or
                              save directly as point cloud in 'npy-pc' numpy files.
                              If None, no depth data will be saved
        :param depth_fn_func: function to generate filenames (string) from integer, default function when None
        :param color_fn_type: file format to store rendered color images, defaults to 'png', if None, no color images
                              will be rendered
        :param color_fn_func: function to generate filenames (string) from integer, default function when None
        """
        self._check_fn_types_are_supported(color_fn_type, depth_fn_type)
        if depth_fn_func is None:
            depth_fn_func = self._default_depth_fn_func
        if color_fn_func is None:
            color_fn_func = self._default_color_fn_func

        output_dir = os.path.join(self.output_dir, sub_dir)
        io.make_sure_directory_exists(output_dir)

        # make sure shape fits if only one pose provided
        camera_poses = camera_poses.reshape(-1, 4, 4)
        render_scene, cam_node = self._setup_scene(mesh, camera_poses, ambient_light=[0.3, 0.3, 0.3])

        # set up rendering settings
        resolution = self.camera.resolution
        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])

        # just for observation
        n_points = np.empty(len(camera_poses))

        for i in tqdm(range(len(camera_poses))):
            render_scene.set_pose(cam_node, pose=camera_poses[i])
            # color is rgb, depth is mono float in [m]
            color, depth = r.render(render_scene)

            if depth_fn_type is not None:
                n_points[i] = np.count_nonzero(depth)
                if depth_fn_type == 'tum':
                    # use tum file format (which is actually a scaled 16bit png)
                    # https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
                    depth_fn = os.path.join(output_dir, depth_fn_func(i) + '.png')
                    imageio.imwrite(depth_fn, (depth * 5000).astype(np.uint16))
                elif depth_fn_type == 'exr':
                    # store images to file (extend to three channels and store in exr)
                    # this is for compatibility with GPNet dataset, although it bloats the file size
                    img = np.repeat(depth, 3).reshape(depth.shape[0], depth.shape[1], 3)
                    depth_fn = os.path.join(output_dir, depth_fn_func(i) + '.exr')
                    pyexr.write(depth_fn, img, channel_names=['R', 'G', 'B'], precision=pyexr.FLOAT)
                elif depth_fn_type == 'npy-pc':
                    # convert to point cloud data and store that in npy file
                    self.camera.pose = camera_poses[i]
                    pc = point_cloud_from_depth(depth, self.camera)
                    pc_fn = os.path.join(output_dir, depth_fn_func(i) + '.npy')
                    np.save(pc_fn, pc)

            if color_fn_type is not None:
                if color_fn_type == 'png':
                    imageio.imwrite(os.path.join(output_dir, color_fn_func(i) + '.png'), color)

        logging.debug(f'nonzero pixels in depth images (n points): avg {np.mean(n_points)}, min {np.min(n_points)}, ' +
                      f'max {np.max(n_points)}')

        # save camera info to npy file // format based on GPNet dataset
        cam_info = np.empty(len(camera_poses), dtype=([
            ('id', 'S16'),
            ('position', '<f4', (3,)),
            ('orientation', '<f4', (4,)),
            ('calibration_matrix', '<f8', (9,)),
            ('distance', '<f8')]))

        cam_info['id'] = [f'view{i}'.encode('UTF-8') for i in range(len(camera_poses))]
        cam_info['position'] = camera_poses[:, 0:3, 3]
        cam_info['orientation'] = quaternion.as_float_array(quaternion.from_rotation_matrix(camera_poses[:, 0:3, 0:3]))

        intrinsics = self.camera.intrinsic_parameters
        cam_info['calibration_matrix'] = np.array([intrinsics['fx'], 0, intrinsics['cx'],
                                                   0, intrinsics['fy'], intrinsics['cy'],
                                                   0, 0, 1])
        # not really sure what to put in the distance field
        # alternative would be to find some actual distance to object (e.g. depth value at center point), but
        # this seems arbitrary as well. i don't think it's used by gpnet anyways.
        cam_info['distance'] = np.linalg.norm(cam_info['position'], axis=-1)
        np.save(os.path.join(output_dir, self.cam_info_fn), cam_info)

    @staticmethod
    def _clip_and_scale(image, bg_color=255, size=128):
        # assumes rgb image (w, h, c) and bg color = 255 ?
        intensity_img = np.mean(image, axis=2)

        # identify indices of non-background rows and columns, then look for min/max indices
        non_bg_rows = np.nonzero(np.mean(intensity_img, axis=1) != bg_color)
        non_bg_cols = np.nonzero(np.mean(intensity_img, axis=0) != bg_color)
        r1, r2 = np.min(non_bg_rows), np.max(non_bg_rows)
        c1, c2 = np.min(non_bg_cols), np.max(non_bg_cols)

        # create square white image with some margin, fit in the crop
        h, w = r2+1-r1, c2+1-c1
        new_width = max(h, w)
        thumbnail = np.full((new_width, new_width, image.shape[2]), bg_color)
        start_h = int((new_width-h)/2)
        start_w = int((new_width-w)/2)
        thumbnail[start_h:start_h+h, start_w:start_w+w, :] = image[r1:r2+1, c1:c2+1, :]

        # use PIL Image to resize and convert back to numpy
        thumbnail = np.array(Image.fromarray(np.uint8(thumbnail)).resize((size, size)))
        return thumbnail

    def render_thumbnail(self, mesh, camera_position=None, thumbnail_fn=None, size=128):
        """
        Creates a square thumbnail of the given mesh object.

        :param mesh: Can be open3d.geometry.TriangleMesh or Trimesh
        :param camera_position: 3d-vector with desired position of camera, some reasonable default is preset. From that
                                position, the camera will be oriented towards the centroid of the mesh.
        :param thumbnail_fn: filepath where to save the thumbnail. If None provided, it will not be saved.
        :param size: The desired width/height of the thumbnail.

        :return: (size, size, 3) ndarray with thumbnail of the object.
        """
        # aim camera at mesh centroid
        centroid = mesh_processing.centroid(mesh)
        if camera_position is None:
            camera_position = [0.3, 0.3, 0.2 + centroid[2]]

        camera_pose = util.look_at(position=camera_position, target=centroid, flip=True)
        render_scene, cam_node = self._setup_scene(mesh, camera_pose[None, :, :], ambient_light=[1., 1., 1.])

        resolution = self.camera.resolution
        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])

        render_scene.set_pose(cam_node, pose=camera_pose)
        color, _ = r.render(render_scene)

        # do clipping / resizing
        color = self._clip_and_scale(color, size=size)

        if thumbnail_fn is not None:
            io.make_sure_directory_exists(os.path.dirname(thumbnail_fn))
            imageio.imwrite(thumbnail_fn, color)

        return color


def point_cloud_from_depth(depth_image, camera):
    """
    Takes a depth_image as well as a Camera object and computes the partial point cloud.

    :param depth_image: numpy array with distance values in [m] representing the depth image.
    :param camera: Camera object describing the camera parameters applying to the depth image.

    :return: (n, 3) array with xyz values of points.
    """
    w, h = camera.resolution
    if not (w == depth_image.shape[1] and h == depth_image.shape[0]):
        raise ValueError(f'shape of depth image {depth_image.shape} does not fit camera resolution {camera.resolution}')

    mask = np.where(depth_image > 0)
    x, y = mask[1], mask[0]

    fx, fy = camera.intrinsic_parameters['fx'], camera.intrinsic_parameters['fy']
    cx, cy = camera.intrinsic_parameters['cx'], camera.intrinsic_parameters['cy']

    world_x = (x - cx) * depth_image[y, x] / fx
    world_y = -(y - cy) * depth_image[y, x] / fy
    world_z = -depth_image[y, x]
    ones = np.ones(world_z.shape[0])

    points = np.vstack((world_x, world_y, world_z, ones))
    points = camera.pose @ points
    return points[:3, :].T


def render_orthographic_projection(scene, px_per_mm=2, z_min=None, z_max=None, transparent=False):
    """
    Renders an orthographic projection of the scene onto its ground plane in xy.
    The background can be made transparent.
    If desired, the meshes can be cut off at any particular height using `z_min` and `z_max`.

    :param scene: A scene to be projected onto the xy plane.
    :param px_per_mm: Resolution of the target image, pixels per mm
    :param z_min: Disregard all vertices below z_min (or None to include all)
    :param z_max: Disregard all vertices above z_max (or None to include all)
    :param transparent: Whether or not to make the background transparent. Else white.

    :return: (w, h, 3) numpy image if not transparent, (w, h, 4) if transparent
    """
    # set the objects into the pyrender scene
    render_scene = pyrender.Scene(ambient_light=[0.8, 0.8, 0.8])
    meshes = scene.get_mesh_list(with_plane=True)
    for mesh in meshes:
        mesh = mesh_processing.as_trimesh(mesh)
        vertices = mesh.vertices
        mask = np.full(len(vertices), fill_value=True, dtype=bool)
        if z_min is not None:
            mask &= vertices[:, 2] > z_min
        if z_max is not None:
            mask &= vertices[:, 2] < z_max
        mesh.update_vertices(mask)
        # todo: the update_vertices occasionally adds some weird faces sometimes
        # maybe we can do this more cleanly by first removing the faces referring to the vertices we want to
        # remove and only then removing those vertices (?)

        render_scene.add(pyrender.Mesh.from_trimesh(mesh))

    # create orthographic camera
    # magnitude in x is actually ignored, only magnitude in y is relevant
    # mag =
    camera = pyrender.OrthographicCamera(0.2, .15, znear=0.05, zfar=2)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, scene.ground_area[0] / 2],
        [0.0, 1.0, 0.0, scene.ground_area[1] / 2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ])
    render_scene.add(camera, pose=cam_pose)
    res = tuple(int(px_per_mm * x * 1000) for x in scene.ground_area)  # 1000 to convert from [m] to [mm]
    r = pyrender.OffscreenRenderer(res[0], res[1])
    print('offscreen renderer platform:', r._platform)
    color, depth = r.render(render_scene)

    if transparent:
        # convert to RGBA and make white pixels transparent
        color = Image.fromarray(color).convert('RGBA')
        data = color.load()
        width, height = color.size
        for y in range(height):
            for x in range(width):
                if data[x, y] == (255, 255, 255, 255):
                    data[x, y] = (255, 255, 255, 0)
        color = np.array(color)

    return color
