import os
import abc

import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image

from . import util
from . import io
from . import mesh_processing
from . import sim
from . import core


class Camera:
    """
    Holds intrinsic parameters as well as resolution.
    """
    def __init__(self, width, height, fx, fy, cx, cy):
        self.resolution = [width, height]
        self.intrinsic_parameters = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
        }

    @classmethod
    def create_kinect_like(cls):
        """
        Factory method that creates a kinect-like camera.
        """
        return cls(640, 480, 572.41140, 573.57043, 325.26110, 242.04899)

    @classmethod
    def from_camera_matrix(cls, width, height, camera_matrix):
        """
        Factory method that draws the intrinsics parameters from a camera matrix.
        Will only consider fx, fy, cx, cy.

        :param width: int, width of images from this camera
        :param height: int, height of images from this camera
        :param camera_matrix: (3, 3) ndarray, intrinsic parameters of camera
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        return cls(width, height, fx, fy, cx, cy)

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

    def point_cloud_from_depth(self, depth_image, camera_pose):
        """
        Takes a depth_image as well as a Camera object and computes the partial point cloud.

        :param depth_image: numpy array with distance values in [m] representing the depth image.
        :param camera_pose: (4, 4) ndarray, specifies the camera pose

        :return: (n, 3) array with xyz values of points.
        """
        w, h = self.resolution
        if not (w == depth_image.shape[1] and h == depth_image.shape[0]):
            raise ValueError(
                f'shape of depth image {depth_image.shape} does not fit camera resolution {self.resolution}')

        mask = np.where(depth_image > 0)
        x, y = mask[1], mask[0]

        fx, fy = self.intrinsic_parameters['fx'], self.intrinsic_parameters['fy']
        cx, cy = self.intrinsic_parameters['cx'], self.intrinsic_parameters['cy']

        world_x = (x - cx) * depth_image[y, x] / fx
        world_y = -(y - cy) * depth_image[y, x] / fy
        world_z = -depth_image[y, x]
        ones = np.ones(world_z.shape[0])

        points = np.vstack((world_x, world_y, world_z, ones))
        points = camera_pose @ points
        return points[:3, :].T


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


class RenderEngine(abc.ABC):
    """
    Abstract class for RenderEngines. Currently, we support pyrender and pybullet.
    This class defines a basic common interface and some common default values.
    """
    DEFAULT_Z_NEAR = 0.02
    DEFAULT_Z_FAR = 2

    @abc.abstractmethod
    def setup_scene(self, scene, camera, ambient_light=None, with_plane=False):
        """
        Call this method to prepare rendering -- setting up the scene. The scene can then be used repeatedly to
        render images from different camera poses.

        :param scene: core.Scene
        :param camera: render.Camera (for intrinsic parameters and resolution)
        :param ambient_light: list of 3 values, defining ambient light color and intensity
        :param with_plane: bool, if True, will add a ground plane to the scene
        """
        pass

    @abc.abstractmethod
    def render(self, camera_pose):
        """
        This method renders the scene from the given camera pose.

        :param camera_pose: (4, 4) ndarray, pose of camera as in OpenGL (z axis pointing away from the target)

        :return: (color_image, depth_image), ndarrays with dim (h, w, 3) and (h, w) respectively
        """
        pass


class PyRenderEngine(RenderEngine):
    def __init__(self):
        super().__init__()
        # parameters that can be adjusted
        self.zfar = self.DEFAULT_Z_FAR
        self.znear = self.DEFAULT_Z_NEAR

        # internals
        self._render_scene = None
        self._cam_node = None
        self._renderer = None

    def setup_scene(self, scene, camera, ambient_light=None, with_plane=False):
        if ambient_light is None:
            ambient_light = [1., 1., 1.]

        self._cam_node = pyrender.Node(
            camera=pyrender.IntrinsicsCamera(
                camera.intrinsic_parameters['fx'],
                camera.intrinsic_parameters['fy'],
                camera.intrinsic_parameters['cx'],
                camera.intrinsic_parameters['cy'],
                znear=self.znear,
                zfar=self.zfar
            )
        )

        self._render_scene = pyrender.Scene(ambient_light=ambient_light)
        for mesh in scene.get_mesh_list(with_plane=with_plane, as_trimesh=True):
            self._render_scene.add(pyrender.Mesh.from_trimesh(mesh))
        self._render_scene.add_node(self._cam_node)

        resolution = camera.resolution
        self._renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])

    def render(self, camera_pose):
        self._render_scene.set_pose(self._cam_node, pose=camera_pose)
        color, depth = self._renderer.render(self._render_scene)
        return color, depth


class PyBulletRenderEngine(sim.SimulatorBase, RenderEngine):
    def __init__(self):
        super().__init__()
        # parameters that can be adjusted
        self.zfar = self.DEFAULT_Z_FAR
        self.znear = self.DEFAULT_Z_NEAR

        # for retrieval after rendering
        self.segmentation_mask = None

        # internals
        self._projection_matrix = None
        self._ambient_light = None
        self._w = None
        self._h = None

    def setup_scene(self, scene, camera, ambient_light=None, with_plane=False):
        self.segmentation_mask = None  # reset

        if ambient_light is None:
            self._ambient_light = [1., 1., 1.]
        else:
            self._ambient_light = ambient_light

        self._reset(plane_and_gravity=with_plane)
        for instance in scene.objects:
            self._add_object(instance)
        for bg_instance in scene.bg_objects:
            self._add_object(bg_instance, fixed_base=True)

        # compute projection matrix from camera parameters
        # see https://stackoverflow.com/questions/60430958/ and https://stackoverflow.com/questions/22064084/
        # (plus some adjustments to pybullet)
        self._w, self._h = camera.resolution
        cx, cy = camera.intrinsic_parameters['cx'], camera.intrinsic_parameters['cy']
        fx, fy = camera.intrinsic_parameters['fx'], camera.intrinsic_parameters['fy']
        self._projection_matrix = np.array([
            [2*fx/self._w, 0, 0, 0],
            [0, 2*fy/self._h, 0, 0],
            [-(2*cx/self._w - 1), 2*cy/self._h - 1, (self.znear+self.zfar)/(self.znear-self.zfar), -1],
            [0, 0, (2 * self.znear * self.zfar) / (self.znear - self.zfar), 0]
        ])

    def render(self, camera_pose):
        view_matrix = np.linalg.inv(camera_pose).T
        w, h, rgb, depth, seg_mask = self._p.getCameraImage(
            self._w, self._h,
            viewMatrix=view_matrix.flatten(),
            projectionMatrix=self._projection_matrix.flatten()
        )
        self.segmentation_mask = seg_mask  # to allow retrieval (?)

        # post-processing
        rgb = rgb[:, :, :3]  # remove alpha

        # convert to meter, set non-existent depth values to zero
        no_depth_mask = depth == 1
        depth = self.zfar * self.znear / (self.zfar - (self.zfar - self.znear) * depth)
        depth[no_depth_mask] = 0

        return rgb, depth


class RenderEngineFactory:
    DEFAULT_ENGINE = 'pyrender'

    @staticmethod
    def create(engine=DEFAULT_ENGINE):
        """
        Factory method to create RenderEngines. Will choose a default engine if not requesting a specific one.

        :param engine: string, optional, either 'pyrender' or 'pybullet', if not provided, DEFAULT_ENGINE is used

        :return: object of requested subclass of RenderEngine
        """
        # could do something fancy like checking available dependencies to determine which one is the default engine
        if engine == 'pyrender':
            return PyRenderEngine()
        elif engine == 'pybullet':
            return PyBulletRenderEngine()
        else:
            raise NotImplementedError(f'{engine} render engine not implemented')


class ThumbnailRenderer:
    def __init__(self, engine=None, size=128):
        self._engine = engine or RenderEngineFactory.create()
        self._size = size

    def render(self, object_type, thumbnail_fn=None):
        """
        Creates a thumbnail for the given object type.

        :param object_type: core.ObjectType - first stable pose will be used for rendering, if available
        :param thumbnail_fn: filepath where to save the thumbnail. If None provided, it will not be saved.

        :return: (size, size, 3) ndarray with thumbnail of the object.
        """
        if object_type.stable_poses is None:
            pose = np.eye(4)
        else:
            pose = object_type.stable_poses[0][1].copy()

        scene = core.Scene()
        x = scene.ground_area[0] / 2
        y = scene.ground_area[1] / 2
        pose[0:2, 3] = [x, y]
        instance = core.ObjectInstance(object_type, pose)
        scene.objects.append(instance)

        # aim camera at mesh centroid
        centroid = mesh_processing.centroid(instance.get_mesh())
        camera_position = [0.3, 0.3, 0.2 + centroid[2]]
        camera_pose = util.look_at(position=camera_position, target=centroid, flip=True)

        self._engine.setup_scene(scene, Camera.create_kinect_like())
        image, depth = self._engine.render(camera_pose)

        # do clipping / resizing
        image = self._clip_and_scale(image, size=self._size)

        if thumbnail_fn is not None:
            io.make_sure_directory_exists(os.path.dirname(thumbnail_fn))
            imageio.imwrite(thumbnail_fn, image)

        return image

    @staticmethod
    def _clip_and_scale(image, bg_color=255, size=128):
        # assumes rgb image (w, h, c) and bg color = 255 ?
        intensity_img = np.mean(image, axis=2)

        # identify indices of non-background rows and columns, then look for min/max indices
        non_bg_rows = np.nonzero(np.mean(intensity_img, axis=1) != bg_color)
        non_bg_cols = np.nonzero(np.mean(intensity_img, axis=0) != bg_color)
        if len(non_bg_rows[0]) == 0:
            raise ValueError('cannot clip/scale image, as it is bg_color only')
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
