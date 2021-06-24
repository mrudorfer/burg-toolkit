import os

import numpy as np
import trimesh
import pyrender
import open3d as o3d
from tqdm import tqdm
try:
    import pyexr
    print('OpenEXR support: ACTIVE')
except ImportError:
    pyexr = None
    print('OpenEXR support: NONE')

# todo: include pyexr, pyrender dependencies in package
# rendering requires openexr which does not install without prerequisites.
# see https://stackoverflow.com/a/68102521/1264582

from . import util
from . import scene
from . import visualization
## todo temp


def _check_pyexr():
    if pyexr is None:
        raise ImportError('pyexr pacakge is missing. It depends on OpenEXR, for which some prerequisites must be ' +
                          'installed on the system. Please see https://stackoverflow.com/a/68102521/1264582 for ' +
                          'additional info on how to install this.')


class CameraPoseGenerator:
    """
    A class that offers various ways to generate camera poses. All poses look towards the origin. Their distance is
    between ``cam_distance_min`` and ``cam_distance_max``. ``rand_seed`` can be set to get reproducible results.
    You can also choose the hemispheres from which you want to get poses.

    :param cam_distance_min: float, minimum distance of camera poses (to origin)
    :param cam_distance_max: float, maximum distance of camera poses (to origin)
    :param upper_hemisphere: bool, whether to sample poses from upper hemisphere
    :param lower_hemisphere: bool, whether to sample poses from lower hemisphere
    :param rand_seed: int, provide seed for the rng for reproducible results, use None (default) for random seed
    """
    def __init__(self, cam_distance_min=0.6, cam_distance_max=0.9, upper_hemisphere=True,
                 lower_hemisphere=False, rand_seed=None):
        self.cam_distance_min = cam_distance_min
        self.cam_distance_max = cam_distance_max
        self.upper_hemisphere = upper_hemisphere
        self.lower_hemisphere = lower_hemisphere
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

        return poses


class MeshRenderer:
    """
    This class can render images of individual meshes and store them to files.
    Default intrinsic parameters will be set which resemble a Kinect and can be overridden using
    `set_camera_parameters()` function.

    :param mesh: o3d.geometry.TriangleMesh - the mesh which shall be rendered
    :param output_dir: directory where to put files
    :param camera: burg.scene.Camera that holds relevant intrinsic parameters
    """
    def __init__(self, mesh, output_dir='../data/output/', camera=None):
        self.output_dir = output_dir
        self.mesh = mesh
        if camera is None:
            self.camera = scene.Camera()
        else:
            self.camera = camera

    def render_depth(self, camera_poses):
        #############################################
        # prepare the scene:
        # load model, configure camera and lights
        #############################################

        # if only one provided
        camera_poses = camera_poses.reshape(-1, 4, 4)

        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            tmesh = util.o3d_mesh_to_trimesh(self.mesh)
        else:
            # assume trimesh
            tmesh = self.mesh

        if not isinstance(tmesh, trimesh.Trimesh):
            raise ValueError(f'provided mesh must be either o3d.geometry.TriangleMesh or trimesh.Trimesh, but ' +
                             f'is {type(tmesh)} instead.')

        render_scene = pyrender.Scene()
        render_scene.add(pyrender.Mesh.from_trimesh(tmesh))

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
        render_scene.add_node(cam_node)

        # todo todo todo
        # do we need light for depth images?
        # set up the light -- a single spot light in the same spot as the camera
        # light_node = pyrender.Node(light=pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0))
        # render_scene.add_node(light_node)

        ###############################
        # start rendering loop
        ###############################

        # set up rendering settings
        resolution = self.camera.resolution
        r = pyrender.OffscreenRenderer(resolution[0], resolution[1])

        for i in tqdm(range(len(camera_poses))):
            # get next pose and set camera and light accordingly
            render_scene.set_pose(cam_node, pose=camera_poses[i])
            # render_scene.set_pose(light_node, pose=camera_poses[i])

            # render the color and depth images
            # color is rgb, depth is mono float in [m]
            _, depth = r.render(render_scene)

            # create mask... we might not need this, just let it be zero (?)
            # mask = depth == 0
            # depth[mask] = np.inf

            # store images to file (extend to three channels and store in exr)
            depth_fn = f'render{i:d}Depth0001.exr'
            img = np.repeat(depth, 3).reshape(depth.shape[0], depth.shape[1], 3)
            _check_pyexr()  # raises error if package is not available
            pyexr.write(os.path.join(self.output_dir, depth_fn), img, channel_names=['R', 'G', 'B'])

    # save camera info to npy file
        pass
