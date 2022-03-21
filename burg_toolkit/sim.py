import logging
import time
import os

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from matplotlib import pyplot as plt

from . import util, io
from .gripper import MountedGripper


_log = logging.getLogger(__name__)


class SimulatorBase:
    """
    This is an abstract base class for all simulators.
    It ensures that settings are consistent across different simulator use cases and provides some convenience
    methods.

    :param verbose: If set to True, it will show the simulation in GUI mode.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.dt = 1. / 240.  # this is the default and should not be changed light-heartedly
        self.SOLVER_STEPS = 100  # a bit more than default helps in contact-rich tasks
        self.TIME_SLEEP = self.dt * 2  # for visualization
        self.LATERAL_FRICTION = 1.0
        self.SPINNING_FRICTION = 0.003  # defaults for loading objects / robots
        self.ROLLING_FRICTION = 0.0001
        self.MIN_OBJ_MASS = 0.05  # small masses will be replaced by this (could tune a bit more, in combo with solver)
        self.JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

        self._color_idx = 0
        self.color_map = plt.get_cmap('tab20')
        self._moving_bodies = {}
        self._env_bodies = {}
        self._coms = {}  # dictionary to store center of mass belonging to the body id's
        self._simulated_steps = 0

        self._step_funcs = []  # functions that will be executed on stepping

        self._recording_config = None

        self._p = None
        self._reset()

    @property
    def simulated_seconds(self):
        """Gives the simulated time in seconds."""
        return self._simulated_steps * self.dt

    def _reset(self, plane_and_gravity=False):
        """
        This method resets the simulation to the starting point. Shall be used to clean up after a simulation run.

        :param plane_and_gravity: If yes, will call _load_plane_and_gravity() with default arguments after resetting.
        """
        if self._p is None:
            # connect using bullet client makes sure we can connect to multiple servers in parallel
            # options="--mp4=moviename.mp4" (records movie, requires ffmpeg)
            self._p = bullet_client.BulletClient(connection_mode=p.GUI if self.verbose else p.DIRECT)
        else:
            self._p.resetSimulation()

        self._moving_bodies = {}
        self._env_bodies = {}
        self._coms = {}
        self._color_idx = 0
        self._simulated_steps = 0
        self._step_funcs = []

        self._p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=self.SOLVER_STEPS)
        if self.verbose:
            self._p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=0, cameraPitch=-30,
                                               cameraTargetPosition=[0, 0, 0.1])
        if plane_and_gravity:
            self._load_plane_and_gravity()

    def _load_plane_and_gravity(self, plane_id='plane'):
        """
        Loads a plane and sets gravity.

        :param plane_id: string, the body ID to be used for the plane (in self._body_ids dict)
        """
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._env_bodies[plane_id] = self._p.loadURDF("plane.urdf")
        self._p.setGravity(0, 0, -9.81)
        self._p.changeDynamics(self._env_bodies[plane_id], -1, lateralFriction=1.0)

    def register_step_func(self, step_func):
        if step_func in self._step_funcs:
            _log.debug('step func was already registered. will only be executed once per step.')
            return
        self._step_funcs.append(step_func)

    def unregister_step_func(self, step_func):
        if step_func not in self._step_funcs:
            _log.debug('asking me to unregister step func, but step func was not registered in the first place.')
            return
        self._step_funcs.remove(step_func)

    def step(self, n=1, seconds=None):
        """
        Steps the simulation for n steps if seconds is None.
        If seconds provided, will simulate the equivalent number of steps.
        Using `register_step_func` you can provide functions that are executed before each step. To unregister, use
        `unregister_step_func`.

        :param n: number of steps, defaults to 1
        :param seconds: number of seconds to step, if given, parameter n will be overridden
        """
        if seconds is not None:
            n = int(seconds / self.dt)
        for i in range(n):
            for step_func in self._step_funcs:
                step_func()
            self._p.stepSimulation()
            self._simulated_steps += 1
            if self.verbose:
                time.sleep(self.TIME_SLEEP)

    def dismiss(self):
        """
        This method shall be called when the simulation is not needed anymore as it cleans up the object.
        """
        self._p.disconnect()

    def _get_next_color(self):
        """
        Returns a new color from the colormap and moves the index forward.
        Color is used to "paint" objects so they are distinguishable in visual mode.
        """
        color = self.color_map(self._color_idx)
        self._color_idx = (self._color_idx + 1) % self.color_map.N
        return color

    def add_object(self, object_instance, fixed_base=False):
        """
        Adds an object to the simulator.

        :param object_instance: core.ObjectInstance (with type and pose)
        :param fixed_base: if True, the object is immovable (defaults to False), immovable objects usually not
                           considered in collision detection etc., as they count as environment

        :return: object id if object could be added, else raises an Error
        """
        if object_instance.object_type.urdf_fn is None:
            raise ValueError(f'object instance of type {object_instance.object_type.identifier} has no urdf_fn.')
        if not os.path.exists(object_instance.object_type.urdf_fn):
            raise ValueError(f'could not find urdf file for object type {object_instance.object_type.identifier}.' +
                             f'expected it at {object_instance.object_type.urdf_fn}.')

        # pybullet uses center of mass as reference for the transforms in BasePositionAndOrientation
        # except in loadURDF - i couldn't figure out which reference system is used in loadURDF
        # because just putting the pose of the instance (i.e. the mesh's frame?) is not (always) working
        # workaround:
        #   all our visual/collision models have the same orientation, i.e. it is only the offset to COM
        #   add obj w/o pose, get COM, compute the transform burg2py manually and resetBasePositionAndOrientation
        object_id = self._p.loadURDF(object_instance.object_type.urdf_fn)
        if object_id < 0:
            raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

        self._coms[object_id] = np.array(self._p.getDynamicsInfo(object_id, -1)[3])
        tf_burg2py = np.eye(4)
        tf_burg2py[0:3, 3] = self._coms[object_id]
        start_pose = object_instance.pose @ tf_burg2py
        pos, quat = util.position_and_quaternion_from_tf(start_pose, convention='pybullet')
        self._p.resetBasePositionAndOrientation(object_id, pos, quat)

        # dynamics don't work for very small masses, so let's increase mass if necessary
        mass = np.max([object_instance.object_type.mass, self.MIN_OBJ_MASS])
        if fixed_base:
            mass = 0
        self._p.changeDynamics(object_id, -1, lateralFriction=object_instance.object_type.friction_coeff,
                               spinningFriction=self.SPINNING_FRICTION, rollingFriction=self.ROLLING_FRICTION,
                               mass=mass)

        if self.verbose:
            self._p.changeVisualShape(object_id, -1, rgbaColor=self._get_next_color())
            _log.debug(f'added object {object_instance.object_type.identifier}')

        # add to the corresponding body id dicts to keep track of objects
        body_id_dict = self._env_bodies if fixed_base else self._moving_bodies
        if object_instance in body_id_dict.keys():
            raise ValueError('already added the object instance to simulator, please remove it first')
        body_id_dict[object_instance] = object_id
        return object_id

    def add_scene(self, scene, with_bg_objects=True):
        """
        adding all object instances of a scene to the simulator

        :param scene: core.Scene
        :param with_bg_objects: whether to include bg_objects
        """
        for instance in scene.objects:
            self.add_object(instance)
        if with_bg_objects:
            for bg_instance in scene.bg_objects:
                self.add_object(bg_instance, fixed_base=True)

    def load_robot(self, urdf_file, position=None, orientation=None, fixed_base=False, lateral_friction=None,
                   spinning_friction=None, rolling_friction=None, friction_anchor=False):
        """
        Loads a robot and creates a data structure to access all of the robots info as well.

        :param urdf_file: string containing the path to the urdf file.
        :param position: (3,) base position; optional, defaults to [0, 0, 0]
        :param orientation: (x, y, z, w) base orientation; optional, defaults to [0, 0, 0, 1]
        :param fixed_base: whether or not to use fixed base; optional, defaults to False
        :param lateral_friction: lateral friction to be set for each joint, if None some defaults are used
        :param spinning_friction: spinning friction, if None the SimulatorBase.LATERAL_FRICTION is used
        :param rolling_friction: rolling friction, using default from SimulatorBase if None
        :param friction_anchor: whether to use friction anchor. pybullet docs are not very helpful on this...
        """
        if position is None:
            position = [0, 0, 0]
        if orientation is None:
            orientation = [0, 0, 0, 1]

        body_id = self._p.loadURDF(urdf_file, basePosition=position, baseOrientation=orientation,
                                   useFixedBase=int(fixed_base))

        num_joints = self._p.getNumJoints(body_id)
        joint_infos = {}
        for joint_idx in range(num_joints):
            self._p.changeDynamics(
                body_id,
                joint_idx,
                lateralFriction=lateral_friction or self.LATERAL_FRICTION,
                spinningFriction=spinning_friction or self.SPINNING_FRICTION,
                rollingFriction=rolling_friction or self.ROLLING_FRICTION,
                frictionAnchor=friction_anchor
            )
            joint_info = self._get_joint_info(body_id, joint_idx)
            joint_infos[joint_info['link_name']] = joint_info

        # keep track of bodies
        i = 0
        name = f'robot_{i}'
        while name in self._moving_bodies.keys():
            i += 1
            name = f'robot_{i}'
        self._moving_bodies[name] = body_id

        return body_id, joint_infos

    def look_up_body_id(self, body_key, moving_objects=True, env_objects=True):
        """
        find the body ID of an object

        :param body_key: can be the body_id itself, or a str, or an object instance, etc...
        :param moving_objects: whether to look in moving objects
        :param env_objects: whether to look in env_objects

        :return: body_id used in pybullet
        """
        if isinstance(body_key, int):
            # is most likely already a body id
            return body_key
        if moving_objects:
            if body_key in self._moving_bodies.keys():
                return self._moving_bodies[body_key]
        if env_objects:
            if body_key in self._env_bodies.keys():
                return self._env_bodies[body_key]
        # nothing found...
        raise ValueError(f'could not find a body with the corresponding key: {body_key}')

    def get_body_pose(self, body_id, convert2burg=False):
        """
        Returns the base position and orientation of the body with respect to center of mass frame as used by
        pybullet. If `convert2burg` is True, it will be transformed back to normal frame of reference.

        :param body_id: either the pybullet body id as int, or a string used in the self._body_ids dict.
        :param convert2burg: If set to True, frame of reference is world instead of center of mass.

        :return: (4, 4) transformation matrix describing the pose of the object
        """
        body_id = self.look_up_body_id(body_id)
        pos, quat = self._p.getBasePositionAndOrientation(body_id)
        pose = util.tf_from_pos_quat(pos, quat, convention="pybullet")
        if convert2burg:
            if body_id not in self._coms.keys():
                self._coms[body_id] = np.array(self._p.getDynamicsInfo(body_id, -1)[3])
            tf_py2burg = np.eye(4)
            tf_py2burg[0:3, 3] = -self._coms[body_id]
            pose = pose @ tf_py2burg
        return pose

    def _get_joint_info(self, body_id, joint_id):
        """returns a dict with some joint info"""
        # todo: make joint_info a class so we don't have to memorise the keys
        info = self._p.getJointInfo(body_id, joint_id)
        joint_info = {
            'id': info[0],
            'link_name': info[12].decode("utf-8"),
            'joint_name': info[1].decode("utf-8"),
            'type': self.JOINT_TYPES[info[2]],
            'friction': info[7],
            'lower_limit': info[8],
            'upper limit': info[9],
            'max_force': info[10],
            'max_velocity': info[11],
            'joint_axis': info[13],
            'parent_pos': info[14],
            'parent_orn': info[15]
        }
        return joint_info

    def _inspect_body(self, body_key):
        """
        prints out some debug info for the given object
        """
        print('****')
        body_id = self.look_up_body_id(body_key)
        print(f'inspecting body id {body_id} ({body_key})')
        print(f'body info: {self._p.getBodyInfo(body_id)}')
        num_joints = self._p.getNumJoints(body_id)
        print(f'num joints: {num_joints}')
        for i in range(num_joints):
            print(f'joint {i}:')
            [print(f'\t{key}: {val}') for key, val in self._get_joint_info(body_id, i).items()]

    def _print_joint_positions(self, body_key):
        """
        prints out the positions of all joints of the body

        :param body_id: id of the body
        """
        body_id = self.look_up_body_id(body_key)
        num_joints = self._p.getNumJoints(body_id)
        print(f'getting {num_joints} joint positions of body {body_id}, {self._p.getBodyInfo(body_id)}')
        joint_states = self._p.getJointStates(body_id, list(range(num_joints)))
        for joint_state in joint_states:
            print(f'\t{joint_state[0]}')

    def are_in_collision(self, body_key_1, body_key_2, threshold=-0.001):
        """
        checks if two bodies are in collision with each other.

        :param body_key_1: first body key
        :param body_key_2: second body key
        :param threshold: float, distance upon which we recognise it as a collision

        :return: bool, True if the two bodies are in collision
        """
        body_id_1 = self.look_up_body_id(body_key_1)
        body_id_2 = self.look_up_body_id(body_key_2)

        # in contrast to getContactPoints, this also works before stepSimulation or performCollisionDetection
        distance = 0.01  # do not return any points for objects that are farther apart than this
        points = self._p.getClosestPoints(body_id_1, body_id_2, distance)

        _log.debug(f'checking collision between {self._p.getBodyInfo(body_id_1)} and {self._p.getBodyInfo(body_id_2)}')
        _log.debug(f'found {len(points)} points that are close')

        n_colliding_points = 0
        distances = []
        for point in points:
            distance = point[8]
            distances.append(distance)
            if distance < threshold:
                n_colliding_points += 1

        _log.debug(f'and {n_colliding_points} points\' distance is below threshold of {threshold}')
        if distances:
            _log.debug(f'minimum distance is: {min(distances)}')
        return n_colliding_points > 0

    def are_in_contact(self, body_id_1, link_id_1, body_id_2, link_id_2):
        """
        checks if the links of two bodies are in contact. does not work with keys, just with the id.
        also relies on a prior call to stepSimulation
        """
        assert self._simulated_steps > 0, 'SimulatorBase.are_in_contact() can only be called after step()'
        contacts = self._p.getContactPoints(body_id_1, body_id_2, link_id_1, link_id_2)
        return len(contacts) > 0

    def _save_image(self):
        assert self._recording_config is not None, 'recording not configured'

        fps = self._recording_config['fps']
        if self._simulated_steps % int(1/(fps*self.dt)) != 0:
            return

        width, height, rgb, depth, seg_mask = self._p.getCameraImage(
            self._recording_config['w'],
            self._recording_config['h'],
            viewMatrix=self._recording_config['view_matrix'],
            projectionMatrix=self._recording_config['projection_matrix']
        )

        rgb = rgb[:, :, :3]  # remove alpha
        index = self._simulated_steps // int(1/(fps*self.dt))
        filename = self._recording_config['filename'] + f'_{index:04d}.png'
        io.save_image(filename, rgb)

    def configure_recording(self, filename, camera, camera_pose, fps=24):
        """
        Use this to continuously capture images from the simulation. Afterwards, they can be combined to a gif or
        mp4 using ffmpeg.
        Note that if the simulation is reset, the recording configuration is lost. You should hence call this function
        after the simulation has been set up, but before it starts simulating.

        :param filename: string, base filename of images, including path. Frame index and file type are added
                         automatically.
        :param camera: render.Camera object
        :param camera_pose: ndarray, 4x4
        :param fps: int, desired frames per second (of simulation time)
        """
        self._recording_config = None

        # compute projection matrix for given camera
        znear, zfar = 0.02, 5
        w, h = camera.resolution
        cx, cy = camera.intrinsic_parameters['cx'], camera.intrinsic_parameters['cy']
        fx, fy = camera.intrinsic_parameters['fx'], camera.intrinsic_parameters['fy']
        projection_matrix = np.array([
            [2 * fx / w, 0, 0, 0],
            [0, 2 * fy / h, 0, 0],
            [-(2 * cx / w - 1), 2 * cy / h - 1, (znear + zfar) / (znear - zfar), -1],
            [0, 0, (2 * znear * zfar) / (znear - zfar), 0]
        ])

        # compute view matrix
        view_matrix = np.linalg.inv(camera_pose).T

        # save config
        self._recording_config = {
            'w': w,
            'h': h,
            'projection_matrix': projection_matrix.flatten(),
            'view_matrix': view_matrix.flatten(),
            'filename': filename,
            'fps': fps
        }

        # register save image function
        self.register_step_func(self._save_image)

    def stop_recording(self):
        """
        After call to configure_recording(), this can be used to stop the recording again.
        """
        self.unregister_step_func(self._save_image)
        self._recording_config = None


class GraspScores:
    COLLISION_WITH_GROUND = 0
    COLLISION_WITH_TARGET = 1
    COLLISION_WITH_CLUTTER = 2
    NO_CONTACT_ESTABLISHED = 3
    SLIPPED_DURING_LIFTING = 4
    SUCCESS = 5

    _s2c_dict = {
        SUCCESS: ([0.1, 0.8, 0.1], 'successfully lifted', 'green'),
        COLLISION_WITH_GROUND: ([0.8, 0.1, 0.1], 'collision with ground', 'red'),
        COLLISION_WITH_TARGET: ([0.4, 0.1, 0.1], 'collision with target object', 'dark red'),
        COLLISION_WITH_CLUTTER: ([0.1, 0.1, 0.8], 'collision with clutter', 'blue'),
        NO_CONTACT_ESTABLISHED: ([0.1, 0.1, 0.4], 'no contact established', 'dark blue'),
        SLIPPED_DURING_LIFTING: ([0.1, 0.4, 0.1], 'object slipped', 'dark green'),
    }

    @classmethod
    def _retrieve(cls, scores, item):
        if not (isinstance(scores, list) or isinstance(scores, np.ndarray)):
            no_list = True
            scores = [scores]
        else:
            no_list = False

        results = []
        for score in scores:
            if score in cls._s2c_dict.keys():
                results.append(cls._s2c_dict[score][item])
            else:
                raise ValueError(f'score value {score} is unknown. only have {cls._s2c_dict.keys()}')

        if no_list:
            return results[0]
        else:
            return results

    @classmethod
    def score2color(cls, score):
        return cls._retrieve(score, 0)

    @classmethod
    def score2description(cls, score):
        return cls._retrieve(score, 1)

    @classmethod
    def score2color_name(cls, score):
        return cls._retrieve(score, 2)


class GraspSimulator(SimulatorBase):
    """
    Simulates grasp executions in a scene. Will be reset after each grasp trial.

    :param scene: the scene for which to execute grasps
    :param verbose: optional, whether to show GUI
    """
    def __init__(self, scene, verbose=False):
        super().__init__(verbose=verbose)
        self.scene = scene
        self.LIFTING_HEIGHT = 0.3
        self._reset_scene()

    @property
    def bullet_client(self):
        return self._p

    def _wait_for_user(self):
        if self.verbose:
            print('press enter to continue')
            input()

    def _reset_scene(self):
        self._reset(plane_and_gravity=True)
        _log.debug('setting scene...')
        self.add_scene(self.scene)

    def _check_collisions(self, gripper, target):
        """
        Checks collisions with environment objects, target object, and other moving scene objects.

        :param gripper: GripperBase object
        :param target: ObjectInstance target

        :return: GraspScores value
        """
        # checking collisions against environment objects (ground plane)
        _log.debug('checking collisions with environment bodies')
        for body_key, body_id in self._env_bodies.items():
            if self.are_in_collision(gripper.body_id, body_id):
                _log.debug(f'gripper in collision with {body_key} ({body_id})')
                return GraspScores.COLLISION_WITH_GROUND

        # checking collisions against target object
        _log.debug('checking collisions with target')
        if self.are_in_collision(gripper.body_id, target):
            _log.debug(f'gripper in collision with target ({target})')
            return GraspScores.COLLISION_WITH_TARGET

        # checking collisions with other scene objects
        _log.debug('checking collisions with other bodies')
        for body_key, body_id in self._moving_bodies.items():
            if body_id == self.look_up_body_id(target) or body_id == self.look_up_body_id(gripper.body_id):
                continue
            if self.are_in_collision(gripper.body_id, body_id):
                _log.debug(f'gripper in collision with {body_key} ({body_id})')
                return GraspScores.COLLISION_WITH_CLUTTER

        _log.debug('COLLISION CHECKS PASSED')
        return GraspScores.SUCCESS

    def _contact_established(self, gripper, target, min_contacts=2):
        """
        checks if there is contact established between the gripper and the target object

        :param gripper: gripper_module.GripperBase, the instantiated gripper
        :param target: ObjectInstance, to be grasped
        :param min_contacts: minimum number of fingers that need to be in contact (might make this dependent on the
                             number of fingers of the gripper in the future...)

        :return: bool, True if `min_contacts` finger of `gripper` are in contact with `target`
        """
        target_id = self.look_up_body_id(target)
        n_contacts = 0
        for finger_link in gripper.get_contact_link_ids():
            # finger_link could be a nested list if there are multiple links per finger
            # we don't care which link is in contact, as long as one link from each finger is in contact
            if not isinstance(finger_link, list):
                finger_link = [finger_link]
            for link in finger_link:
                if self.are_in_contact(gripper.body_id, link, target_id, -1):
                    n_contacts += 1
                    break
            if n_contacts >= min_contacts:
                return True
        return False

    def execute_grasp(self, gripper_type, grasp, target, gripper_scale=1.0, gripper_opening_width=1.0):
        """
        Executes a grasp in the scene.
        The simulator is already set up and will be reset after the grasp is attempted. This ensures that you can
        register step functions with the simulator (which will be removed after reset).

        :param gripper_type: A class that inherits from GripperBase
        :param grasp: A core.Grasp
        :param target: Must be one of the object instances in the scene which we attempt to grasp
        :param gripper_scale: float, Scaling factor for the gripper
        :param gripper_opening_width: float, Factor for scaling opening width, must be in [0.1, 1.0]

        :return: GraspScore
        """
        # create gripper, loading at pose, attaching dummy bot
        _log.debug('loading gripper...')
        gripper = gripper_type(self, gripper_scale)
        gripper.load(grasp.pose)
        gripper.set_open_scale(gripper_opening_width)
        robot = MountedGripper(self, gripper.body_id)

        result = self._check_collisions(gripper, target)
        if result != GraspScores.SUCCESS:
            self._wait_for_user()
            self._reset_scene()
            return result

        _log.debug(f'gripper joint states: {gripper.joint_positions}')
        _log.debug('closing gripper...')
        gripper.close()
        _log.debug('GRIPPER CLOSED')
        _log.debug(f'gripper joint states: {gripper.joint_positions}')

        _log.debug('checking contacts...')
        if not self._contact_established(gripper, target):
            _log.debug('no contact with target object established')
            self._wait_for_user()
            self._reset_scene()
            return GraspScores.NO_CONTACT_ESTABLISHED
        _log.debug('CONTACT ESTABLISHED')

        # start lifting
        _log.debug('lifting object...')
        curr_pos = robot.cartesian_pos()
        curr_pos[2] += 0.3
        robot.go_to_cartesian_pos(curr_pos)
        _log.debug('DONE LIFTING')

        # check again if object is still in contact
        _log.debug('checking contacts...')
        if not self._contact_established(gripper, target):
            _log.debug('no contact with target object established')
            self._wait_for_user()
            self._reset_scene()
            return GraspScores.SLIPPED_DURING_LIFTING

        _log.debug('CONTACT CONFIRMED')
        _log.debug('GRASP SUCCESSFUL')
        self._wait_for_user()
        self._reset_scene()
        return GraspScores.SUCCESS
