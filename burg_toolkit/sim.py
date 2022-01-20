from abc import ABC, abstractmethod
import time
import os

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from matplotlib import pyplot as plt

from . import util
from . import grasp


class SimulatorBase(ABC):
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
        self.TIME_SLEEP = self.dt * 3  # for visualization
        self.SPINNING_FRICTION = 0.1
        self.SPINNING_FRICTION = 0.003  # this setting is used in AdaGrasp
        self.ROLLING_FRICTION = 0.0001
        self.MIN_OBJ_MASS = 0.05  # small masses will be replaced by this (could tune a bit more, in combo with solver)
        self.JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

        self._color_idx = 0
        self.color_map = plt.get_cmap('tab20')
        self._body_ids = {}  # use dictionary for body id's.. will be cleaned automatically after reset
        self._coms = {}  # dictionary to store center of mass belonging to the body id's
        self._simulated_steps = 0

        self._p = None
        self._reset()

    @property
    def _simulated_seconds(self):
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
            self._body_ids = {}
            self._coms = {}
            self._color_idx = 0
            self._simulated_steps = 0

        self._p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=self.SOLVER_STEPS)
        if self.verbose:
            self._p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=0, cameraPitch=-30,
                                               cameraTargetPosition=[0, 0, 0])
        if plane_and_gravity:
            self._load_plane_and_gravity()

    def _load_plane_and_gravity(self, plane_id='plane'):
        """
        Loads a plane and sets gravity.

        :param plane_id: string, the body ID to be used for the plane (in self._body_ids dict)
        """
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._body_ids[plane_id] = self._p.loadURDF("plane.urdf")
        self._p.setGravity(0, 0, -9.81)
        self._p.changeDynamics(self._body_ids[plane_id], -1, lateralFriction=1.0)

    def _step(self, n=1, seconds=None):
        """
        Steps the simulation for n steps if seconds is None.
        If seconds provided, will simulate the equivalent number of steps.
        """
        if seconds is not None:
            n = int(seconds / self.dt)
        for i in range(n):
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

    def _add_object(self, object_instance, fixed_base=False):
        """
        Adds an object to the simulator.

        :param object_instance: core.ObjectInstance (with type and pose)
        :param fixed_base: if True, the object is immovable (defaults to False)

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
            print(f'added object {object_instance.object_type.identifier}')

            print(f'object properties: \n'
                  f'mass, lateral_friction, local inertia diagonal, local inertia pos, '
                  f'local inertia orn, restitution, rolling friction, spinning friction, contact damping,'
                  f'contact stiffness, body type (1 rigid, 2 multi-body, 3 soft), collision margin\n'
                  f'{self._p.getDynamicsInfo(object_id, -1)}')

        return object_id

    def _get_body_pose(self, body_id, convert2burg=False):
        """
        Returns the base position and orientation of the body with respect to center of mass frame as used by
        pybullet. If `convert2burg` is True, it will be transformed back to normal frame of reference.

        :param body_id: either the pybullet body id as int, or a string used in the self._body_ids dict.
        :param convert2burg: If set to True, frame of reference is world instead of center of mass.

        :return: (4, 4) transformation matrix describing the pose of the object
        """
        if isinstance(body_id, str):
            body_id = self._body_ids[body_id]
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

    def _inspect_body(self, body_id):
        """
        prints out some debug info for the given object
        """
        print('****')
        print(f'inspecting body id {body_id}')
        print(f'body info: {self._p.getBodyInfo(body_id)}')
        num_joints = self._p.getNumJoints(body_id)
        print(f'num joints: {num_joints}')
        for i in range(num_joints):
            print(f'joint {i}:')
            [print(f'\t{key}: {val}') for key, val in self._get_joint_info(body_id, i).items()]

    def _print_joint_positions(self, body_id):
        """
        prints out the positions of all joints of the body

        :param body_id: id of the body
        """
        num_joints = self._p.getNumJoints(body_id)
        print(f'getting {num_joints} joint positions of body {body_id}, {self._p.getBodyInfo(body_id)}')
        joint_states = self._p.getJointStates(body_id, list(range(num_joints)))
        for joint_state in joint_states:
            print(f'\t{joint_state[0]}')

    def _are_in_collision(self, body_id_1, body_id_2):
        """
        checks if two bodies are in collision with each other.

        :return: bool, True if the two bodies are in collision
        """
        max_distance = 0.01  # 1cm for now, might want to choose a more reasonable value
        points = self._p.getClosestPoints(body_id_1, body_id_2, max_distance)

        if self.verbose:
            print(f'checking collision between {self._p.getBodyInfo(body_id_1)} and {self._p.getBodyInfo(body_id_2)}')
            print(f'found {len(points)} points')

        n_colliding_points = 0
        distances = []
        for point in points:
            distance = point[8]
            distances.append(distance)
            if distance < 0:
                n_colliding_points += 1

        if self.verbose:
            print(f'of which {n_colliding_points} have a negative distance (i.e. are in collision)')
            print(f'distances are: {distances}')

        return n_colliding_points > 0

    def _are_in_contact(self, body_id_1, link_id_1, body_id_2, link_id_2):
        """
        checks if the links of two bodies are in contact.
        """
        contacts = self._p.getContactPoints(body_id_1, body_id_2, link_id_1, link_id_2)
        return len(contacts) > 0


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


class GraspSimulatorBase(SimulatorBase):
    """
    Base class for all grasp simulators, offers some common methods for convenience.

    :param target_object: the object instance that shall be grasped
    :param gripper: the gripper object which will be used
    :param object_urdf_dir: directory where to look for object urdf files, initialised as data/tmp
    :param verbose: optional, indicates whether to show GUI and output debug info, defaults to False
    """
    def __init__(self, target_object, gripper, object_urdf_dir=None, verbose=False):
        super().__init__(verbose=verbose)
        self.target_object = target_object
        self.gripper = gripper
        if object_urdf_dir is None:
            object_urdf_dir = os.path.join(os.path.dirname(__file__), '../data/tmp/')
        self.object_urdf_dir = object_urdf_dir

        self.DUMMY_ROBOT_URDF = os.path.join(os.path.dirname(__file__), '../data/gripper/dummy_robot.urdf')
        self.DUMMY_XYZ_ROBOT_URDF = os.path.join(os.path.dirname(__file__), '../data/gripper/dummy_xyz_robot.urdf')

    @abstractmethod
    def _prepare(self):
        """
        This method prepares everything for the simulation (except the particular grasp which is to be executed).
        """
        pass

    @abstractmethod
    def _simulate_grasp(self, g):
        """
        This method will simulate the given grasp and return a corresponding score.

        :param g: grasp.Grasp

        :return: score (int)
        """
        pass

    def simulate_grasp_set(self, grasp_set):
        """
        This method runs the simulation for all grasps given in the grasp set and determines a score.

        :param grasp_set: grasp.GraspSet, can also be a single grasp.Grasp

        :return: (n,) scores as int
        """
        if type(grasp_set) is grasp.Grasp:
            grasp_set = grasp_set.as_grasp_set()

        scores = np.zeros(len(grasp_set))

        for i, g in enumerate(grasp_set):
            self._prepare()
            if self.verbose:
                print('press enter to start simulation')
                input()
            scores[i] = self._simulate_grasp(g)
            if self.verbose:
                print(f'this grasp got score {scores[i]}. press enter to proceed with next grasp.')
                input()
            self._reset()
        return scores

    def _add_object(self, object_instance, fixed_base=False):
        """
        Adds an object to the simulator.

        :param object_instance: scene.ObjectInstance (with type and pose)
        :param fixed_base: if True, the object is immovable (defaults to False)

        :return: object id if object could be added, else raises an Error
        """
        urdf_fn = os.path.join(self.object_urdf_dir, object_instance.object_type.identifier + '.urdf')
        if not os.path.exists(urdf_fn):
            raise ValueError(f'could not find urdf file for object type {object_instance.object_type.identifier}.' +
                             f'expected it at {urdf_fn}.')

        pos, quat = util.position_and_quaternion_from_tf(object_instance.pose, convention='pybullet')
        object_id = self._p.loadURDF(urdf_fn,
                                     basePosition=pos, baseOrientation=quat,
                                     useFixedBase=int(fixed_base))

        if object_id < 0:
            raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

        # dynamics don't work for very small masses, so let's increase mass if necessary
        mass = np.max([self.target_object.object_type.mass, self.MIN_OBJ_MASS])
        self._p.changeDynamics(object_id, -1, lateralFriction=object_instance.object_type.friction_coeff,
                               spinningFriction=self.SPINNING_FRICTION, rollingFriction=self.ROLLING_FRICTION,
                               restitution=object_instance.object_type.restitution_coeff, mass=mass)

        self._p.changeVisualShape(object_id, -1, rgbaColor=self.color_map(self._color_idx))
        self._color_idx = (self._color_idx + 1) % self.color_map.N

        if self.verbose:
            print(f'added object {object_instance.object_type.identifier}')

            print(f'object properties: \n'
                  f'mass, lateral_friction, local inertia diagonal, local inertia pos, '
                  f'local inertia orn, restitution, rolling friction, spinning friction, contact damping,'
                  f'contact stiffness, body type (1 rigid, 2 multi-body, 3 soft), collision margin\n'
                  f'{self._p.getDynamicsInfo(object_id, -1)}')

        return object_id

    def _load_robot(self, urdf_file, position=None, orientation=None, fixed_base=False, friction=None):
        """
        Loads a robot and creates a data structure to access all of the robots info as well.

        :param urdf_file: string containing the path to the urdf file.
        :param position: (3,) base position; optional, defaults to [0, 0, 0]
        :param orientation: (x, y, z, w) base orientation; optional, defaults to [0, 0, 0, 1]
        :param fixed_base: whether or not to use fixed base; optional, defaults to False
        :param friction: lateral friction to be set for each joint (will also set default spinning and rolling friction
                         as well as friction anchor); optional, no settings will be made if not provided
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
            if friction is not None:
                self._p.changeDynamics(
                    body_id,
                    joint_idx,
                    lateralFriction=friction,
                    spinningFriction=self.SPINNING_FRICTION,
                    rollingFriction=self.ROLLING_FRICTION,
                    frictionAnchor=False  # todo: not sure whether we really want the friction anchor
                    # documentation says:
                    # enable or disable a friction anchor: friction drift correction
                    # (disabled by default, unless set in the URDF contact section)
                )
            joint_info = self._get_joint_info(body_id, joint_idx)
            joint_infos[joint_info['link_name']] = joint_info

        return body_id, joint_infos


class SingleObjectGraspSimulator(GraspSimulatorBase):
    """
    Simulates a grasp of a single object instance.

    :param target_object: scene.ObjectInstance object
    :param gripper: gripper object that shall execute the grasp
    :param verbose: show GUI and debug output if True
    :param with_ground_plane_and_gravity: if True, xy-plane will be created and gravity will be taken into account.
    """
    def __init__(self, target_object, gripper, verbose=False, with_ground_plane_and_gravity=True):
        super().__init__(target_object=target_object, gripper=gripper, verbose=verbose)

        self._with_plane_and_gravity = with_ground_plane_and_gravity
        self.LIFTING_HEIGHT = 0.1  # 10 cm

    def _prepare(self):
        # todo: maybe it is better to not reload all objects but instead to just move them to the correct pose
        #       i think there are even some snapshot functions we could try to use
        if self._with_plane_and_gravity:
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._body_ids['plane'] = self._p.loadURDF("plane.urdf")
            self._p.setGravity(0, 0, -9.81)

        self._body_ids['target_object'] = self._add_object(self.target_object)

        if self._with_plane_and_gravity:
            if self._are_in_collision(self._body_ids['target_object'], self._body_ids['plane']):
                print('WARNING: target object and plane are in collision. this should not be the case.')

    def _control_follower_joints(self):
        master_id = 1
        # todo: control followers depending on gripper object
        master_joint_state = self._p.getJointState(self._body_ids['gripper'], master_id)[0]
        self._p.setJointMotorControlArray(
            self._body_ids['gripper'], [6, 3, 8, 5, 10], p.POSITION_CONTROL,
            [master_joint_state, -master_joint_state, -master_joint_state, master_joint_state, master_joint_state],
            positionGains=np.ones(5)
        )

    def _both_fingers_touch_object(self, link_finger_1, link_finger_2):
        contact_1 = self._are_in_contact(
            self._body_ids['gripper'], link_finger_1,
            self._body_ids['target_object'], -1)

        if not contact_1:
            return False

        contact_2 = self._are_in_contact(
            self._body_ids['gripper'], link_finger_2,
            self._body_ids['target_object'], -1)

        return contact_2

    def _simulate_grasp(self, g):
        print('************** physics engine parameters **************')
        print(self._p.getPhysicsEngineParameters())
        print('*******************************************************')

        ########################################
        # PHASE 0: PLACING GRIPPER IN GRASP POSE
        # we have TCP grasp representation, hence need to transform gripper to TCP-oriented pose as well
        tf = np.matmul(g.pose, self.gripper.tf_base_to_TCP)
        grasp_pos, grasp_quat = util.position_and_quaternion_from_tf(g.pose, convention='pybullet')
        gripper_pos, gripper_quat = util.position_and_quaternion_from_tf(tf, convention='pybullet')
        # load a dummy robot which we can move everywhere and connect the gripper to it
        self._body_ids['robot'], robot_joints = self._load_robot(
            self.DUMMY_XYZ_ROBOT_URDF, position=gripper_pos, orientation=gripper_quat, fixed_base=True
        )
        self._body_ids['gripper'], gripper_joints = self._load_robot(
            self.gripper.path_to_urdf, position=gripper_pos, orientation=gripper_quat, fixed_base=False, friction=1.0
        )

        self._p.createConstraint(
            self._body_ids['robot'], robot_joints['end_effector_link']['id'],
            self._body_ids['gripper'], -1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )

        if self.verbose:
            print('all objects loaded')
            self._inspect_body(self._body_ids['target_object'])
            self._inspect_body(self._body_ids['robot'])
            self._inspect_body(self._body_ids['gripper'])

        ###################################
        # PHASE 1: CHECK GRIPPER COLLISIONS
        # checking collisions against ground plane and target object
        if self._with_plane_and_gravity:
            if self._are_in_collision(self._body_ids['gripper'], self._body_ids['plane']):
                if self.verbose:
                    print('gripper and plane are in collision')
                return GraspScores.COLLISION_WITH_GROUND
        if self._are_in_collision(self._body_ids['gripper'], self._body_ids['target_object']):
            if self.verbose:
                print('gripper and target object are in collision')
            return GraspScores.COLLISION_WITH_TARGET

        if self.verbose:
            print('COLLISION CHECKS PASSED... press enter to continue')
            input()

        ##############################
        # PHASE 2: CLOSING FINGER TIPS
        # now we need to link the finger tips together, so they mimic their movement
        # this variant is by https://github.com/lzylucy/graspGripper
        # using link 1 as master with velocity control, and all other links use position control to follow 1
        self._p.setJointMotorControl2(self._body_ids['gripper'], 1, p.VELOCITY_CONTROL, targetVelocity=1, force=50)
        seconds = 1.0
        for i in range(int(seconds/self.dt)):
            self._control_follower_joints()
            self._p.stepSimulation()

            if self.verbose:
                time.sleep(self.TIME_SLEEP)

            # checking contact
            if self._both_fingers_touch_object(
                    gripper_joints['robotiq_2f_85_left_pad']['id'],
                    gripper_joints['robotiq_2f_85_right_pad']['id']):
                if self.verbose:
                    print('CONTACT ESTABLISHED')
                    print('proceeding to hold grasp for 0.25 seconds')
                break

        for i in range(int(0.25/self.dt)):
            self._control_follower_joints()
            self._p.stepSimulation()

        if not self._both_fingers_touch_object(
                gripper_joints['robotiq_2f_85_left_pad']['id'],
                gripper_joints['robotiq_2f_85_right_pad']['id']):
            if self.verbose:
                print('gripper does not touch object, grasp FAILED')
            return GraspScores.NO_CONTACT_ESTABLISHED

        #########################
        # PHASE 3: LIFTING OBJECT
        if self.verbose:
            print('OBJECT GRASPED... press enter to lift it')
            input()

        # ok so our dummy robot can be controlled with prismatic (linear) joints in xyz
        # however, the base of the dummy robot has some arbitrary pose in space, so it's not just controlling z
        target_movement = [0, 0, self.LIFTING_HEIGHT]  # x, y, z

        pos, *_ = self._p.getLinkState(
            self._body_ids['robot'],
            robot_joints['end_effector_link']['id']
        )
        target_position = np.array(target_movement) + np.array(pos)

        target_joint_pos = self._p.calculateInverseKinematics(
            self._body_ids['robot'],
            robot_joints['end_effector_link']['id'],
            list(target_position)
        )

        if self.verbose:
            print('target position:', list(target_position))
            print('target joint pos:', target_joint_pos)

        # setup position control with target joint values
        self._p.setJointMotorControlArray(
            self._body_ids['robot'],
            jointIndices=range(len(robot_joints)),
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_joint_pos,
            targetVelocities=[0.01 for _ in robot_joints.keys()],
            forces=[np.minimum(item['max_force'], 80) for _, item in robot_joints.items()]
        )

        n_steps = 0
        timeout = 1.5  # seconds
        max_steps = timeout / self.dt
        while (np.sum(np.abs(target_position - np.array(pos))) > 0.01) and (n_steps < max_steps):
            self._control_follower_joints()
            self._p.stepSimulation()
            pos, *_ = self._p.getLinkState(
                self._body_ids['robot'],
                robot_joints['end_effector_link']['id']
            )
            n_steps += 1

            if self.verbose:
                time.sleep(self.TIME_SLEEP)
                print('***')
                print(f'step {n_steps} / {max_steps}')
                print(pos[2], 'current z of robot end-effector link')
                print(target_position[2], 'target z pos of that link')
                print(f'abs difference of z: {np.abs(target_position[2] - pos[2])}; ')
                print(f'abs diff sum total: {np.sum(np.abs(target_position - np.array(pos)))}')

        if self.verbose:
            print(f'LIFTING done, required {n_steps*self.dt} of max {max_steps*self.dt} seconds')

        # lifting finished, check if object still attached
        if not self._both_fingers_touch_object(
                gripper_joints['robotiq_2f_85_left_pad']['id'],
                gripper_joints['robotiq_2f_85_right_pad']['id']):
            if self.verbose:
                print('gripper does not touch object anymore, grasp FAILED')
            return GraspScores.SLIPPED_DURING_LIFTING

        if self.verbose:
            print('object grasped and lifted successfully')
        return GraspScores.SUCCESS


class SceneGraspSimulator(GraspSimulatorBase):
    """
    SceneGraspSimulator: Simulates grasps in a particular scene.
    """
    def __init__(self, target_object, gripper, scene=None, verbose=False):
        super().__init__(target_object=target_object, gripper=gripper, verbose=verbose)

        self.scene = scene
        self._bg_objects_ids = []
        self._objects_ids = []

    def _prepare(self):
        # self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.setGravity(0, 0, -9.81)
        self._bg_objects_ids = []
        self._objects_ids = []

        # load background objects with fixed base
        for bg_obj in self.scene.bg_objects:
            self._bg_objects_ids.append(self._add_object(bg_obj, fixed_base=True))

        # foreground objects will be movable
        for obj in self.scene.objects:
            self._objects_ids.append(self._add_object(obj, fixed_base=False))

    def _simulate_grasp(self, g):
        # performs a simulation and determines a score
        score = 0
        return score

