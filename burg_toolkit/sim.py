from abc import ABC, abstractmethod
import time

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from matplotlib import pyplot as plt

from . import util
from . import grasp


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
    def _retrieve(cls, score, item):
        if score in cls._s2c_dict.keys():
            return cls._s2c_dict[score][item]
        else:
            raise ValueError(f'score value {score} is unknown. only have {cls._s2c_dict.keys()}')

    @classmethod
    def score2color(cls, score):
        return cls._retrieve(score, 0)

    @classmethod
    def score2description(cls, score):
        return cls._retrieve(score, 1)

    @classmethod
    def score2color_name(cls, score):
        return cls._retrieve(score, 2)


class GraspSimulatorBase(ABC):
    """
    Base class for all grasp simulators, offers some common methods for convenience.

    :param target_object: the object instance that shall be grasped
    :param gripper: the gripper object which will be used
    :param verbose: optional, indicates whether to show GUI and output debug info, defaults to False
    """

    JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    SPINNING_FRICTION = 0.1
    ROLLING_FRICTION = 0.0001

    def __init__(self, target_object, gripper, verbose=False):
        self.target_object = target_object
        self.gripper = gripper
        self.verbose = verbose
        self._color_idx = 0
        self.color_map = plt.get_cmap('tab20')
        self.dt = 1./240.

        self._target_object_id = None
        self._gripper_id = None

        # connect using bullet client makes sure we can connect to multiple servers in parallel
        # options="--mp4=moviename.mp4" (records movie, requires ffmpeg)
        self._p = bullet_client.BulletClient(connection_mode=p.GUI if verbose else p.DIRECT)

    def _reset(self):
        """
        This method resets the simulation to the starting point. Shall be used to clean up after a simulation run.
        """
        self._p.resetSimulation()
        self._color_idx = 0

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

    def dismiss(self):
        """
        This method shall be called when the simulation is not needed anymore as it cleans up the object.
        """
        self._p.disconnect()

    def _add_object(self, object_instance, fixed_base=False):
        """
        Adds an object to the simulator.

        :param object_instance: scene.ObjectInstance (with type and pose)
        :param fixed_base: if True, the object is immovable (defaults to False)

        :return: object id if object could be added, else raises an Error
        """
        if object_instance.object_type.urdf_fn is None:
            raise ValueError(f'object type {object_instance.object_type.identifier} does not provide an urdf file.')

        pos, quat = util.position_and_quaternion_from_tf(object_instance.pose, convention='pybullet')
        object_id = self._p.loadURDF(object_instance.object_type.urdf_fn,
                                     basePosition=pos, baseOrientation=quat,
                                     useFixedBase=int(fixed_base))

        if object_id < 0:
            raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

        self._p.changeDynamics(object_id, -1, lateralFriction=object_instance.object_type.friction_coeff,
                               spinningFriction=self.SPINNING_FRICTION, rollingFriction=self.ROLLING_FRICTION,
                               restitution=object_instance.object_type.restitution_coeff)
        # todo: add coefficient of restitution, potentially check other dynamics params as well

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

    def _inspect_body(self, body_id):
        """
        prints out some debug info for the given object
        """
        print('****')
        print(f'inspecting body id {body_id}')
        print(f'body info: {self._p.getBodyInfo(body_id)}')
        num_joints = self._p.getNumJoints(body_id)
        print(f'num joints: {num_joints}')
        if num_joints > 0:
            for i in range(num_joints):
                info = self._p.getJointInfo(body_id, i)
                print(f'joint {i}:')
                print(f'- joint id {info[0]}, name {info[1].decode("utf-8")}')
                print(f'- joint type {self.JOINT_TYPES[info[2]]}')
                print(f'- link aabb {self._p.getAABB(body_id, i)}')
                print(f'- joint damping {info[6]} and friction {info[7]}')
                print(f'- lower limit {info[8]}, upper limit {info[9]}')
                print(f'- max force {info[10]}, max velocity {info[11]}')
                print(f'- link name {info[12]}, parent link index {info[16]}')

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
        self._plane_id = None

    def _prepare(self):
        # todo: maybe it is better to not reload all objects but instead to just move them to the correct pose
        #       i think there are even some snapshot functions we could try to use
        if self._with_plane_and_gravity:
            self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._plane_id = self._p.loadURDF("plane.urdf")
            self._p.setGravity(0, 0, -9.81)

        self._target_object_id = self._add_object(self.target_object)

        if self._with_plane_and_gravity:
            if self._are_in_collision(self._target_object_id, self._plane_id):
                print('WARNING: target object and plane are in collision. this should not be the case.')

    def _simulate_grasp(self, g):
        print('************** physics engine parameters **************')
        print(self._p.getPhysicsEngineParameters())
        print('*******************************************************')

        # PHASE 0: PLACING GRIPPER IN GRASP POSE
        # we have TCP grasp representation, hence need to transform gripper to TCP-oriented pose as well
        tf = np.matmul(g.pose, self.gripper.tf_base_to_TCP)
        pos, quat = util.position_and_quaternion_from_tf(tf, convention='pybullet')
        self._gripper_id = self._p.loadURDF(self.gripper.path_to_urdf, basePosition=pos, baseOrientation=quat)
        self._p.changeDynamics(self._gripper_id, -1, mass=0)  # make this object static

        if self.verbose:
            self._inspect_body(self._target_object_id)
            self._inspect_body(self._gripper_id)
            print(f'aabb: {self._p.getAABB(self._gripper_id)}')

        # PHASE 1: CHECK GRIPPER COLLISIONS
        # checking collisions against ground plane and target object
        if self._with_plane_and_gravity:
            if self._are_in_collision(self._gripper_id, self._plane_id):
                if self.verbose:
                    print('gripper and plane are in collision')
                return GraspScores.COLLISION_WITH_GROUND
        if self._are_in_collision(self._gripper_id, self._target_object_id):
            if self.verbose:
                print('gripper and target object are in collision')
            return GraspScores.COLLISION_WITH_TARGET

        if self.verbose:
            print('COLLISION CHECKS PASSED... press enter to continue')
            input()

        # PHASE 2: CLOSING FINGER TIPS
        # first set friction coefficients
        for i in range(p.getNumJoints(self._gripper_id)):
            p.changeDynamics(self._gripper_id, i, lateralFriction=1.0, spinningFriction=self.SPINNING_FRICTION,
                             rollingFriction=self.ROLLING_FRICTION, frictionAnchor=True)
        # now we need to link the finger tips together, so they mimic their movement
        # this variant is by https://github.com/lzylucy/graspGripper
        # using link 1 as master with velocity control, and all other links use position control to follow 1
        self._p.setJointMotorControl2(self._gripper_id, 1, p.VELOCITY_CONTROL, targetVelocity=1, force=50)
        for i in range(int(4e2)):
            self._p.stepSimulation()
            gripper_joint_positions = np.array([p.getJointState(self._gripper_id, i)[
                                                    0] for i in range(p.getNumJoints(self._gripper_id))])
            p.setJointMotorControlArray(
                self._gripper_id, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                [
                    gripper_joint_positions[1], -gripper_joint_positions[1],
                    -gripper_joint_positions[1], gripper_joint_positions[1],
                    gripper_joint_positions[1]
                ],
                positionGains=np.ones(5)
            )
            time.sleep(0.01)


        self._p.stepSimulation()


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

