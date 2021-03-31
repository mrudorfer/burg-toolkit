from abc import ABC, abstractmethod

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from matplotlib import pyplot as plt

from . import util
from . import grasp


class GraspSimulatorBase(ABC):
    """
    Base class for all grasp simulators, offers some common methods for convenience.

    :param target_object: the object instance that shall be grasped
    :param gripper: the gripper object which will be used
    :param verbose: optional, indicates whether to show GUI and output debug info, defaults to False
    """
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
        if object_instance.object_type.urdf_fn is None:
            raise ValueError(f'object type {object_instance.object_type.identifier} does not provide an urdf file.')

        pos, quat = util.position_and_quaternion_from_tf(object_instance.pose, convention='pybullet')
        object_id = self._p.loadURDF(object_instance.object_type.urdf_fn,
                                     basePosition=pos, baseOrientation=quat,
                                     useFixedBase=int(fixed_base))

        self._p.changeDynamics(object_id, -1, lateralFriction=object_instance.object_type.friction_coeff)
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

        if object_id < 0:
            raise ValueError(f'could not add object {object_instance.object_type.identifier}. returned id is negative.')

        return object_id


class SceneGraspSimulator(GraspSimulatorBase):
    """
    SceneGraspSimulator: Simulates grasps in a particular scene.
    """
    def __init__(self, target_object, gripper, scene=None, verbose=False):
        super().__init__(target_object, gripper, verbose)

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

