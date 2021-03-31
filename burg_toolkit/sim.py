import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from . import util


# gets a scene with some objects
# gets the target object
# gets a grasp (set)
# gets a gripper

# provides some score for the grasp

class GraspSimulator:
    """
    GraspSimulator: Provides capabilities to simulate grasps for a particular gripper and determine scores.
    """
    def __init__(self, object_instance, gripper, scene=None, verbose=False):
        self.verbose = verbose

        # using bullet client makes sure we can connect to multiple servers in parallel
        self._p = bullet_client.BulletClient(connection_mode=p.GUI)
        # options="--mp4=moviename.mp4" (records movie, requires ffmpeg)
        # p.connect(p.DIRECT)
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)

        # setup the scene
        self._bg_objects_ids = []
        self._objects_ids = []
        self.setup_scene(scene)
        input()

        # maybe do this in some other function
        pass

    def add_object(self, object_instance, fixed_base=False):
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

    def setup_scene(self, scene):
        """
        Loads all bg_objects and objects into the scene.
        Note that bg_objects will be immovable (with a fixed base).

        :param scene: .scene.Scene object
        """
        self._bg_objects_ids = []
        self._objects_ids = []

        for bg_obj in scene.bg_objects:
            self._bg_objects_ids.append(self.add_object(bg_obj, fixed_base=True))

        for obj in scene.objects:
            o_id = self.add_object(obj, fixed_base=False)
            self._objects_ids.append(o_id)

    def simulate_grasp(self, grasp):
        # performs a simulation and determines a score
        score = 0
        return score

