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
    def __init__(self, object_instance, gripper, scene=None):
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

    def setup_scene(self, scene):
        """
        Loads all bg_objects and objects into the scene.
        Note that bg_objects will be immovable (with a fixed base).

        :param scene: .scene.Scene object
        """
        self._bg_objects_ids = []
        self._objects_ids = []
        if scene is None:
            return

        for obj in [*scene.bg_objects, *scene.objects]:
            if obj.object_type.urdf_fn is None:
                raise ValueError(f'provided objects do not have associated urdf files.')

        for bg_obj in scene.bg_objects:
            pos, quat = util.position_and_quaternion_from_tf(bg_obj.pose, convention='pybullet')
            id = self._p.loadURDF(bg_obj.object_type.urdf_fn, basePosition=pos, baseOrientation=quat, useFixedBase=1)
            self._bg_objects_ids.append(id)

        for obj in scene.objects:
            pos, quat = util.position_and_quaternion_from_tf(obj.pose, convention='pybullet')
            id = self._p.loadURDF(obj.object_type.urdf_fn, basePosition=pos, baseOrientation=quat)
            # todo: we may want to set friction coefficient here! (currently it's not included in urdf)
            self._objects_ids.append(id)

    def simulate_grasp(self, grasp):
        # performs a simulation and determines a score
        score = 0
        return score

