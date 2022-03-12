import abc
import os


class GripperBase(abc.ABC):
    r"""Base class for all grippers.
    Any gripper should subclass this class.
    You have to implement the following class method:
        - load(): load URDF and return the body_id
        - configure(): configure the gripper (e.g. friction)
        - open(): open gripper
        - close(): close gripper
        - get_pos_offset(): return [x, y, z], the coordinate of the grasping center relative to the base
        - get_orn_offset(): the base orientation (in quaternion) when loading the gripper
        - get_vis_pts(open_scale): [(x0, y0), (x1, y1), (x2, y2s), ...], contact points for visualization (in world coordinate)
    """
    @abc.abstractmethod
    def __init__(self, simulator, gripper_size):
        self._body_id = None
        self._sim = simulator
        self._bullet_client = simulator.bullet_client
        self._gripper_size = gripper_size
        self.__mass = None

    @abc.abstractmethod
    def load(self, position, orientation, open_scale):
        pass

    @abc.abstractmethod
    def configure(self):
        pass

    @abc.abstractmethod
    def open(self, open_scale):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def get_pos_offset(self):
        pass

    @abc.abstractmethod
    def get_orn_offset(self):
        pass

    @abc.abstractmethod
    def get_contact_link_ids(self):
        pass

    @abc.abstractmethod
    def get_vis_pts(self, open_scale):
        pass

    @staticmethod
    def get_asset_path(gripper_fn):
        return os.path.join(os.path.dirname(__file__), 'assets/gripper', gripper_fn)

    @property
    def body_id(self):
        return self._body_id

    @property
    def mass(self):
        if self.__mass is None:
            assert self.body_id is not None, 'can only get mass after gripper is added to simulation'

            # sum up mass of base and all links
            mass = self._bullet_client.getDynamicsInfo(self.body_id, -1)[0]
            for i in range(self._bullet_client.getNumJoints(self.body_id)):
                mass += self._bullet_client.getDynamicsInfo(self.body_id, i)[0]
            self.__mass = mass

        return self.__mass
