import abc
import os


class GripperBase(abc.ABC):
    r"""Base class for all grippers.
    Any gripper should subclass this class.
    You have to implement the following class method:
        - load(): load URDF and configure gripper according to open_scale
        - close(): close gripper
        - get_pos_offset(): return [x, y, z], the coordinate of the grasping center relative to the base
        - get_orn_offset(): the base orientation (in quaternion) when loading the gripper
        - get_contact_link_ids(): list of contacting finger links. if multiple links per finger, then use nested lists
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

    def is_loaded(self):
        """
        returns True, if the gripper instance has been added to a simulation
        """
        return self.body_id is not None

    @property
    def mass(self):
        if self.__mass is None:
            assert self.is_loaded(), 'can only get mass after gripper is loaded in simulation'

            # sum up mass of base and all links
            mass = self._bullet_client.getDynamicsInfo(self.body_id, -1)[0]
            for i in range(self.num_joints):
                mass += self._bullet_client.getDynamicsInfo(self.body_id, i)[0]
            self.__mass = mass

        return self.__mass

    def set_color(self, color):
        """
        changes the visual appearance of the gripper links by adding color
        """
        assert self.is_loaded(), 'can only set color after gripper is loaded in simulation'
        if len(color) == 3:
            color.append(1)

        for link_id in range(-1, self.num_joints):
            self._bullet_client.changeVisualShape(self.body_id, link_id, rgbaColor=color)

    @property
    def num_joints(self):
        assert self.is_loaded(), 'can only determine number of joints after gripper is loaded in simulation'
        return self._bullet_client.getNumJoints(self.body_id)

    def configure_friction(self, lateral_friction=1.0, spinning_friction=1.0, rolling_friction=0.0001,
                           friction_anchor=True):
        """
        configures the friction properties of all gripper joints
        """
        assert self.is_loaded(), 'can only configure friction after gripper is loaded in simulation'
        for i in range(self.num_joints):
            self._bullet_client.changeDynamics(self.body_id, i,
                                               lateralFriction=lateral_friction, spinningFriction=spinning_friction,
                                               rollingFriction=rolling_friction, frictionAnchor=friction_anchor)

    def configure_mass(self, base_mass=0.4, combined_finger_mass=0.1):
        """
        configures the mass of the gripper such that all grippers have a uniform mass and can hence be controlled
        uniformly
        """
        assert self.is_loaded()
        self._bullet_client.changeDynamics(self.body_id, -1, mass=base_mass)
        for i in range(self.num_joints):
            self._bullet_client.changeDynamics(self.body_id, i, mass=combined_finger_mass/self.num_joints)

    def joint_positions(self):
        joint_states = self._bullet_client.getJointStates(self.body_id, range(self.num_joints))
        pos = []
        for joint in range(self.num_joints):
            pos.append(joint_states[joint][0])
        return pos
