import abc
import os

from .. import util


class GripperBase(abc.ABC):
    """
    Base class for all grippers that can be used in simulation.
    After initialisation, call load() to actually load the gripper at a certain pose in the sim environment.

    :param simulator: burg.sim.GraspSimulator instance, as we need access to bullet_client and other sim funcs
    :param gripper_size: float, scale factor for loading the gripper model
    """
    @abc.abstractmethod
    def __init__(self, simulator, gripper_size):
        self._body_id = None
        self._sim = simulator
        self._bullet_client = simulator.bullet_client
        self._gripper_size = gripper_size
        self.__mass = None

    @abc.abstractmethod
    def load(self, grasp_pose):
        """
        Loads the gripper in the simulation, configures friction and mass, and registers constraints with the simulator.
        Only after loading, other methods can return reasonable output (e.g. body_id etc.) - check status with
        is_loaded(). Per default, the gripper will be loaded fully opened. Use set_open_scale() to adjust for that.

        :param grasp_pose: pose of the grasping center
        """
        pass

    @abc.abstractmethod
    def set_open_scale(self, open_scale):
        """
        Sets the open scale of the gripper, i.e. resets joints to match the scaling factor.
        Should be called after the gripper is loaded, but before the simulation starts.

        :param open_scale: float, between 0.1 and 1.0, determines how much the gripper is opened
        """
        pass

    def is_loaded(self):
        """
        returns True, if the gripper instance has been added to a simulation
        """
        return self.body_id is not None

    def _get_pos_orn_from_grasp_pose(self, grasp_pose):
        """
        Given the pose of a grasping center, will compute the position and orientation for placing the gripper in
        simulation.

        :param grasp_pose: 4x4 transformation matrix

        :return: pos [x, y, z], orn [x, y, z, w]
        """
        tf2hand = util.tf_from_pos_quat(self.get_pos_offset(), self.get_orn_offset(), convention='pybullet')
        gripper_pose = grasp_pose @ tf2hand
        pos_gripper, orn_gripper = util.position_and_quaternion_from_tf(gripper_pose, convention='pybullet')
        return pos_gripper, orn_gripper

    @abc.abstractmethod
    def close(self):
        """
        Uses the simulator to execute the grasp and close the gripper. Shall return after the gripper is closed.
        """
        pass

    @abc.abstractmethod
    def get_pos_offset(self):
        """
        Returns the position offset [x, y, z] of the grasping center relative to the base
        """
        pass

    @abc.abstractmethod
    def get_orn_offset(self):
        """
        Returns the orientation [x, y, z, w] of the grasping center relative to the base
        """
        pass

    @abc.abstractmethod
    def get_contact_link_ids(self):
        """
        Returns the link/joint ids which are expected to be in contact with the object when grasped.
        List can be nested, e.g. [link1, [link3, link4]] means that contact with link1 is required, and contact with
        either link3 or link4 is required in order to successfully grasp the object.
        """
        pass

    @abc.abstractmethod
    def get_vis_pts(self, open_scale):
        """
        This method is currently unused by us.

        :return: [(x0, y0), (x1, y1), (x2, y2s), ...], contact points for visualization (in world coordinate)
        """
        pass

    @staticmethod
    def get_asset_path(gripper_fn):
        """
        :param gripper_fn: Filename within assets/gripper folder.

        :return: Returns full path to the requested file.
        """
        return os.path.join(os.path.dirname(__file__), 'assets/', gripper_fn)

    @property
    def body_id(self):
        """
        Gives the body id, will be None if the gripper has not been loaded in simulation.
        """
        return self._body_id

    @property
    def mass(self):
        """
        :return: The summed mass of all links of this gripper.
        """
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

    @property
    def joint_positions(self):
        joint_states = self._bullet_client.getJointStates(self.body_id, range(self.num_joints))
        pos = []
        for joint in range(self.num_joints):
            pos.append(joint_states[joint][0])
        return pos
