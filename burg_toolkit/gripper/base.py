import abc
import os
import logging

import numpy as np
import open3d as o3d

from .. import util


_log = logging.getLogger(__name__)
ASSET_PATH = os.path.join(os.path.dirname(__file__), 'assets/')


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
        return os.path.join(ASSET_PATH, gripper_fn)

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


class ParallelJawGripper:
    """
    Represents a general parallel jawed gripper.
    Fingers are assumed to be cuboids with same width and height (the `finger_thickness`) and a specified
    `finger_length`.
    The inside of the fingers are at most `opening_width` apart.
    All values are given in meters.

    This also serves as a base class for all parallel jawed grippers. It provides a uniform interface as well as
    capabilities to produce a simplified mesh that can be used for visualization tasks.

    The meshes may be in arbitrary poses, before using they must be transformed using `tf_base_to_TCP` property.
    After applying this transform, the gripper is said to be TCP-oriented. This means that the TCP will be in origin,
    the gripper is approaching the grasp from the positive z-direction and the fingers will close in x-direction.

    :param finger_length: Length of the fingers.
    :param opening_width: Maximum distance between both fingers.
    :param finger_thickness: Side-lengths of the fingers.
    :param mesh: A mesh representation of the gripper.
    :param tf_base_to_TCP: (4, 4) np array with transformation matrix that transforms the grippers mesh and urdf
                           to the TCP-oriented pose.
    :param path_to_urdf: path to the URDF file of the gripper which can be used in simulation.
    """

    def __init__(self, finger_length=0.04, opening_width=0.08, finger_thickness=0.003, mesh=None, tf_base_to_TCP=None,
                 path_to_urdf=None):
        self._finger_length = finger_length
        self._opening_width = opening_width
        self._finger_thickness = finger_thickness
        self._mesh = mesh
        self._simplified_mesh = None
        if tf_base_to_TCP is None:
            tf_base_to_TCP = np.eye(4)
        self._tf_base_to_TCP = tf_base_to_TCP
        self._path_to_urdf = path_to_urdf

    def _create_simplified_mesh(self):
        """
        Creates a simple gripper mesh, consisting of the two fingers and a stem or bridge connecting them.

        :return: The created mesh as o3d.geometry.TriangleMesh.
        """
        # boxes spawn with left, front, bottom corner at 0, 0, 0
        finger1 = o3d.geometry.TriangleMesh.create_box(
            self.finger_thickness, self.finger_thickness, self.finger_length)
        finger2 = o3d.geometry.TriangleMesh(finger1)
        finger1.translate(np.array([-self.finger_thickness - self.opening_width/2, -self.finger_thickness/2, 0]))
        finger2.translate(np.array([self.opening_width/2, -self.finger_thickness/2, 0]))

        stem = o3d.geometry.TriangleMesh.create_box(
            self.opening_width + 2 * self.finger_thickness, self.finger_thickness, self.finger_thickness)
        stem.translate(np.array([-self.finger_thickness - self.opening_width / 2, -self.finger_thickness / 2,
                                 self.finger_length]))

        mesh = util.merge_o3d_triangle_meshes([finger1, finger2, stem])
        inv_tf = np.linalg.inv(self._tf_base_to_TCP)
        mesh.transform(inv_tf)
        mesh.compute_vertex_normals()
        return mesh

    @property
    def finger_thickness(self):
        return self._finger_thickness

    @property
    def finger_length(self):
        return self._finger_length

    @property
    def opening_width(self):
        return self._opening_width

    @property
    def mesh(self):
        """
        The mesh representation of this gripper. If gripper has none, a simplified mesh will be provided instead
        based on the dimensions of the gripper.
        """
        if self._mesh is None:
            return self.simplified_mesh
        return self._mesh

    @property
    def simplified_mesh(self):
        """
        Provides a simplified mesh based on the dimensions of the gripper.
        """
        if self._simplified_mesh is None:
            self._simplified_mesh = self._create_simplified_mesh()
        return self._simplified_mesh

    @property
    def tf_base_to_TCP(self):
        return self._tf_base_to_TCP

    @property
    def path_to_urdf(self):
        return self._path_to_urdf


class MountedGripper:
    """
    This class represents a gripper on a dummy mount. The mount can be moved linearly in x/y/z but not rotated.
    The corresponding gripper_type is instantiated in the given pose, where the pose is the grasp center (TCP).

    :param grasp_simulator: sim.GraspSimulator object that is using the MountedGripper
    :param gripper_type: GripperBase, class implementation of requested gripper type
    :param grasp_pose: numpy 4x4 pose, grasp center (TCP)
    :param gripper_scale: float, scale size of gripper
    :param opening_width: float, between 0.1 and 1.0, sets initial opening width
    """
    def __init__(self, grasp_simulator, gripper_type, grasp_pose, gripper_scale=1.0, opening_width=1.0):
        self._simulator = grasp_simulator

        # create gripper object, load and init
        self.gripper = gripper_type(grasp_simulator, gripper_scale)
        self.gripper.load(grasp_pose)
        self.gripper.set_open_scale(opening_width)

        # we want to place the mount at the base of the gripper (which differs from pos_gripper, orn_gripper!)
        pos_mount, orn_mount = self._simulator.bullet_client.getBasePositionAndOrientation(self.gripper.body_id)
        mount_urdf = os.path.join(ASSET_PATH, 'dummy_xyz_robot.urdf')
        self.mount_id, self.robot_joints = self._simulator.load_robot(mount_urdf, position=pos_mount,
                                                                      orientation=orn_mount, fixed_base=True)
        # attach gripper to mount
        self._simulator.bullet_client.createConstraint(
            self.mount_id, self.robot_joints['end_effector_link']['id'],
            self.gripper.body_id, -1,
            jointType=self._simulator.bullet_client.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )
        # self._simulator._inspect_body(self.gripper.body_id)  # use this for debugging of grippers

        # control all mount joints to stay at 0 with high force
        # makes sure gripper stays at the same position while closing
        mount_joints = [joint for joint in range(self._simulator.bullet_client.getNumJoints(self.mount_id))]
        self._simulator.bullet_client.setJointMotorControlArray(
            self.mount_id,
            mount_joints,
            self._simulator.bullet_client.POSITION_CONTROL,
            targetPositions=[0] * len(mount_joints),
            forces=[1000] * len(mount_joints)
        )

    def joint_pos(self):
        mount_joints = range(len(self.robot_joints))
        joint_states = self._simulator.bullet_client.getJointStates(self.mount_id, mount_joints)
        pos = []
        for joint in mount_joints:
            pos.append(joint_states[joint][0])
        return np.array(pos)

    def cartesian_pos(self):
        pos, *_ = self._simulator.bullet_client.getLinkState(
            self.mount_id,
            self.robot_joints['end_effector_link']['id']
        )
        return np.array(pos)

    def go_to_cartesian_pos(self, target_pos, timeout=5, tolerance=0.001):
        """
        Moves the mount (incl. gripper) to the desired cartesian target position.

        :param target_pos: [x, y, z] target position
        :param timeout: float, timeout in simulated seconds, will return even if position not attained
        :param tolerance: float, if Euclidean distance between target_pos and current position is smaller than this,
                          will return success

        :return: bool, True if position attained, False otherwise
        """
        _log.debug(f'go_to_cartesian_pos: moving to {target_pos}')

        end_time = self._simulator.simulated_seconds + timeout
        target_joint_pos = self._simulator.bullet_client.calculateInverseKinematics(
            self.mount_id,
            self.robot_joints['end_effector_link']['id'],
            list(target_pos)
        )

        pos_gain = 0.2
        for joint in range(len(self.robot_joints)):
            self._simulator.bullet_client.setJointMotorControl2(
                self.mount_id,
                joint,
                self._simulator.bullet_client.POSITION_CONTROL,
                targetPosition=target_joint_pos[joint],
                force=500,
                positionGain=pos_gain,
                maxVelocity=0.2
            )

        def point_reached(point):
            return np.linalg.norm(point - self.cartesian_pos()) < tolerance

        while not point_reached(target_pos) and self._simulator.simulated_seconds < end_time:
            self._simulator.step()

        # end of lifting operation, check if we actually arrived at the goal
        _log.debug(f'lifting took {self._simulator.simulated_seconds - (end_time - timeout):.3f} seconds')
        if point_reached(target_pos):
            _log.debug('goal position reached')
            return True
        _log.warning(f'go_to_pose reached timeout before attaining goal pose '
                     f'(d={np.linalg.norm(target_pos - self.cartesian_pos()):.3f})')
        _log.debug(f'current cartesian pos: {self.cartesian_pos()}')
        return False
