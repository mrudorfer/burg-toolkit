import logging
import os

import numpy as np
import open3d as o3d

from . import util
from . import gripper_module


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


class Robotiq2F85(ParallelJawGripper):
    """
    An implementation of the Robotiq 2F-85 gripper.
    Note that this gripper does not perform purely parallel grasps - in fact the finger pads will move towards
    the object (up to 14mm) as the grasp closing is executed.
    If the object contact is made below the equilibrium line, the grasp will be an encompassing grasp rather
    than a parallel grasp.
    These behaviours will be considered only during simulation, for all other concerns we treat it as simple
    parallel jaw gripper (with open jaw).
    """
    def __init__(self):
        super().__init__(
            finger_length=0.0375,  # length of the pads, but position is changing during the grasp
            opening_width=0.085,
            finger_thickness=0.022,  # width of the pads
        )
        self._tf_base_to_TCP = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.15],  # in manual, TCP is at 0.171, which is with finger tips closed
            [0.0, 0.0, 0.0, 1.0]
        ])
        # todo: if we actually make a package distribution, this might not work anymore
        self._path_to_urdf = os.path.join(os.path.dirname(__file__), '../data/gripper/robotiq-2f-85/robotiq_2f_85.urdf')
        print(f'path to urdf evaluated to {self._path_to_urdf}')


class MountedGripper:
    """
    This class represents a gripper on a dummy mount. The mount can be moved linearly in x/y/z but not rotated.
    The corresponding gripper_type is instantiated in the given pose, where the pose is the grasp center (TCP).

    :param grasp_simulator: GraspSimulator object that is using the MountedGripper
    :param gripper_type: string, requested gripper type
    :param grasp_pose: numpy 4x4 pose, grasp center (TCP)
    """
    def __init__(self, grasp_simulator, gripper_type, grasp_pose, gripper_scale=1.0):
        self._simulator = grasp_simulator

        # create gripper object
        self.gripper = gripper_module.all_grippers[gripper_type](grasp_simulator, gripper_scale)

        # compute gripper pose, load and init gripper
        tf2hand = util.tf_from_pos_quat(self.gripper.get_pos_offset(), self.gripper.get_orn_offset(),
                                        convention='pybullet')
        gripper_pose = grasp_pose @ tf2hand
        pos_gripper, orn_gripper = util.position_and_quaternion_from_tf(gripper_pose, convention='pybullet')
        self.gripper_id = self.gripper.load(pos_gripper, orn_gripper)
        self.gripper.configure()

        # we want to place the mount at the base of the gripper (which differs from pos_gripper, orn_gripper!)
        pos_mount, orn_mount = self._simulator.bullet_client.getBasePositionAndOrientation(self.gripper_id)
        mount_urdf = os.path.join(os.path.dirname(__file__), '../data/gripper/dummy_xyz_robot.urdf')
        self.mount_id, self.robot_joints = self._simulator.load_robot(mount_urdf, position=pos_mount,
                                                                      orientation=orn_mount, fixed_base=True)
        # attach gripper to mount
        self._simulator.bullet_client.createConstraint(
            self.mount_id, self.robot_joints['end_effector_link']['id'],
            self.gripper_id, -1,
            jointType=self._simulator.bullet_client.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )

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
        print(joint_states)
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

    def go_to_cartesian_pos(self, pos, timeout=10):
        """
        Attempts to move the mounted gripper to pos.

        :param pos: [x, y, z], target position
        :param timeout: max seconds to simulate

        :return: bool, True if position attained, False if timeout reached
        """
        logging.debug(f'go_to_cartesian_pos: moving to {pos}')

        target_joint_pos = self._simulator.bullet_client.calculateInverseKinematics(
            self.mount_id,
            self.robot_joints['end_effector_link']['id'],
            list(pos)
        )
        logging.debug(f'found target joint positions: {target_joint_pos}')

        self._simulator.bullet_client.setJointMotorControlArray(
            self.mount_id,
            range(len(self.robot_joints)),
            self._simulator.bullet_client.POSITION_CONTROL,
            targetPositions=target_joint_pos,
            targetVelocities=[0.01] * len(self.robot_joints),
            forces=[np.minimum(item['max_force'], 50) for _, item in self.robot_joints.items()]
        )

        def goal_reached(tolerance=0.001):
            return np.linalg.norm(pos - self.cartesian_pos()) < tolerance

        end_time = self._simulator.simulated_seconds + timeout
        while not goal_reached() and self._simulator.simulated_seconds <= end_time:
            self._simulator.step()

        if goal_reached():
            logging.debug('goal position reached')
            return True
        logging.debug('reached timeout before attaining goal position')
        logging.debug(f'current cartesian pos: {self.cartesian_pos()}')
        return False
