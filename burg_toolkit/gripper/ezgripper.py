import numpy as np
from . import GripperBase


class GripperEZGripper(GripperBase):
    """ EZgripper

    There are a couple of different grasping strategies for the EZGripper, as each finger has two independent joints.
    We could use only the main joint, however, this leads to a large occupied volume during grasping, potentially
    colliding with environment objects.
    Instead, we control the finger such that the outer links are always staying vertical while the inner links are
    closing. Furthermore, we prevent fully opening the gripper (which would mean the upper finger links are horizontal).
    This behaviour is governed by the self._joint_lower variable.
    """
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.1805 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi / 2, np.pi / 2, 0])
        
        # define force and speed (grasping)
        self._force = 500
        self._grasp_speed = 1.5

        finger1_joint_ids = [0, 1]
        finger2_joint_ids = [2, 3]
        self._finger_joint_ids = [finger1_joint_ids, finger2_joint_ids]
        self._driver_joint_id = 0
        self._follower_joint_ids = [1, 2, 3]
        self._joint_ids = [self._driver_joint_id] + self._follower_joint_ids

        self._joint_lower = 0.6  # opened
        self._joint_upper = 1.9  # closed

    def _get_joint_positions(self, open_scale):
        driver_pos = open_scale*self._joint_lower + (1-open_scale)*self._joint_upper
        return [driver_pos, self._joint_upper-driver_pos, driver_pos, self._joint_upper-driver_pos]

    def load(self, grasp_pose):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        gripper_urdf = self.get_asset_path('ezgripper/model.urdf')
        self._body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        self.configure_friction()
        self.configure_mass()
        self.set_open_scale(1.0)
        self._sim.register_step_func(self.step_constraints)

    def set_open_scale(self, open_scale):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        init_joint_pos = self._get_joint_positions(open_scale)
        for i, pos in enumerate(init_joint_pos):
            self._bullet_client.resetJointState(self.body_id, i, targetValue=pos)

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[self._joint_upper-pos, pos, self._joint_upper-pos],
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return pos

    def open(self, open_scale=1.0):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'

        # larger joint position corresponds to smaller open width
        driver_pos = self._get_joint_positions(open_scale)[0]
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=driver_pos,
            force=self._force
        )
        self._sim.step(seconds=2)

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_contact_link_ids(self):
        # this is a nested list with two joints per finger
        return self._finger_joint_ids

    def get_vis_pts(self, open_scale):
        width = 0.09 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])
