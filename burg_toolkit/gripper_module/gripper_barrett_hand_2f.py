import numpy as np
from .gripper_base import GripperBase


class GripperBarrettHand2F(GripperBase):
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        self._finger_rotation = np.pi / 2
        self._pos_offset = np.array([0, 0, 0.18 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, 0])
        
        # define force and speed (grasping)
        self._force = 100
        # self._grasp_speed = 2
        self._grasp_speed = 0.4

        # define driver joint; the follower joints need to satisfy constraints when grasping
        finger2_joint_ids = [1, 2]  # index finger
        finger3_joint_ids = [4, 5]  # middle finger
        self._finger_joint_ids = finger2_joint_ids+finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        self._palm_joint_ids = [0, 3]
        self._joint_lower = 1.2
        self._joint_upper = 1.8

        self._contact_joint_ids = [2, 5]

    def load(self, position, orientation, open_scale=1.0):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        gripper_urdf = self.get_asset_path('barrett_hand_2f/model.urdf')
        self._body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        # self.set_color([0.5, 0.5, 0.5])
        self.configure_friction()
        # self.configure_mass()

        # open gripper according to open_scale
        driver_pos = open_scale * self._joint_lower + (1 - open_scale) * self._joint_upper
        follower_pos = self._get_follower_pos(driver_pos)
        self._bullet_client.resetJointState(self.body_id, self._driver_joint_id, targetValue=driver_pos)
        for i, pos in zip(self._follower_joint_ids, follower_pos):
            self._bullet_client.resetJointState(self.body_id, i, targetValue=pos)
        # also set finger rotations
        for i in self._palm_joint_ids:
            self._bullet_client.resetJointState(self.body_id, i, targetValue=self._finger_rotation)

        self._sim.register_step_func(self.step_constraints)

    @staticmethod
    def _get_follower_pos(driver_pos):
        return [-0.25+0.25*driver_pos, driver_pos, -0.25+0.25*driver_pos]

    def step_constraints(self):
        # rotate finger2 and finger3
        self._bullet_client.setJointMotorControlArray(
                self.body_id,
                self._palm_joint_ids,
                self._bullet_client.POSITION_CONTROL,
                targetPositions=[self._finger_rotation] * 2,
                forces=[self._force] * 2,
                positionGains=[1] * 2
            )
        driver_pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=self._get_follower_pos(driver_pos),
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return driver_pos

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force
        )
        self._sim.step(seconds=2)
        # break if driver_pos > self._joint_upper+0.1:

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_vis_pts(self, open_scale):
        return self._gripper_size * np.array([
            [- (0.075*open_scale), 0],
            [0.075*open_scale, 0],
        ])

    def get_contact_link_ids(self):
        return self._contact_joint_ids
