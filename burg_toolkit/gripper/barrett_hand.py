import numpy as np
from . import GripperBase


class BarrettHand(GripperBase):
    """
    This gripper got additional arguments for rotation of the fingers within the palm... why not the others?
    """
    # def __init__(self, bullet_client, gripper_size, palm_joint, palm_joint_another=None):
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        # define palm configuration
        palm_joint = 0.2
        palm_joint_another = None
        self._finger_rotation1 = np.pi * palm_joint
        self._finger_rotation2 = np.pi * palm_joint_another if palm_joint_another is not None else np.pi * palm_joint

        self._pos_offset = np.array([0, 0, 0.181 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi / 2])
        
        # define force and speed (grasping)
        self._palm_force = 100
        self._force = 50
        self._grasp_speed = 0.4

        # define driver joint; the follower joints need to satisfy constraints when grasping
        finger1_joint_ids = [1, 2]  # thumb
        finger2_joint_ids = [4, 5]  # index finger
        finger3_joint_ids = [7, 8]  # middle finger
        self._finger_joint_ids = finger1_joint_ids+finger2_joint_ids+finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        self._contact_link_ids = [finger1_joint_ids, finger2_joint_ids, finger3_joint_ids]

        self._palm_joint_ids = [3, 6]
        self._joint_lower = 1
        self._joint_upper = 1.6

    def load(self, grasp_pose):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        gripper_urdf = self.get_asset_path('barrett_hand/model.urdf')
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
        driver_pos = open_scale * self._joint_lower + (1 - open_scale) * self._joint_upper
        follower_pos = self._get_follower_pos(driver_pos)

        self._bullet_client.resetJointState(self.body_id, self._driver_joint_id, targetValue=driver_pos)
        for joint_id, pos in zip(self._follower_joint_ids, follower_pos):
            self._bullet_client.resetJointState(self.body_id, joint_id, targetValue=pos)

        # also set finger rotations
        for i, pos in zip(self._palm_joint_ids, [self._finger_rotation1, self._finger_rotation2]):
            self._bullet_client.resetJointState(self.body_id, i, targetValue=pos)

    @staticmethod
    def _get_follower_pos(driver_pos):
        return [0.32-0.2*driver_pos, driver_pos, 0.32-0.2*driver_pos, driver_pos, 0.32-0.2*driver_pos]

    def step_constraints(self):
        # rotate finger2 and finger3
        # for palm_joint_id, palm_joint_pos in zip(self._palm_joint_ids, [self._finger_rotation1, self._finger_rotation2]):
        #     self._bullet_client.setJointMotorControl2(
        #         self.body_id,
        #         palm_joint_id,
        #         self._bullet_client.POSITION_CONTROL,
        #         targetPosition=palm_joint_pos,
        #         force=self._palm_force,
        #         positionGain=1,
        #         targetVelocity=0,
        #         maxVelocity=0.1
        #     )
        self._bullet_client.setJointMotorControlArray(
                self.body_id,
                self._palm_joint_ids,
                self._bullet_client.POSITION_CONTROL,
                targetPositions=[self._finger_rotation1, self._finger_rotation2],
                forces=[self._palm_force] * 2,
                positionGains=[1] * 2,
                # targetVelocities=[0.0] * 2
            )

        # follow finger joints
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        #for joint_id, joint_pos in zip(self._follower_joint_ids, self._get_follower_pos(pos)):
        #    self._bullet_client.setJointMotorControl2(
        #        self.body_id,
        #        joint_id,
        #        self._bullet_client.POSITION_CONTROL,
        #        targetPosition=joint_pos,
        #        force=self._force,
        #        positionGain=1.2,
        #        maxVelocity=self._grasp_speed * 2
        #    )
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=self._get_follower_pos(pos),
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids),
        )
        pass

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            maxVelocity=self._grasp_speed*2,
            force=self._force
        )
        # self._bullet_client.setJointMotorControl2(
        #     self.body_id,
        #     self._driver_joint_id,
        #     self._bullet_client.POSITION_CONTROL,
        #     targetPosition=self._joint_upper,
        #     force=self._force,
        #     maxVelocity=0.2
        # )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_vis_pts(self, open_scale):
        k = 0.023 + 0.0481 * np.sin(2*open_scale - 0.8455)
        m = 0.025
        return self._gripper_size * np.array([
            [-k * np.cos(self._finger_rotation1), -m - k * np.sin(self._finger_rotation1)],
            [-k * np.cos(self._finger_rotation2), m + k * np.sin(self._finger_rotation2)],
            [k, 0]
        ])

    def get_contact_link_ids(self):
        return self._contact_link_ids
