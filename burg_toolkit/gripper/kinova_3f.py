import numpy as np
from . import GripperBase


class Kinova3F(GripperBase):
    """
    The Kinova3F gripper has one finger which is opposing two other fingers.
    The x-axis of the grasp pose is pointing towards the single finger.

    todo: check alignment of visual/collision shapes
    """
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.207 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([0, 0, -np.pi/2])
        
        # define force and speed (grasping)
        self._force = 1000
        self._grasp_speed = 1.5

        finger1_joint_ids = [0, 1]
        finger2_joint_ids = [2, 3]
        finger3_joint_ids = [4, 5]
        self._finger_joint_ids = finger1_joint_ids + finger2_joint_ids + finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        self._contact_link_ids = [finger1_joint_ids, finger2_joint_ids, finger3_joint_ids]

        self._joint_lower = 0.2
        self._joint_upper = 1.3
        self._pos_sum = 1.4

    def load(self, grasp_pose):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        gripper_urdf = self.get_asset_path('kinova_3f/model.urdf')
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
        follower_pos = self._get_follower_joint_pos(driver_pos)

        self._bullet_client.resetJointState(self.body_id, self._driver_joint_id, targetValue=driver_pos)
        for joint_id, pos in zip(self._follower_joint_ids, follower_pos):
            self._bullet_client.resetJointState(self.body_id, joint_id, targetValue=pos)

    def _get_follower_joint_pos(self, driver_pos):
        pos = [
            self._pos_sum - driver_pos,
            driver_pos,
            self._pos_sum - driver_pos,
            driver_pos,
            self._pos_sum - driver_pos
        ]
        return pos

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=self._get_follower_joint_pos(pos),
            forces=[self._force]*len(self._follower_joint_ids),
            positionGains=[1.2]*len(self._follower_joint_ids)
        )
        return pos
    
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

    def get_vis_pts(self, open_scale):
        x = 0.03 + 0.4*0.05428 * np.sin(2*open_scale - 0.82865)
        return self._gripper_size * np.array([
            [-x, 0.03],
            [-x, -0.03],
            [x, 0]
        ])

    def get_contact_link_ids(self):
        return self._contact_link_ids
