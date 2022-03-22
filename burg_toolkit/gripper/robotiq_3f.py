import numpy as np
from . import GripperBase


class Robotiq3F(GripperBase):
    """
    Not exactly sure why we use 0.8 times the gripper size here. Is the URDF model wrong?
    """
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, 0.8 * gripper_size)

        self._pos_offset = np.array([0, 0, 0.163 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([-np.pi/2, 0, 0])
        
        # define force and speed (grasping)
        self._palm_force = 100
        self._force = 50
        self._grasp_speed = 0.1

        self._palm_joint_ids = [0, 4]  # the joints that link the palm and fingers. not moved in grasping
        finger1_joint_ids = [1, 2, 3]
        finger2_joint_ids = [5, 6, 7]
        finger3_joint_ids = [9, 10, 11]
        self._finger_joint_ids = finger1_joint_ids+finger2_joint_ids+finger3_joint_ids
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        self._contact_joint_ids = [finger1_joint_ids, finger2_joint_ids, finger3_joint_ids]

        # joint limits
        self._joint_lower = 0.1
        self._joint_upper = 0.4

    def load(self, grasp_pose):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        gripper_urdf = self.get_asset_path('robotiq_3f/model.urdf')
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

    @staticmethod
    def _get_follower_pos(driver_pos):
        return [driver_pos, driver_pos-0.5,
                driver_pos, driver_pos, driver_pos-0.5,
                driver_pos, driver_pos, driver_pos-0.5]

    def step_constraints(self):
        # fix 2 fingers in 0 position
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._palm_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[0.0] * len(self._palm_joint_ids),
            forces=[self._palm_force] * len(self._palm_joint_ids),
            positionGains=[1.6] * len(self._palm_joint_ids)
        )

        # follow with finger joints
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=self._get_follower_pos(pos),
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
            maxVelocity=self._grasp_speed*2,
            force=self._force
        )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_vis_pts(self, open_scale):
        x = 0.0455 + 0.072 * np.sin(2*open_scale - 1.1418)
        return self._gripper_size * np.array([
            [-x, 0.04],
            [-x, -0.04],
            [x, 0]
        ])

    def get_contact_link_ids(self):
        return self._contact_joint_ids
