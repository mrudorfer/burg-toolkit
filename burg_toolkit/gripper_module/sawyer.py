import numpy as np
from . import GripperBase


class GripperSawyer(GripperBase):
    """
    The gripper model is self-colliding and unable to close.
    We therefore disable self-collisions via pybullet, leading the fingertips to overshoot a little when closing.
    """
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.121 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        # driver joint will move from 0 (open) -> upper_limit - lower_limit (closed)
        # follower will move from upper_limit (open) -> 0 + lower_limit (closed)
        self._driver_joint_id = 0
        self._follower_joint_id = 1
        self._upper_limit = 0.044 * self._gripper_size
        self._lower_limit = 0.01 * self._gripper_size  # this is not reflected in urdf joint limits

        self._contact_joints = [0, 1]
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 0.1

    def load(self, grasp_pose):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        gripper_urdf = self.get_asset_path('sawyer/model.urdf')
        self._body_id = self._bullet_client.loadURDF(
            gripper_urdf,  # flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        self.set_color([0.5, 0.5, 0.5])
        self.configure_friction()
        self.configure_mass()
        self.set_open_scale(1.0)
        self._sim.register_step_func(self.step_constraints)

    def set_open_scale(self, open_scale):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        driver_pos, follower_pos = self._get_target_joint_pos(open_scale)
        self._bullet_client.resetJointState(self.body_id, self._driver_joint_id, driver_pos)
        self._bullet_client.resetJointState(self.body_id, self._follower_joint_id, follower_pos)

    def _get_target_joint_pos(self, open_scale):
        driver_pos = (self._upper_limit - self._lower_limit) * (1 - open_scale)
        follower_pos = self._upper_limit - driver_pos
        return [driver_pos, follower_pos]

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._follower_joint_id,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=self._upper_limit-pos,
            force=self._force,
            targetVelocity=2 * self._grasp_speed,
            positionGain=1.8
        )
        return pos
    
    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._driver_joint_id,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force,
        )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_vis_pts(self, open_scale):
        return np.array([
            [self._upper_limit * open_scale, 0],
            [-self._upper_limit * open_scale, 0]
        ])

    def get_contact_link_ids(self):
        return self._contact_joints
