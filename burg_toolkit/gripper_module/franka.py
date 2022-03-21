import numpy as np
from . import GripperBase


class GripperFranka(GripperBase):
    def __init__(self, simulator, gripper_size=1.0):
        r""" Initialization of Franka 2 finger gripper
        specific args for Franka:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.115 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, np.pi/2])

        self._finger_open_distance = 0.04 * self._gripper_size
        self._moving_joint_ids = [0, 1]
        
        # define force and speed (grasping)
        self._force = 100
        self._grasp_speed = 0.1

    def load(self, grasp_pose, open_scale=1.0):
        position, orientation = self._get_pos_orn_from_grasp_pose(grasp_pose)
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        gripper_urdf = self.get_asset_path('franka/model.urdf')
        self._body_id = self._bullet_client.loadURDF(
            gripper_urdf, flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        self.set_color([0.5, 0.5, 0.5])
        self.configure_friction()
        self.configure_mass()

        # open gripper
        for joint_id in self._moving_joint_ids:
            self._bullet_client.resetJointState(self.body_id, joint_id, self._finger_open_distance * open_scale)

        self._sim.register_step_func(self.step_constraints)

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._moving_joint_ids[0])[0]
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._moving_joint_ids[1],
            self._bullet_client.POSITION_CONTROL,
            targetPosition=pos,
            force=self._force,
            targetVelocity=2 * self._grasp_speed,
            positionGain=1.8
        )
        return pos

    def open(self, open_scale=1.0):
        joint_ids = [i for i in self._moving_joint_ids]
        target_states = [self._finger_open_distance * open_scale, self._finger_open_distance * open_scale]
        
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_states, 
            forces=[self._force] * len(joint_ids)
        )
        self._sim.step(seconds=2)

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.body_id,
            self._moving_joint_ids[0],
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=-self._grasp_speed,
            force=self._force,
        )
        self._sim.step(seconds=2)

    def get_pos_offset(self):
        return self._pos_offset

    def get_orn_offset(self):
        return self._orn_offset

    def get_contact_link_ids(self):
        return self._moving_joint_ids

    def get_vis_pts(self, open_scale):
        return np.array([
            [self._finger_open_distance * open_scale, 0],
            [-self._finger_open_distance * open_scale, 0]
        ])
