import numpy as np
from .gripper_base import GripperBase


class GripperRG2(GripperBase):
    """ RG2 Gripper """
    def __init__(self, simulator, gripper_size=1.0):
        super().__init__(simulator, gripper_size)

        # offset the gripper to a down facing pose for grasping
        self._pos_offset = np.array([0, 0, 0.163 * self._gripper_size])  # offset from base to center of grasping
        self._orn_offset = self._bullet_client.getQuaternionFromEuler([np.pi, 0, 0])
        
        # define force and speed (grasping)
        self._force = 500
        self._grasp_speed = 0.5

        # define driver joint; the follower joints need to satisfy constraints when grasping
        self._driver_joint_id = 1
        self._driver_joint_lower = 0
        self._driver_joint_upper = 0.86
        self._follower_joint_ids = [0, 2, 3, 4, 5]
        self._follower_joint_sign = [1, -1, -1, -1, 1]

        self._contact_joint_ids = [2, 5]

    def load(self, position, orientation, open_scale=1.0):
        assert 0.1 <= open_scale <= 1.0, 'open_scale is out of range'
        gripper_urdf = self.get_asset_path('rg2/model.urdf')
        self._body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            # flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=position,
            baseOrientation=orientation
        )
        self.configure_friction()
        self.configure_mass()

        # set initial opening width
        driver_pos = open_scale*self._driver_joint_lower + (1-open_scale)*self._driver_joint_upper
        follower_pos = driver_pos * np.array(self._follower_joint_sign)
        self._bullet_client.resetJointState(self.body_id, self._driver_joint_id, targetValue=driver_pos)
        for i, pos in zip(self._follower_joint_ids, follower_pos):
            self._bullet_client.resetJointState(self.body_id, i, targetValue=pos)

        self._sim.register_step_func(self.step_constraints)

    def step_constraints(self):
        pos = self._bullet_client.getJointState(self.body_id, self._driver_joint_id)[0]
        targets = pos * np.array(self._follower_joint_sign)
        self._bullet_client.setJointMotorControlArray(
            self.body_id,
            self._follower_joint_ids,
            self._bullet_client.POSITION_CONTROL,
            targetPositions=targets,
            forces=[self._force] * len(self._follower_joint_ids),
            positionGains=[1.5] * len(self._follower_joint_ids)
        )
        return pos

    def open(self, mount_gripper_id, n_joints_before, open_scale):
        open_scale = np.clip(open_scale, 0.1, 1.0)
        target_pos = open_scale*self._driver_joint_lower + (1-open_scale)*self._driver_joint_upper  # recalculate scale because larger joint position corresponds to smaller open width
        self._bullet_client.setJointMotorControl2(
            mount_gripper_id,
            self._driver_joint_id+n_joints_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._force
        )
        for i in range(240 * 2):
            driver_pos = self.step_constraints(mount_gripper_id, n_joints_before)
            if np.abs(driver_pos - target_pos)<1e-5:
                break
            self._bullet_client.stepSimulation()
    
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
        width = 0.05 * np.sin(open_scale)
        return self._gripper_size * np.array([
            [-width, 0],
            [width, 0]
        ])

    def get_contact_link_ids(self):
        return self._contact_joint_ids
