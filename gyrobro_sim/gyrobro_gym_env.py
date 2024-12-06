"""
This file implements the functionalities of a Gyrobro Gym environment using pybullet physical engine.
"""
import copy
import numpy as np
import os
import inspect
from transforms3d.quaternions import quat2mat
import vbcontrolpy as vb

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# print("!!!!!!!!!!!!!")
# print(currentdir)
# print(parentdir)
# print("!!!!!!!!!!!!!")

import time
import random
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pybullet
from pybullet_utils import bullet_client as bc
import pybullet_data

import math

INIT_POSITION = [0, 0, .3]
WHEELS = ["LEFT", "RIGHT"]
MOTOR_NAMES = ["left", "right"]
WHEEL_ID = [2, 3]
BASE_LINK_ID = 0
AXIS_LINK_ID = 1

NUM_SUBSTEPS = 5
NUM_MOTORS = 2
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 720
RENDER_WIDTH = 960
K_SCALE = 2

MOTOR_VELOCITY_MAX = 12
MOTOR_TORQUE_MAX = 110
BASE_ORIENTATION_MAX = 1.57
MOTOR_TIMESTEP = 0.0001


class GyrobroBulletEnv(gym.Env):
    """
    The gym environment for Walkerbro.

    It simulates the locomotion of a walkerbro, a quadruped robot. The state space
    include base orientation, the angles, velocities and torques for all the motors and the action
    space is the desired motor angle for each motor. The reward function is based
    on how fast robot can move without twitches and side deviations.

    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                sim_freq=500.0,
                urdf_root = "/home/yoggi/gyrobro_control/src/gyrobro_sim/urdf/gyrobro.urdf",
                hard_reset=False,
                render=True,
                real_time=True,
                on_rack=False,
                debug_mode = False,
                floating_camera = False,
                initial_pitch = -0.01,

                # Motors
                motor_model_enabled=False, # True - voltage control, False - torque control
                motor_velocity_limit=np.inf,

                # RL stuff
                max_timesteps = np.inf,
                action_repeat=1,
                rew_scale = 1,
                distance_limit=float("inf"),
                observation_noise_stdev=[0.0, 0.0, 0.0, 0.0, 0.0], # [wheel_angle, pitch, d_pitch, yaw, d_yaw]
                normalization=False, # todo

                ext_disturbance_enabled=False,
                ):

        self._time_step = 1/sim_freq
        self._urdf_root = urdf_root
        self._num_bullet_solver_iterations = sim_freq*NUM_SUBSTEPS
        self._action_repeat = action_repeat
        self._motor_velocity_limit = motor_velocity_limit
        self._is_render = render
        self._real_time = real_time
        self._floating_camera = floating_camera
        self._distance_limit = distance_limit
        self._observation_noise_stdev = observation_noise_stdev
        self._on_rack = on_rack
        self._max_timesteps = max_timesteps
        self._rew_scale = rew_scale
        self._hard_reset = True
        self._debug_mode = debug_mode
        self._normalization = normalization
        self._ext_disturbance_enabled = ext_disturbance_enabled
        self._initial_pitch = initial_pitch
        self._motor_model_enabled = motor_model_enabled

        self._lateral_friction = 0.6
        self._spinning_friction = 0.06

        self._action_bound = 1

        self._motor_direction = 1

        self._cam_dist = 0.8
        self._cam_yaw = 0
        self._cam_pitch = 0
        self._last_frame_time = 0.0

        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI, options="--width=2280 --height=1500")
        else:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.action_dim = NUM_MOTORS
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-np.float32(action_high), np.float32(action_high), dtype=np.float32)
        self.action = np.array([0]*self.action_dim, dtype=np.float32)
        self.action_prev = np.array([0]*self.action_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=np.float32(-1), high=np.float32(1), shape=(self.action_dim,), dtype=np.float32)

        if self._motor_model_enabled:
            self.mot_R = 0.4
            self.mot_U_max = 48
            self.mot_L = 0.0000344
            self.mot_J = 0.00001
            self.mot_Kt = 0.42
            self.mot_Kw = 0.37
            self.mot_b = 0.005
            self.mot_pole_pairs = 14
            self.mot_gear_ratio = 1
            self.mot_Z = 24
            self.mot_Tk = [0, 0, 0, 0]
            self.mot_k = [1, 2, 3, 4]
            self.mot_alpha = [0, 0, 0, 0]

            self.motor_left = vb.dynamics.pmsm(MOTOR_TIMESTEP)
            self.motor_left.set_physical_params(self.mot_R,
                                      self.mot_U_max,
                                      self.mot_L,
                                      self.mot_J,
                                      self.mot_Kt,
                                      self.mot_Kw,
                                      self.mot_b,
                                      self.mot_pole_pairs,
                                      self.mot_gear_ratio)
            self.motor_left.set_cogging_torque_params(self.mot_Tk,
                                            self.mot_k,
                                            self.mot_alpha,
                                            self.mot_Z)

            self.motor_right = vb.dynamics.pmsm(MOTOR_TIMESTEP)
            self.motor_right.set_physical_params(self.mot_R,
                                      self.mot_U_max,
                                      self.mot_L,
                                      self.mot_J,
                                      self.mot_Kt,
                                      self.mot_Kw,
                                      self.mot_b,
                                      self.mot_pole_pairs,
                                      self.mot_gear_ratio)
            self.motor_right.set_cogging_torque_params(self.mot_Tk,
                                            self.mot_k,
                                            self.mot_alpha,
                                            self.mot_Z)

            self.mot_state_l = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.mot_state_r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            self.num_mot_iterations = int(self._time_step / MOTOR_TIMESTEP)

        self.reset()

        self._hard_reset = hard_reset

        if self._is_render == True:
            self.view = True
            self.reset_id = self._pybullet_client.addUserDebugParameter("Reset",1,0,1)
            self.view_id = self._pybullet_client.addUserDebugParameter("View",1,0,1)
            self.reset_clicked_prev = self._pybullet_client.readUserDebugParameter(self.reset_id)
            self.reset_clicked = self._pybullet_client.readUserDebugParameter(self.reset_id)
            self.view_clicked_prev = self._pybullet_client.readUserDebugParameter(self.view_id)
            self.view_clicked = self._pybullet_client.readUserDebugParameter(self.view_id)

        print("Inited")


    def reset(self):
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SHADOWS, 0)
            self._pybullet_client.setGravity(0, 0, -9.81)

            planeShape = self._pybullet_client.createCollisionShape(shapeType=self._pybullet_client.GEOM_PLANE)
            ground_id = self._pybullet_client.createMultiBody(0, planeShape)
            self._pybullet_client.resetBasePositionAndOrientation(ground_id, [0, 0, 0], [0, 0, 0, 1])
            self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=self._lateral_friction, spinningFriction=self._spinning_friction)

            self._reset_robot(reload_urdf=True)
        else:
            self._reset_robot(reload_urdf=False)

        self._env_step_counter = 0

        self.mot_state_l = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.mot_state_r = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # self.obs = self._noisy_observation()
        # self.obs = self.norm_obs(self.obs)
        self.obs = self._get_noisy_observation()
        return self.obs

    def _reset_robot(self, reload_urdf=True):
        # addr = "/home/yoggi/gyrobro_control/src/gyrobro_sim/urdf/gyrobro.urdf" #currentdir +
        addr = self._urdf_root

        INIT_ORIENTATION = self._pybullet_client.getQuaternionFromEuler([0, self._initial_pitch, 0])
        if reload_urdf:
            self.gyrobro = self._pybullet_client.loadURDF(
                addr,
                INIT_POSITION,
                INIT_ORIENTATION)

            if self._on_rack:
                self._pybullet_client.createConstraint(self.gyrobro, BASE_LINK_ID, -1, -1,
                                                    self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                                    [0, 0, 0], INIT_POSITION, [0,0,0,1], INIT_ORIENTATION)
            self._RecordMassInfoFromURDF(self.gyrobro)
            self._ResetPose(self.gyrobro)

        else:
            self._pybullet_client.resetBasePositionAndOrientation(self.gyrobro, INIT_POSITION,
                                                                    INIT_ORIENTATION)
            self._pybullet_client.resetBaseVelocity(self.gyrobro, [0, 0, 0], [0, 0, 0])
            self._ResetPose(self.gyrobro)


    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
        action: A list of desired motor angles for eight motors.

        Returns:
        observations: The angles, velocities and torques of all motors.
        reward: The reward for the current state-action pair.
        done: Whether the episode has ended.
        info: A dictionary that stores diagnostic information.

        Raises:
        ValueError: The action dimension is not the same as the number of motors.
        ValueError: The magnitude of actions is out of bounds.
        """

        # Sleep, otherwise the computation takes less time than real time,
        if self._real_time == True:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # reset button
        if self._is_render == True:
            self.reset_clicked = self._pybullet_client.readUserDebugParameter(self.reset_id)
            if self.reset_clicked != self.reset_clicked_prev:
                self._reset_robot(reload_urdf=False)
            self.reset_clicked_prev = self.reset_clicked
            # view button
            self.view_clicked = self._pybullet_client.readUserDebugParameter(self.view_id)
            if self.view_clicked != self.view_clicked_prev:
                self.view = not self.view
            self.view_clicked_prev = self.view_clicked

            if self.view == True:
                base_pos, _ = self.get_base_position_and_orientation(self.gyrobro)
                self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, base_pos)

        if self._ext_disturbance_enabled:
            self._apply_force()

        if self._motor_model_enabled == False:
            self.action = action
        else:
            # state = self.get_real_observation()
            # t0 = time.time()
            for _ in range(self.num_mot_iterations):
                self.mot_state_l_ = self.motor_left.nonlinear_dynamics(self.mot_state_l, [0, action[0]])
                self.mot_state_r_ = self.motor_right.nonlinear_dynamics(self.mot_state_r, [0, action[1]])

                self.mot_state_l = self.mot_state_l_
                self.mot_state_r = self.mot_state_r_

            self.action = [4.0/self.mot_Kt*self.mot_state_l[1],
                           4.0/self.mot_Kt*self.mot_state_r[1]]

            # t1 = time.time()
            # print(t1-t0)

        for _ in range(self._action_repeat):
            self._apply_action(self.action)
            self._pybullet_client.stepSimulation()

        self._env_step_counter += 1
        reward = self._rew_scale*self._reward()
        done = self._termination()
        self.obs = self._get_noisy_observation()

        self.action_prev[:] = self.action[:]
        return self.obs, reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, _ = self.get_base_position_and_orientation(self.gyrobro)
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                    aspect=float(RENDER_WIDTH) /
                                                                    RENDER_HEIGHT,
                                                                    nearVal=0.1,
                                                                    farVal=100.0)
        (_, _, px, _,
        _) = self._pybullet_client.getCameraImage(width=RENDER_WIDTH,
                                                height=RENDER_HEIGHT,
                                                viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix,
                                                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array


    def seed(self, seed=None):
        pass

    def close(self):
        """
        Close the environment and disconnect physical engine.
        It is recommended to use this function at the end of your program.
        """
        super().close()
        self._pybullet_client.disconnect()


    def is_fallen(self):
        pass


    def _termination(self):
        pass


    def _reward(self):
        return 5

    def get_real_observation(self):
        """Get the observations of walkerbro.

        It includes the angles, velocities, torques and the orientation of the base.

        Returns:
        The observation list
        observation[0:1] are motor angles
        observation[2:3] are motor velocities
        observation[4:5] are base pitch and yaw
        observation[6:7] are base pitch and yaw angular velocities
        """
        observation = []
        self.motor_angles = self.get_motor_angles(self.gyrobro).tolist()
        self.motor_velocities = self.get_motor_velocities(self.gyrobro).tolist()
        self.base_orientation = list(self.get_base_orientation_euler(self.gyrobro))
        self.base_ang_vel = list(self.get_base_ang_vel(self.gyrobro))
        observation.extend(self.motor_angles)
        observation.extend(self.motor_velocities)

        observation.extend(self.base_orientation[1:3])
        observation.extend(self.base_ang_vel[1:3])
        return observation

    def _get_noisy_observation(self):
        obs = self.get_real_observation()
        obs[:2] += np.random.normal(loc=0.0, scale=self._observation_noise_stdev[0], size=2)
        obs[4] += float(np.random.normal(loc=0.0, scale=self._observation_noise_stdev[1], size=1)[0])
        obs[5] += float(np.random.normal(loc=0.0, scale=self._observation_noise_stdev[3], size=1)[0])
        obs[6] += float(np.random.normal(loc=0.0, scale=self._observation_noise_stdev[2], size=1)[0])
        obs[7] += float(np.random.normal(loc=0.0, scale=self._observation_noise_stdev[4], size=1)[0])

        return obs


    def _apply_action(self, command):
        # print(WHEEL_ID[0])

        self._SetMotorTorqueById(self.gyrobro, WHEEL_ID[0], float(command[0]))
        self._SetMotorTorqueById(self.gyrobro, WHEEL_ID[1], float(command[1]))

    def _SetMotorTorqueById(self, robot, motor_id, torque : float):
        self._pybullet_client.setJointMotorControl2(bodyIndex=robot,
                                                    jointIndex=motor_id,
                                                    controlMode=pybullet.TORQUE_CONTROL,
                                                    force=torque)

    def _RecordMassInfoFromURDF(self, robot):
        self._base_mass_urdf = self._pybullet_client.getDynamicsInfo(robot, BASE_LINK_ID)[0]
        self._axis_mass_urdf = self._pybullet_client.getDynamicsInfo(robot, AXIS_LINK_ID)[0]
        self._wheel_mass_urdf = self._pybullet_client.getDynamicsInfo(robot, WHEEL_ID[0])[0]
        # print("!!!!!!!!!!!!!!!!")
        # print(self._base_mass_urdf)
        # print(self._axis_mass_urdf)
        # print(self._wheel_mass_urdf)
        # print("!!!!!!!!!!!!!!!!")

    def _ResetPose(self, robot):
        """Reset the pose of the walkerbro.
        constraint: Whether to add a constraint at the joints of two feet.
        """
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # num_joints=self._pybullet_client.getNumJoints(robot)
        # for i in range(num_joints):
        #     joint_info = self._pybullet_client.getJointInfo(robot, i)
        #     print(joint_info)

        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        self._pybullet_client.resetJointState(robot,
                                          WHEEL_ID[0],
                                          0,
                                          targetVelocity=0)
        self._pybullet_client.resetJointState(robot,
                                          WHEEL_ID[1],
                                          0,
                                          targetVelocity=0)

        self._pybullet_client.setJointMotorControl2(robot, WHEEL_ID[0], pybullet.VELOCITY_CONTROL, force=0)
        self._pybullet_client.setJointMotorControl2(robot, WHEEL_ID[1], pybullet.VELOCITY_CONTROL, force=0)

        self._pybullet_client.changeDynamics(robot, WHEEL_ID[0], linearDamping=0, angularDamping=0.005)
        self._pybullet_client.changeDynamics(robot, WHEEL_ID[1], linearDamping=0, angularDamping=0.005)

        self._pybullet_client.setJointMotorControl2(robot, WHEEL_ID[0], pybullet.TORQUE_CONTROL, force=0)
        self._pybullet_client.setJointMotorControl2(robot, WHEEL_ID[1], pybullet.TORQUE_CONTROL, force=0)


    # Sensor Data
    def get_motor_angles(self, robot):
        """Get the eight motor angles at the current moment.

        Returns:
        Motor angles.
        """
        motor_angles = [
            self._pybullet_client.getJointState(robot, motor_id)[0]
            for motor_id in WHEEL_ID
        ]
        motor_angles = np.multiply(motor_angles, self._motor_direction)
        return motor_angles

    def get_motor_velocities(self, robot):
        """Get the velocity of all motors.

        Returns:
        Velocities of all motors.
        """
        motor_velocities = [
            self._pybullet_client.getJointState(robot, motor_id)[1]
            for motor_id in WHEEL_ID
        ]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities

    def get_base_orientation_euler(self, robot):
        """Get the orientation of walkerbro's base, represented as Euler angles.

        Returns:
        The orientation of walkerbro's base.
        """
        _, orientation = (self._pybullet_client.getBasePositionAndOrientation(robot))
        euler_orient = self._pybullet_client.getEulerFromQuaternion(orientation)
        return euler_orient

    def get_base_ang_vel(self, robot):
        # vel = self._pybullet_client.getBaseVelocity(robot)
        # return vel[1]
        (vx, vy, vz), (vr, vp, vy) = self._pybullet_client.getBaseVelocity(robot)
        # rotate to robot frame
        _, (x, y, z, w) = self.get_base_position_and_orientation(robot)
        # rotation matrix
        M = quat2mat((w, x, y, z))
        # velocities as vector
        v = np.array([vr, vp, vy]).T
        angular_vel_robot_frame = np.matmul(M.T, v)
        # return (vx, vy, vz), angular_vel_robot_frame
        return angular_vel_robot_frame

    def get_base_position_and_orientation(self, robot):
        position, quat = self._pybullet_client.getBasePositionAndOrientation(robot)
        return position, quat

    # ----------------------
    # External Forces
    # ----------------------
    def set_ext_forces_params(self, magn, duration, interval): #duration and interval in timesteps
        self._ext_force_magn = magn
        self._ext_force_duration = duration
        self._ext_force_interval = interval

    def _apply_force(self):
        if 100 < (self._env_step_counter%self._ext_force_interval) <= (100+self._ext_force_duration):
            if (self._env_step_counter%self._ext_force_interval) < 102:
                self._force_dir = -self._force_dir
            force = [0, self._force_dir*self._ext_force_magn, 0]
            print(force)
            self._pybullet_client.applyExternalForce(self.quadruped, -1, force, [0,0,0], pybullet.LINK_FRAME)
            com_pos = self.GetCOMPos()
            self._pybullet_client.addUserDebugLine([com_pos[0], com_pos[1], com_pos[2]],
                                                        [com_pos[0]+force[0]/1000, com_pos[1]+force[1]/1000, com_pos[2]+force[2]/1000],
                                                        [1, 0, 0], 3, 0.4)

    def set_motor_params(self, R, U_max, L, J, Kt, Kw, b, pole_pairs, gear_ratio, Z, Tk, k, alpha):
        self.mot_R = R
        self.mot_U_max = U_max
        self.mot_L = L
        self.mot_J = J
        self.mot_Kt = Kt
        self.mot_Kw = Kw
        self.mot_b = b
        self.mot_pole_pairs = pole_pairs
        self.mot_gear_ratio = gear_ratio
        self.mot_Z = Z
        self.mot_Tk = Tk
        self.mot_k = k
        self.mot_alpha = alpha


if __name__ == "__main__":
    print("Hi!!!!!!!!!!!")
    gyrobro = GyrobroBulletEnv(render=True, on_rack=False, motor_model_enabled=False)
    #     observation[0:1] are motor angles
    #     observation[2:3] are motor velocities
    #     observation[4:5] are base pitch and yaw
    #     observation[6:7] are base pitch and yaw angular velocities
    for i in range(10000):
        obs, rew, done, info = gyrobro.step([1.2, -1.2])

        # print(obs)
        print(f"Theta:   {obs[0]:.2f},  {obs[1]:.2f}")
        print(f"D_Theta: {obs[2]:.2f},  {obs[3]:.2f}")
        print(f"Pitch:   {obs[4]:.2f},  Yaw: {obs[5]:.2f}")
        print(f"D_Pitch: {obs[6]:.2f},  D_Yaw: {obs[7]:.2f}")
        print("====================")
    gyrobro.close()
    print("Bye!!!!!!!!!!")
