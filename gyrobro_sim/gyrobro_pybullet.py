import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu

from gyrobro_sim.gyrobro_gym_env import GyrobroBulletEnv
# from pyrr import Quaternion
from scipy.spatial.transform import Rotation
import time
import numpy as np


class GyrobroPybullet(Node):

    def __init__(self):
        super().__init__('gyrobro_pybullet')

        self.declare_parameter('freq', 100.0)
        self.declare_parameter('wheels_left_cmd_topic', "wheels/left/torque")
        self.declare_parameter('wheels_right_cmd_topic', "wheels/right/torque")
        self.declare_parameter('wheels_state_topic', "wheels/state")
        self.declare_parameter('imu_topic', "bno055/imu")
        self.declare_parameter('sensor_noise', [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('torq_max', 20.0)
        self.declare_parameter('Kt', 1.0)
        self.declare_parameter('Psi0', 0.1)

        freq = self.get_parameter('freq').get_parameter_value().double_value
        wheels_left_cmd_topic = self.get_parameter('wheels_left_cmd_topic').get_parameter_value().string_value
        wheels_right_cmd_topic = self.get_parameter('wheels_right_cmd_topic').get_parameter_value().string_value
        wheels_state_topic = self.get_parameter('wheels_state_topic').get_parameter_value().string_value
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.torq_max = self.get_parameter('torq_max').get_parameter_value().double_value
        self.Kt = self.get_parameter('Kt').get_parameter_value().double_value
        self.Psi0 = self.get_parameter('Psi0').get_parameter_value().double_value
        self.sensor_noise = self.get_parameter('sensor_noise').get_parameter_value().double_array_value

        self.wheels_state_pub = self.create_publisher(JointState, wheels_state_topic, 10)
        self.imu_pub = self.create_publisher(Imu, imu_topic, 10)

        self.wheel_left_cmd_sub = self.create_subscription(
            Float64,
            wheels_left_cmd_topic,
            self.wheel_left_cmd_callback,
            10)
        self.wheel_right_cmd_sub = self.create_subscription(
            Float64,
            wheels_right_cmd_topic,
            self.wheel_right_cmd_callback,
            10)
        self.wheel_left_cmd_sub  # prevent unused variable warning
        self.wheel_right_cmd_sub

        self.wheels_state_msg = JointState()
        self.imu_msg = Imu()

        # init simulator
        self.env = GyrobroBulletEnv(render=True,
                                    real_time=False,
                                    sim_freq=freq,
                                    urdf_root="/home/yoggi/gyrobro_control/src/gyrobro_sim/urdf/gyrobro.urdf",
                                    # urdf_root="./urdf",
                                    on_rack=False,
                                    initial_pitch=self.Psi0,
                                    observation_noise_stdev=self.sensor_noise)
        self.action = [0.0, 0.0]

        timer_period = 1/freq  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)


    def wheel_left_cmd_callback(self, msg : Float64):
        # self.get_logger().info('Ref torque left: "%s" Nm' % msg.data)
        self.action[0] = msg.data


    def wheel_right_cmd_callback(self, msg : Float64):
        # self.get_logger().info('Ref torque right: "%s" Nm' % msg.data)
        self.action[1] = msg.data


    def timer_callback(self):

        self._last_frame_time = time.time()

        self.action[0] *= self.Kt
        self.action[1] *= self.Kt
        np.clip(self.action, self.torq_max, -self.torq_max)

        obs, rew, done, info = self.env.step(self.action)

        time_spent = time.time() - self._last_frame_time

        # get output values
        theta_l = obs[0]
        theta_r = obs[1]

        d_theta_l = obs[2]
        d_theta_r = obs[3]

        psi = -obs[4]
        d_psi = obs[6]

        phi = obs[5]
        d_phi = -obs[7]

        roll = 0

        # convert from euler to quaternion
        rot = Rotation.from_euler('xyz', [roll, psi, phi], degrees=False)
        rot_quat = rot.as_quat()

        # set publisher messages
        self.wheels_state_msg.name = ["left", "right"]
        self.wheels_state_msg.position = [theta_l, theta_r]
        self.wheels_state_msg.velocity = [d_theta_l, d_theta_r]

        self.imu_msg.orientation.x = rot_quat[0]
        self.imu_msg.orientation.y = rot_quat[1]
        self.imu_msg.orientation.z = rot_quat[2]
        self.imu_msg.orientation.w = rot_quat[3]
        self.imu_msg.angular_velocity.x = 0.0
        self.imu_msg.angular_velocity.y = d_psi
        self.imu_msg.angular_velocity.z = d_phi

        # publish messages
        self.wheels_state_pub.publish(self.wheels_state_msg)
        self.imu_pub.publish(self.imu_msg)



def main(args=None):
    rclpy.init(args=args)

    gb_pybullet = GyrobroPybullet()

    rclpy.spin(gb_pybullet)


    gb_pybullet.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
