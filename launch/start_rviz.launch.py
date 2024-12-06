# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
# from launch_ros.substitutions import FindPackageShare

# def generate_launch_description():
    # Author: Addison Sears-Collins
# Date: September 14, 2021
# Description: Launch a two-wheeled robot URDF file using Rviz.
# https://automaticaddison.com
 
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
 
def generate_launch_description():

    # Set the path to this package.
    pkg_share = FindPackageShare(package='gyrobro_sim').find('gyrobro_sim')
    print(pkg_share)

    # Set the path to the RViz configuration settings
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/rviz_basic_settings.rviz')

    # Set the path to the URDF file
    default_urdf_model_path = os.path.join(pkg_share, 'urdf/gyrobro.xacro')

    ########### YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE ##############  
    # Launch configuration variables specific to simulation
    gui = LaunchConfiguration('gui')
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare the launch arguments  
    declare_urdf_model_path_cmd = DeclareLaunchArgument(
    name='urdf_model', 
    default_value=default_urdf_model_path, 
    description='Absolute path to robot urdf file')
        
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
    name='rviz_config_file',
    default_value=default_rviz_config_path,
    description='Full path to the RVIZ config file to use')
        
    declare_use_joint_state_publisher_cmd = DeclareLaunchArgument(
    name='gui',
    default_value='True',
    description='Flag to enable joint_state_publisher_gui')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
    name='use_robot_state_pub',
    default_value='True',
    description='Whether to start the robot state publisher')

    declare_use_rviz_cmd = DeclareLaunchArgument(
    name='use_rviz',
    default_value='True',
    description='Whether to start RVIZ')
        
    declare_use_sim_time_cmd = DeclareLaunchArgument(
    name='use_sim_time',
    default_value='True',
    description='Use simulation (Gazebo) clock if true')

    # Specify the actions

    # Publish the joint state values for the non-fixed joints in the URDF file.
    start_joint_state_publisher_cmd = Node(
    condition=UnlessCondition(gui),
    package='joint_state_publisher',
    executable='joint_state_publisher',
    name='joint_state_publisher')

    # A GUI to manipulate the joint state values
    start_joint_state_publisher_gui_node = Node(
    condition=IfCondition(gui),
    package='joint_state_publisher_gui',
    executable='joint_state_publisher_gui',
    name='joint_state_publisher_gui')

    # Subscribe to the joint states of the robot, and publish the 3D pose of each link.
    start_robot_state_publisher_cmd = Node(
    condition=IfCondition(use_robot_state_pub),
    package='robot_state_publisher',
    executable='robot_state_publisher',
    parameters=[{'use_sim_time': use_sim_time, 
    'robot_description': Command(['xacro ', urdf_model])}],
    arguments=[default_urdf_model_path])

    # Launch RViz
    start_rviz_cmd = Node(
    condition=IfCondition(use_rviz),
    package='rviz2',
    executable='rviz2',
    name='rviz2',
    output='screen',
    arguments=['-d', rviz_config_file])

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_joint_state_publisher_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)  
    ld.add_action(declare_use_rviz_cmd) 
    ld.add_action(declare_use_sim_time_cmd)

    # Add any actions
    ld.add_action(start_joint_state_publisher_cmd)
    ld.add_action(start_joint_state_publisher_gui_node)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    return ld
    # ld = LaunchDescription()

    # node_path = FindPackageShare('gyrobro_sim')
    # default_model_path = PathJoinSubstitution(['urdf', 'test.urdf'])
    # default_rviz_config_path = PathJoinSubstitution([node_path, 'rviz', 'urdf.rviz'])

    # gui_arg = DeclareLaunchArgument(name='gui', default_value='true', choices=['true', 'false'],
    #                                 description='Flag to enable joint_state_publisher_gui')
    # ld.add_action(gui_arg)

    # rviz_arg = DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
    #                                  description='Absolute path to rviz config file')
    # ld.add_action(rviz_arg)

    # ld.add_action(DeclareLaunchArgument(name='model', default_value=default_model_path,
    #                                     description='Path to robot urdf file relative to urdf_tutorial package'))

    # ld.add_action(IncludeLaunchDescription(
    #     PathJoinSubstitution([FindPackageShare('gyrobro_sim'), 'launch', 'display.launch.py']),
    #     launch_arguments={
    #         'urdf_package': 'gyrobro_sim',
    #         'urdf_package_path': LaunchConfiguration('model'),
    #         'rviz_config': LaunchConfiguration('rvizconfig'),
    #         'jsp_gui': LaunchConfiguration('gui')}.items()
    # ))

    # return ld

    

    # return LaunchDescription([
    #     Node(
    #         package='turtlesim',
    #         namespace='turtlesim1',
    #         executable='turtlesim_node',
    #         name='sim'
    #     ),
    #     Node(
    #         package='turtlesim',
    #         namespace='turtlesim2',
    #         executable='turtlesim_node',
    #         name='sim'
    #     ),
    #     Node(
    #         package='turtlesim',
    #         executable='mimic',
    #         name='mimic',
    #         remappings=[
    #             ('/input/pose', '/turtlesim1/turtle1/pose'),
    #             ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),
    #         ]
    #     )
    # ])