# Copyright (c) 2026 Antonio Avila. MIT License.
"""Launch the Go2 RL velocity controller."""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

import os


def generate_launch_description():
    share = get_package_share_directory('go2_rl_velocity')
    default_params = os.path.join(share, 'config', 'go2_rl_velocity.yaml')

    args = [
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Parameter YAML for the controller.'),
        DeclareLaunchArgument(
            'policy_path', default_value='',
            description='ONNX policy. Empty uses the one installed with the package.'),
        DeclareLaunchArgument(
            'remote_debug', default_value='false',
            description='Log the decoded wireless remote, to verify the button layout.'),
        DeclareLaunchArgument(
            'dry_run', default_value='false',
            description='Run everything but publish no LowCmd.'),
        DeclareLaunchArgument(
            'use_joystick_command', default_value='true',
            description='Velocity command from the remote sticks (else /cmd_vel).'),
    ]

    controller = Node(
        package='go2_rl_velocity',
        executable='go2_rl_velocity_node',
        name='go2_rl_velocity',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'policy_path': LaunchConfiguration('policy_path'),
                'remote_debug': LaunchConfiguration('remote_debug'),
                'dry_run': LaunchConfiguration('dry_run'),
                'use_joystick_command': LaunchConfiguration('use_joystick_command'),
            },
        ],
        # The Go2's low-level channels are plain /lowstate and /lowcmd, which is what the
        # unitree_ros2 bridge exposes. Remapped here rather than hard-coded so the node
        # can also be pointed at a bag or a namespaced bridge.
        remappings=[
            ('lowstate', '/lowstate'),
            ('lowcmd', '/lowcmd'),
            ('cmd_vel', '/cmd_vel'),
        ],
    )

    return LaunchDescription(args + [controller])
