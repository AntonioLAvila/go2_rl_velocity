# Copyright (c) 2026 Antonio Avila. MIT License.
"""Launch the Go2 flat-ground (torque) RL velocity controller.

The sibling of go2_rl_velocity.launch.py, which launches the rugged position controller.
The two are separate nodes with separate parameter files and must not be run at the same
time -- they would both publish /lowcmd.
"""

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import os


def generate_launch_description():
    share = get_package_share_directory('go2_rl_velocity')
    default_params = os.path.join(share, 'config', 'go2_rl_flat.yaml')

    args = [
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Parameter YAML for the flat controller.'),
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
        DeclareLaunchArgument(
            'torque_scale', default_value='1.0',
            description='Global derate on the policy torque, for bring-up. 1.0 is the '
                        'trained authority.'),
    ]

    controller = Node(
        package='go2_rl_velocity',
        executable='go2_rl_flat_node',
        name='go2_rl_flat',
        output='screen',
        emulate_tty=True,
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'policy_path': LaunchConfiguration('policy_path'),
                'remote_debug': LaunchConfiguration('remote_debug'),
                'dry_run': LaunchConfiguration('dry_run'),
                'use_joystick_command': LaunchConfiguration('use_joystick_command'),
                # Declared as a double in the node, so the substitution has to be typed
                # explicitly -- an untyped '0.5' would arrive as a string and be rejected.
                'torque_scale': ParameterValue(
                    LaunchConfiguration('torque_scale'), value_type=float),
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
