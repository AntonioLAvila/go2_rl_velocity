cmake_minimum_required(VERSION 3.16)
project(go2_rl_velocity)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep the ONNX Runtime library directory in the installed binaries' RPATH. Without this
# everything builds and installs cleanly and then fails at exec time with
# "libonnxruntime.so.1: cannot open shared object file", because ORT is normally unpacked
# somewhere outside the loader's default search path.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(ament_cmake REQUIRED)
find_package(ONNXRUNTIME REQUIRED)

##
# Shared, policy-agnostic pieces.
#
# The Unitree wire format and the wireless remote have nothing to do with which policy is
# driving, so both controllers link this rather than carrying a copy each. Everything that
# *is* policy-specific stays duplicated between go2_rl_policy and go2_rl_flat_policy on
# purpose -- see CLAUDE.md.
##
add_library(go2_rl_common SHARED
  src/crc32.cpp
  src/unitree_remote.cpp
)
target_include_directories(go2_rl_common PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

##
# Core policy library, rugged (PD position, 225-dim stacked observation).
#
# Deliberately free of ROS and of unitree_go: it is the part that must be exactly right,
# and keeping it dependency-free is what lets the parity test run anywhere.
##
add_library(go2_rl_policy SHARED
  src/observation_builder.cpp
  src/policy_runner.cpp
)
target_link_libraries(go2_rl_policy PUBLIC go2_rl_common onnxruntime::onnxruntime)

##
# Core policy library, flat ground (direct torque, 45-dim single frame).
#
# Same constraints as go2_rl_policy, and independent of it. The FSM lives here rather than
# in the node target -- unlike the rugged controller's -- because the position-to-torque
# handover and the torque watchdogs are exactly the logic worth testing without a robot.
##
add_library(go2_rl_flat_policy SHARED
  src/flat/policy_runner.cpp
  src/flat/controller_fsm.cpp
)
target_link_libraries(go2_rl_flat_policy PUBLIC go2_rl_common onnxruntime::onnxruntime)

##
# Controller node.
#
# unitree_go comes from the unitree_ros2 workspace, which has no rosdep key. When it is
# not on the prefix path we still build and test the core library -- that is the whole
# point of the split -- and just skip the node.
##
find_package(rclcpp QUIET)
find_package(unitree_go QUIET)
find_package(geometry_msgs QUIET)
find_package(std_msgs QUIET)
find_package(std_srvs QUIET)
find_package(ament_index_cpp QUIET)

set(node_deps rclcpp unitree_go geometry_msgs std_msgs std_srvs ament_index_cpp)
set(node_deps_found TRUE)
foreach(dep IN LISTS node_deps)
  if(NOT ${dep}_FOUND)
    set(node_deps_found FALSE)
    list(APPEND node_deps_missing ${dep})
  endif()
endforeach()

if(node_deps_found)
  add_executable(go2_rl_velocity_node
    src/main.cpp
    src/controller_fsm.cpp
    src/rl_controller_node.cpp
    src/lowcmd_builder.cpp
  )
  # Modern target linking rather than ament_target_dependencies: the latter uses the
  # plain target_link_libraries signature internally, which cannot be mixed with the
  # keyword form used for go2_rl_policy.
  target_link_libraries(go2_rl_velocity_node PRIVATE
    go2_rl_policy
    rclcpp::rclcpp
    ament_index_cpp::ament_index_cpp
    ${unitree_go_TARGETS}
    ${geometry_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${std_srvs_TARGETS}
  )

  add_executable(go2_rl_flat_node
    src/flat/main.cpp
    src/flat/flat_controller_node.cpp
    src/lowcmd_builder.cpp
  )
  target_link_libraries(go2_rl_flat_node PRIVATE
    go2_rl_flat_policy
    rclcpp::rclcpp
    ament_index_cpp::ament_index_cpp
    ${unitree_go_TARGETS}
    ${geometry_msgs_TARGETS}
    ${std_msgs_TARGETS}
    ${std_srvs_TARGETS}
  )

  install(TARGETS go2_rl_velocity_node go2_rl_flat_node DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING
    "missing ${node_deps_missing} -- building the policy core libraries and their tests "
    "only, skipping go2_rl_velocity_node and go2_rl_flat_node. Source the unitree_ros2 "
    "workspace as an underlay to build the nodes (see README.md).")
endif()

install(TARGETS go2_rl_common go2_rl_policy go2_rl_flat_policy
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  # The scripts/gen_*_reference_trace.py generators are developer tools that run under the
  # training repo's virtualenv and follow that repo's style, not ament's. They are not
  # installed and never run on the robot; reformatting them would only make them read
  # inconsistently against the code they exist to mirror.
  set(ament_cmake_flake8_FOUND TRUE)
  set(ament_cmake_pep257_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_observation_builder test/test_observation_builder.cpp)
  target_link_libraries(test_observation_builder go2_rl_policy)

  ament_add_gtest(test_joint_map test/test_joint_map.cpp)
  target_link_libraries(test_joint_map go2_rl_policy)

  ament_add_gtest(test_unitree_remote test/test_unitree_remote.cpp)
  target_link_libraries(test_unitree_remote go2_rl_policy)

  ament_add_gtest(test_command_envelope test/test_command_envelope.cpp)
  target_link_libraries(test_command_envelope go2_rl_policy)

  # Parity against a trace generated from the training repo's own observation packer.
  # Paths are baked in so the test needs no ROS environment.
  ament_add_gtest(test_policy_parity test/test_policy_parity.cpp)
  target_link_libraries(test_policy_parity go2_rl_policy)
  target_compile_definitions(test_policy_parity PRIVATE
    GO2_RL_POLICY_ONNX="${CMAKE_CURRENT_SOURCE_DIR}/config/policy/latest.onnx"
    GO2_RL_REFERENCE_CSV="${CMAKE_CURRENT_SOURCE_DIR}/test/reference/latest_reference.csv"
  )

  ##
  # Flat-ground controller.
  ##
  ament_add_gtest(test_flat_state_conversion test/test_flat_state_conversion.cpp)
  target_link_libraries(test_flat_state_conversion go2_rl_flat_policy)

  ament_add_gtest(test_flat_controller_fsm test/test_flat_controller_fsm.cpp)
  target_link_libraries(test_flat_controller_fsm go2_rl_flat_policy)
  target_compile_definitions(test_flat_controller_fsm PRIVATE
    GO2_RL_FLAT_POLICY_ONNX="${CMAKE_CURRENT_SOURCE_DIR}/config/policy/flat-v0.onnx"
  )

  # The flat exports carry no ONNX metadata, so this is not merely the primary guard on
  # the flat contract -- it is the only one. See flat/policy_runner.hpp.
  ament_add_gtest(test_flat_policy_parity test/test_flat_policy_parity.cpp)
  target_link_libraries(test_flat_policy_parity go2_rl_flat_policy)
  target_compile_definitions(test_flat_policy_parity PRIVATE
    GO2_RL_FLAT_POLICY_ONNX="${CMAKE_CURRENT_SOURCE_DIR}/config/policy/flat-v0.onnx"
    GO2_RL_FLAT_REFERENCE_CSV="${CMAKE_CURRENT_SOURCE_DIR}/test/reference/flat_feet_time_25_reference.csv"
    GO2_RL_RUGGED_POLICY_ONNX="${CMAKE_CURRENT_SOURCE_DIR}/config/policy/latest.onnx"
  )

  # Needs unitree_go for the real LowCmd wire type, so it is gated the same way the node
  # itself is -- skipped where unitree_go is not on the prefix path.
  if(node_deps_found)
    ament_add_gtest(test_lowcmd_builder test/test_lowcmd_builder.cpp src/lowcmd_builder.cpp)
    target_link_libraries(test_lowcmd_builder go2_rl_policy rclcpp::rclcpp
      ${unitree_go_TARGETS})
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(go2_rl_common go2_rl_policy go2_rl_flat_policy)
ament_package()
