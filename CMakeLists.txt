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
# Core policy library.
#
# Deliberately free of ROS and of unitree_go: it is the part that must be exactly right,
# and keeping it dependency-free is what lets the parity test run anywhere.
##
add_library(go2_rl_policy SHARED
  src/observation_builder.cpp
  src/policy_runner.cpp
  src/crc32.cpp
  src/unitree_remote.cpp
)
target_include_directories(go2_rl_policy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(go2_rl_policy PUBLIC onnxruntime::onnxruntime)

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

  install(TARGETS go2_rl_velocity_node DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING
    "missing ${node_deps_missing} -- building the policy core library and its tests "
    "only, skipping go2_rl_velocity_node. Source the unitree_ros2 workspace as an "
    "underlay to build the node (see README.md).")
endif()

install(TARGETS go2_rl_policy
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
  # scripts/gen_reference_trace.py is a developer tool that runs under the training
  # repo's virtualenv and follows that repo's style, not ament's. It is not installed and
  # never runs on the robot; reformatting it would only make it read inconsistently
  # against the code it exists to mirror.
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

  # Parity against a trace generated from the training repo's own observation packer.
  # Paths are baked in so the test needs no ROS environment.
  ament_add_gtest(test_policy_parity test/test_policy_parity.cpp)
  target_link_libraries(test_policy_parity go2_rl_policy)
  target_compile_definitions(test_policy_parity PRIVATE
    GO2_RL_POLICY_ONNX="${CMAKE_CURRENT_SOURCE_DIR}/config/policy/rugged-v2.onnx"
    GO2_RL_REFERENCE_CSV="${CMAKE_CURRENT_SOURCE_DIR}/test/reference/rugged_v2_reference.csv"
  )
endif()

ament_export_include_directories(include)
ament_export_libraries(go2_rl_policy)
ament_package()
