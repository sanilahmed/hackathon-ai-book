cmake_minimum_required(VERSION 3.8)
project(cpp_examples)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Publisher example
add_executable(talker_cpp src/talker.cpp)
ament_target_dependencies(talker_cpp rclcpp std_msgs)

# Subscriber example
add_executable(listener_cpp src/listener.cpp)
ament_target_dependencies(listener_cpp rclcpp std_msgs)

# Service server example
add_executable(service_server_cpp src/service_server.cpp)
ament_target_dependencies(service_server_cpp rclcpp std_msgs)

# Service client example
add_executable(service_client_cpp src/service_client.cpp)
ament_target_dependencies(service_client_cpp rclcpp std_msgs)

# Action server example
add_executable(action_server_cpp src/action_server.cpp)
ament_target_dependencies(action_server_cpp rclcpp std_msgs)

# Action client example
add_executable(action_client_cpp src/action_client.cpp)
ament_target_dependencies(action_client_cpp rclcpp std_msgs)

# Install executables
install(TARGETS
  talker_cpp
  listener_cpp
  service_server_cpp
  service_client_cpp
  action_server_cpp
  action_client_cpp
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and cpplint is enabled
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()