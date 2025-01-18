cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0074 NEW)

project(cone_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization filters)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(rosidl_default_generators REQUIRED)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Add library
add_library(${PROJECT_NAME}_library
  src/cone_detection_node.cpp
  include/cone_detection/cone_visualizer.h
  include/cone_detection/common_defs.h
  include/cone_detection/cone_detection_node.h
)

# Link libraries
target_link_libraries(${PROJECT_NAME}_library
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Add dependencies for the library
ament_target_dependencies(${PROJECT_NAME}_library
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
  rosidl_default_runtime
)

# Add executable
add_executable(${PROJECT_NAME}_node
  src/cone_detection_node.cpp
)

# Link the library to the executable
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_library
)

# Add dependencies for the executable
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
  rosidl_default_runtime
)

# Install targets
install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}_library
  DESTINATION lib/${PROJECT_NAME}
)

# Install include directory
install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

# Install msg files
install(
  DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export package
ament_package()
