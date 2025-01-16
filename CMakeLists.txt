cmake_minimum_required(VERSION 3.5)
project(cone_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization features)
find_package(Boost REQUIRED COMPONENTS thread)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MyMsgPair.msg"
  "msg/MyMsgPairArray.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(
  rosidl_default_runtime
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_library
  src/cone_detection_node.cpp
  include/cone_detection/cone_visualizer.h
  include/cone_detection/common_defs.h
  include/cone_detection/cone_detection_node.h
)

target_link_libraries(${PROJECT_NAME}_library
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
)

ament_target_dependencies(${PROJECT_NAME}_library
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
  rosidl_default_runtime
)

add_executable(${PROJECT_NAME}_node
  src/cone_detection_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_library
)

ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
  rosidl_default_runtime
)

install(TARGETS ${PROJECT_NAME}_node ${PROJECT_NAME}_library
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY msg/
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
