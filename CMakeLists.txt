cmake_minimum_required(VERSION 3.0.2)
project(synesthesiax)

## Compile as C++14 (you can adjust if needed)
add_compile_options(-std=c++14)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  message_filters
)

## Find OpenCV
find_package(OpenCV REQUIRED)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs cv_bridge pcl_ros pcl_conversions message_filters
)

## Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare executable
add_executable(synesthesiax_node
  src/synesthesiax.cpp
  src/Projector.cpp
)

## Link libraries
target_link_libraries(synesthesiax_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## Add dependency to generated messages
add_dependencies(synesthesiax_node ${catkin_EXPORTED_TARGETS})

## Install rules (optional but recommended)
install(TARGETS synesthesiax_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
