cmake_minimum_required(VERSION 2.8.3)
project(serial_example)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  serial
  std_msgs
  sensor_msgs
  message_generation
)

catkin_package(
  CATKIN_DEPENDS
    serial
    std_msgs
    sensor_msgs
    message_runtime
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(serial_example_node src/serial_example_node.cpp)
add_executable(listener src/listener.cpp)

target_link_libraries(serial_example_node
  ${catkin_LIBRARIES}
)
target_link_libraries(listener
  ${catkin_LIBRARIES}
)
