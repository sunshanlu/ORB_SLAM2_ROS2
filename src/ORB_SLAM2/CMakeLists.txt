cmake_minimum_required(VERSION 3.5)
project(orb_slam2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

include_directories(include /usr/include/eigen3 ${THIRDPARTY_INCLUDE_DIR})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 3.1.0 REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(G2O REQUIRED)
find_package(Pangolin REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(orb_slam2_interfaces REQUIRED)

add_library(DEPEND_LIBS INTERFACE)
target_link_libraries(
    DEPEND_LIBS
    INTERFACE g2o_core
              g2o_stuff
              g2o_types_sba
              g2o_types_slam3d
              g2o_types_sim3)

file(GLOB SOURCE_FILE src/*.cc)
add_library(${PROJECT_NAME} ${SOURCE_FILE})
target_link_libraries(${PROJECT_NAME} DEPEND_LIBS)
ament_target_dependencies(
    ${PROJECT_NAME}
    rclcpp
    geometry_msgs
    cv_bridge
    orb_slam2_interfaces
    DBoW3
    OpenCV
    Pangolin)

add_executable(kitti_example example/Stereo/KittiStereo.cc)
target_link_libraries(kitti_example PUBLIC ${PROJECT_NAME})

add_executable(tum_example example/RGB-D/TUMRGBD.cc)
target_link_libraries(tum_example PUBLIC ${PROJECT_NAME})

add_executable(ros2_example example/ROS2/ROS2Example.cc)
target_link_libraries(ros2_example PUBLIC ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME} kitti_example tum_example ros2_example
        DESTINATION lib/${PROJECT_NAME})
ament_package()
