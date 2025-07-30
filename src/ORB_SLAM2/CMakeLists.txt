cmake_minimum_required(VERSION 3.5)
project(orb_slam2)
cmake_policy(SET CMP0072 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

option(BUILD_EXAMPLES "Build Example Application for test? " ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(orb_slam2_interfaces REQUIRED)

add_library(DEPEND_LIBS INTERFACE)
target_link_libraries(
    DEPEND_LIBS
    INTERFACE g2o::core
              g2o::stuff
              g2o::types_sba
              g2o::types_slam3d
              g2o::types_sim3
              ${Pangolin_LIBRARIES}
              ${OpenCV_LIBS}
              ${DBoW3_LIBS})

file(GLOB HEADER_FILE src/*.cc)
file(GLOB SOURCE_FILE src/*.cc)
add_library(${PROJECT_NAME} ${SOURCE_FILE} ${HEADER_FILE})
target_link_libraries(${PROJECT_NAME} DEPEND_LIBS)
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS}
                                                  ${DBoW3_INCLUDE_DIRS} include)
ament_target_dependencies(${PROJECT_NAME} rclcpp geometry_msgs cv_bridge
                          orb_slam2_interfaces)

if(BUILD_EXAMPLES)
    add_executable(kitti_example example/Stereo/KittiStereo.cc)
    target_link_libraries(kitti_example PUBLIC ${PROJECT_NAME})
    add_executable(tum_example example/RGB-D/TUMRGBD.cc)
    target_link_libraries(tum_example PUBLIC ${PROJECT_NAME})
    add_executable(ros2_example example/ROS2/ROS2Example.cc)
    target_link_libraries(ros2_example PUBLIC ${PROJECT_NAME})
    install(TARGETS ${PROJECT_NAME} kitti_example tum_example ros2_example
            DESTINATION lib/${PROJECT_NAME})
endif()

ament_package()
