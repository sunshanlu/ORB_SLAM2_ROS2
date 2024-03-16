cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM2)

set(CMAKE_CXX_STANDARD 14)

include_directories(include /usr/include/eigen3)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV 3.1.0 REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Glog REQUIRED)
find_package(G2O REQUIRED)

add_library(DEPEND_LIBS INTERFACE)
target_link_libraries(
    DEPEND_LIBS
    INTERFACE ${OpenCV_LIBS}
              DBoW3
              ${GLOG_LIBRARIES}
              g2o_core
              g2o_stuff
              g2o_types_sba
              g2o_solver_csparse
              g2o_csparse_extension)

file(GLOB SOURCE_FILE src/*.cc)
add_executable(${PROJECT_NAME} ${SOURCE_FILE})
ament_target_dependencies(${PROJECT_NAME} rclcpp)
# target_link_libraries传入的是target的名字
target_link_libraries(${PROJECT_NAME} DEPEND_LIBS)
install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

add_subdirectory(test)
ament_package()
