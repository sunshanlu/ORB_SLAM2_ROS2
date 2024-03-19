file(GLOB SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/*.cc)
# 注意，这里需要传入的是变量名，而不是变量值
list(REMOVE_ITEM SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/main.cc)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ORB特征点提取器测试
add_executable(ORBExtractorTest ${SOURCE_FILE}
                                ./ORBExtractorTest/ORBExtractorTest.cc)
ament_target_dependencies(ORBExtractorTest rclcpp)
target_link_libraries(ORBExtractorTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByStereoTest
add_executable(SearchByStereoTest ${SOURCE_FILE}
                                  ./ORBMatcherTest/SearchByStereo.cc)
ament_target_dependencies(SearchByStereoTest rclcpp)
target_link_libraries(SearchByStereoTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByBowTest
add_executable(SearchByBowTest ${SOURCE_FILE} ./ORBMatcherTest/SearchByBow.cc)
ament_target_dependencies(SearchByBowTest rclcpp)
target_link_libraries(SearchByBowTest DEPEND_LIBS)

# 优化器测试OptimizePoseOnlyTest
add_executable(OptimizePoseOnlyTest ${SOURCE_FILE} ./OptimizerTest/OptimizePoseOnly.cc)  
ament_target_dependencies(OptimizePoseOnlyTest rclcpp)
target_link_libraries(OptimizePoseOnlyTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByProjection中的恒速模型匹配
add_executable(SearchByProjectionMMTest ${SOURCE_FILE} ./ORBMatcherTest/SearchByProjectionMM.cc)
ament_target_dependencies(SearchByProjectionMMTest rclcpp)
target_link_libraries(SearchByProjectionMMTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByProjection中的跟踪局部地图
add_executable(SearchByProjectionLMTest ${SOURCE_FILE} ./ORBMatcherTest/SearchByProjectionLM.cc)
ament_target_dependencies(SearchByProjectionLMTest rclcpp)
target_link_libraries(SearchByProjectionLMTest DEPEND_LIBS)