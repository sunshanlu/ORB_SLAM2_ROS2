file(GLOB SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/*.cc)
# 注意，这里需要传入的是变量名，而不是变量值
list(REMOVE_ITEM SOURCE_FILE ${PROJECT_SOURCE_DIR}/src/main.cc)

include_directories(${PROJECT_SOURCE_DIR}/include)

# ORB特征点提取器测试
add_executable(
    ORBExtractorTest ${SOURCE_FILE} ./ORBExtractorTest/ORBExtractorTest.cc
                     ./ORBExtractorTest/ORB_SLAM2_E/ORBextractor.cc)
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
add_executable(OptimizePoseOnlyTest ${SOURCE_FILE}
                                    ./OptimizerTest/OptimizePoseOnly.cc)
ament_target_dependencies(OptimizePoseOnlyTest rclcpp)
target_link_libraries(OptimizePoseOnlyTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByProjection中的恒速模型匹配
add_executable(SearchByProjectionMMTest
               ${SOURCE_FILE} ./ORBMatcherTest/SearchByProjectionMM.cc)
ament_target_dependencies(SearchByProjectionMMTest rclcpp)
target_link_libraries(SearchByProjectionMMTest DEPEND_LIBS)

# ORB特征点匹配器测试SearchByProjection中的跟踪局部地图
add_executable(SearchByProjectionLMTest
               ${SOURCE_FILE} ./ORBMatcherTest/SearchByProjectionLM.cc)
ament_target_dependencies(SearchByProjectionLMTest rclcpp)
target_link_libraries(SearchByProjectionLMTest DEPEND_LIBS)

# PnPSolver求解器测试EPnP
add_executable(EPnPTest ${SOURCE_FILE} ./PnPSolverTest/EPnPTest.cc)
ament_target_dependencies(EPnPTest rclcpp)
target_link_libraries(EPnPTest DEPEND_LIBS)

# KeyFrameDB关键帧数据库测试FindRelocKfs
add_executable(FindRelocKfs ${SOURCE_FILE} ./KeyFrameDBTest/FindRelocKfs.cc)
ament_target_dependencies(FindRelocKfs rclcpp)
target_link_libraries(FindRelocKfs DEPEND_LIBS)

# Tracking跟踪模块测试 普通跟踪测试
add_executable(NormalTracking ${SOURCE_FILE} ./TrackingTest/NormalTracking.cc)
ament_target_dependencies(NormalTracking rclcpp)
target_link_libraries(NormalTracking DEPEND_LIBS)

# Tracking跟踪模块测试 重定位跟踪测试
add_executable(RelocTracking ${SOURCE_FILE} ./TrackingTest/RelocTracking.cc)
ament_target_dependencies(RelocTracking rclcpp)
target_link_libraries(RelocTracking DEPEND_LIBS)

# LocalMapping局部建图模块测试 串联测试
add_executable(ScLocalMappingTest ${SOURCE_FILE}
                                  ./LocalMappingTest/ScLocalMappingTest.cc)
ament_target_dependencies(ScLocalMappingTest rclcpp)
target_link_libraries(ScLocalMappingTest DEPEND_LIBS)

# LocalMapping局部建图模块测试 并联测试
add_executable(ParallMappingTest ${SOURCE_FILE}
                                 ./LocalMappingTest/ParallMappingTest.cc)
ament_target_dependencies(ParallMappingTest rclcpp)
target_link_libraries(ParallMappingTest DEPEND_LIBS)

# Sim3Solver求解器测试
add_executable(Sim3SolverTest ${SOURCE_FILE} ./Sim3SolverTest/Sim3Test.cc)
ament_target_dependencies(Sim3SolverTest rclcpp)
target_link_libraries(Sim3SolverTest DEPEND_LIBS)

# Sim3重投影匹配测试
add_executable(SearchBySim3Test ${SOURCE_FILE} ./ORBMatcherTest/SearchBySim3.cc)
ament_target_dependencies(SearchBySim3Test rclcpp)
target_link_libraries(SearchBySim3Test DEPEND_LIBS)

# Sim3重投影匹配测试
add_executable(OptimizeSim3Test ${SOURCE_FILE} ./OptimizerTest/OptimizeSim3.cc)
ament_target_dependencies(OptimizeSim3Test rclcpp)
target_link_libraries(OptimizeSim3Test DEPEND_LIBS)

# Sim3重投影匹配测试(投影地图点)
add_executable(SearchBySim3MpsTest ${SOURCE_FILE} ./ORBMatcherTest/SearchBySim3Mps.cc)
ament_target_dependencies(SearchBySim3MpsTest rclcpp)
target_link_libraries(SearchBySim3MpsTest DEPEND_LIBS)

# 关键帧数据库测试（寻找回环闭合关键帧）
add_executable(FindLoopCloseKfsTest ${SOURCE_FILE} ./KeyFrameDBTest/FindLoopCloseKfs.cc)
ament_target_dependencies(FindLoopCloseKfsTest rclcpp)
target_link_libraries(FindLoopCloseKfsTest DEPEND_LIBS)

# 回环闭合线程测试
add_executable(LoopClosingTest ${SOURCE_FILE} ./LoopClosingTest/LoopClosingTest.cc)
ament_target_dependencies(LoopClosingTest rclcpp)
target_link_libraries(LoopClosingTest DEPEND_LIBS)