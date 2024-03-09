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
