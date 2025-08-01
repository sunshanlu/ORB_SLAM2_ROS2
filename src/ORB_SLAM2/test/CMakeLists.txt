add_executable(TxtVsProto ${CMAKE_CURRENT_SOURCE_DIR}/TxtVsProto.cc)
target_link_libraries(TxtVsProto ${PROJECT_NAME} DEPEND_LIBS)
target_include_directories(
    TxtVsProto PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                       ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
