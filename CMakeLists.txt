cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB headers mathtoolbox/*.hpp)
file(GLOB sources mathtoolbox/*.cpp)

add_library(mathtoolbox STATIC ${headers} ${sources})

install(FILES ${headers} DESTINATION include/mathtoolbox)
install(TARGETS mathtoolbox ARCHIVE DESTINATION lib)

option(MATHTOOLBOX_BUILD_EXAMPLES "Build example applications" OFF)

if(MATHTOOLBOX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()
