cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)

######################################################################
# Include and link paths
######################################################################

set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
link_directories(${EXTERNAL_INSTALL_DIR}/lib)
include_directories(${EXTERNAL_INSTALL_DIR}/include)

find_package(Eigen3 REQUIRED)

######################################################################
# Build
######################################################################

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/*.hpp)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(mathtoolbox STATIC ${headers} ${sources})
target_include_directories(mathtoolbox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(mathtoolbox nlopt Eigen3::Eigen)

######################################################################
# Installation
######################################################################

install(FILES ${headers} DESTINATION include/mathtoolbox)
install(TARGETS mathtoolbox ARCHIVE DESTINATION lib)

######################################################################
# Example demos
######################################################################

option(MATHTOOLBOX_BUILD_EXAMPLES "Build example applications" OFF)

if(MATHTOOLBOX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

######################################################################
# External libraries
######################################################################

include(ExternalProject)
ExternalProject_Add(
	nlopt
	GIT_REPOSITORY ${CMAKE_SOURCE_DIR}/external/nlopt
	GIT_TAG eafdc8ce9403a4e1532239c1bebbdb86064eda38
	INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
	CMAKE_ARGS 
	-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
	-DBUILD_SHARED_LIBS=OFF
	-DNLOPT_GUILE=OFF
	-DNLOPT_LINK_PYTHON=OFF
	-DNLOPT_MATLAB=OFF
	-DNLOPT_OCTAVE=OFF
	-DNLOPT_PYTHON=OFF
	-DNLOPT_SWIG=OFF
)
ExternalProject_Add(
	nlopt-util
	GIT_REPOSITORY ${CMAKE_SOURCE_DIR}/external/nlopt-util
	GIT_TAG a45dd9aed0439a8d7eab3e0917d4a746d356097c
	INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
add_dependencies(mathtoolbox nlopt nlopt-util)
