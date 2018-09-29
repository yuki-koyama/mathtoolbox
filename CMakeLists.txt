cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)

######################################################################
# Options
######################################################################

option(MATHTOOLBOX_NLOPT_BASED_MODULES "Build NLopt-based modules" ON)
option(MATHTOOLBOX_BUILD_EXAMPLES "Build example applications" OFF)

######################################################################
# Include and link paths
######################################################################

if(MATHTOOLBOX_NLOPT_BASED_MODULES)
	set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
	link_directories(${EXTERNAL_INSTALL_DIR}/lib)
	include_directories(${EXTERNAL_INSTALL_DIR}/include)
endif()

find_package(Eigen3 REQUIRED)

######################################################################
# Build
######################################################################

file(GLOB
	headers
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/classical-mds.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/rbf-interpolation.hpp
)
file(GLOB
	sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/rbf-interpolation.cpp
)
file(GLOB
	headers_nlopt
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/gaussian-process-regression.hpp	
)
file(GLOB
	sources_nlopt
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian-process-regression.cpp
)

if(MATHTOOLBOX_NLOPT_BASED_MODULES)
	add_library(mathtoolbox STATIC ${headers} ${sources} ${headers_nlopt} ${sources_nlopt})
	target_link_libraries(mathtoolbox nlopt Eigen3::Eigen)
else()
	add_library(mathtoolbox STATIC ${headers} ${sources})
	target_link_libraries(mathtoolbox Eigen3::Eigen)
endif()
target_include_directories(mathtoolbox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

######################################################################
# Installation
######################################################################

if(MATHTOOLBOX_NLOPT_BASED_MODULES)
	install(FILES ${headers} ${headers_nlopt} DESTINATION include/mathtoolbox)
else()
	install(FILES ${headers} DESTINATION include/mathtoolbox)
endif()
install(TARGETS mathtoolbox ARCHIVE DESTINATION lib)

######################################################################
# Example demos
######################################################################

if(MATHTOOLBOX_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

######################################################################
# External libraries
######################################################################

if(MATHTOOLBOX_NLOPT_BASED_MODULES)
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
endif()
