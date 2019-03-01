cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)

######################################################################
# Options
######################################################################

option(MATHTOOLBOX_NLOPT_BASED_MODULES "Build NLopt-based modules" ON)
option(MATHTOOLBOX_BUILD_EXAMPLES "Build example applications" OFF)

######################################################################
# Check prerequisites
######################################################################

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
	target_link_libraries(mathtoolbox Eigen3::Eigen nlopt nlopt-util)
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
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nlopt)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nlopt-util)
endif()
