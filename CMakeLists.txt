cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)
if(MATHTOOLBOX_PYTHON_BINDINGS)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

######################################################################
# Options
######################################################################

option(MATHTOOLBOX_BUILD_EXAMPLES  "Build example applications" OFF)
option(MATHTOOLBOX_PYTHON_BINDINGS "Build python bindings"      OFF)

######################################################################
# Prerequisites
######################################################################

find_package(Eigen3 REQUIRED)
if((NOT TARGET Eigen3::Eigen) AND (DEFINED EIGEN3_INCLUDE_DIR))
	add_library(AliasEigen3 INTERFACE)
	target_include_directories(AliasEigen3 INTERFACE ${EIGEN3_INCLUDE_DIR})
	add_library(Eigen3::Eigen ALIAS AliasEigen3)
endif()

######################################################################
# External libraries
######################################################################

if(MATHTOOLBOX_PYTHON_BINDINGS)
	set(PYBIND11_INSTALL        OFF CACHE INTERNAL "" FORCE)
	set(PYBIND11_PYTHON_VERSION ""  CACHE INTERNAL "" FORCE)
	set(PYBIND11_TEST           OFF CACHE INTERNAL "" FORCE)
	set(USE_PYTHON_INCLUDE_DIR  OFF CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
endif()

######################################################################
# Core library
######################################################################

set(headers
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/backtracking-line-search.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/bfgs.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/classical-mds.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/constants.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/distributions.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/gaussian-process-regression.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/l-bfgs.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/numerical-optimization.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/rbf-interpolation.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/strong-wolfe-conditions-line-search.hpp
)
set(sources
	${CMAKE_CURRENT_SOURCE_DIR}/src/bfgs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/classical-mds.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/distributions.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/gaussian-process-regression.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/l-bfgs.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/rbf-interpolation.cpp
)

add_library(mathtoolbox STATIC ${headers} ${sources})
target_link_libraries(mathtoolbox PUBLIC Eigen3::Eigen)
target_include_directories(mathtoolbox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

######################################################################
# Python bindings
######################################################################

set(sources_python_bindings
	${CMAKE_CURRENT_SOURCE_DIR}/src/python-bindings.cpp
)

if(MATHTOOLBOX_PYTHON_BINDINGS)
	pybind11_add_module(pymathtoolbox ${sources_python_bindings})
	target_link_libraries(pymathtoolbox PUBLIC mathtoolbox)
endif()

######################################################################
# Installation
######################################################################

install(FILES ${headers} DESTINATION include/mathtoolbox)
install(TARGETS mathtoolbox ARCHIVE DESTINATION lib)

######################################################################
# Example demos
######################################################################

if(MATHTOOLBOX_BUILD_EXAMPLES)
	set(OTF_WITH_EIGEN ON CACHE INTERNAL "" FORCE)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/optimization-test-functions)

	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/bfgs)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/classical-mds)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/distributions)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/gaussian-process-regression)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/l-bfgs)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/rbf-interpolation)
endif()

######################################################################
# Tests
######################################################################

enable_testing()
if(MATHTOOLBOX_BUILD_EXAMPLES)
	add_test(NAME bfgs-test COMMAND $<TARGET_FILE:bfgs-test>)
	add_test(NAME classical-mds-test COMMAND $<TARGET_FILE:classical-mds-test>)
	add_test(NAME distributions-test COMMAND $<TARGET_FILE:distributions-test>)
	add_test(NAME gaussian-process-regression-test COMMAND $<TARGET_FILE:gaussian-process-regression-test> .)
	add_test(NAME l-bfgs-test COMMAND $<TARGET_FILE:l-bfgs-test>)
	add_test(NAME rbf-interpolation-test COMMAND $<TARGET_FILE:rbf-interpolation-test>)

	if(MATHTOOLBOX_PYTHON_BINDINGS)
		add_test(NAME python-classical-mds-test COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pymathtoolbox> ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/python-bindings/classical-mds.py)
	endif()
endif()
