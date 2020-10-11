cmake_minimum_required(VERSION 3.3)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)
if(MATHTOOLBOX_PYTHON_BINDINGS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
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
  set(PYBIND11_TEST           OFF CACHE INTERNAL "" FORCE)
  set(USE_PYTHON_INCLUDE_DIR  OFF CACHE INTERNAL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11)
endif()

######################################################################
# Core library
######################################################################

file(GLOB headers ${CMAKE_CURRENT_SOURCE_DIR}/include/mathtoolbox/*.hpp)
file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_library(mathtoolbox STATIC ${headers} ${sources})
target_link_libraries(mathtoolbox PUBLIC Eigen3::Eigen)
target_include_directories(mathtoolbox PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

######################################################################
# Python bindings
######################################################################

set(sources_python_bindings
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pybind/python-bindings.cpp
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

  set(TIMER_BUILD_TEST OFF CACHE INTERNAL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/timer)

  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/acquisition-function)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/bayesian-optimization)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/bfgs)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/classical-mds)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/gaussian-process-regression)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/gradient-descent)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/l-bfgs)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/log-determinant)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/matrix-inversion)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/probability-distributions)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/rbf-interpolation)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples/som)
endif()

######################################################################
# Tests
######################################################################

enable_testing()
if(MATHTOOLBOX_BUILD_EXAMPLES)
  add_test(NAME acquisition-function-test COMMAND $<TARGET_FILE:acquisition-function-test>)
  add_test(NAME bayesian-optimization-test COMMAND $<TARGET_FILE:bayesian-optimization-test>)
  add_test(NAME bfgs-test COMMAND $<TARGET_FILE:bfgs-test>)
  add_test(NAME classical-mds-test COMMAND $<TARGET_FILE:classical-mds-test>)
  add_test(NAME gaussian-process-regression-test COMMAND $<TARGET_FILE:gaussian-process-regression-test> .)
  add_test(NAME gradient-descent-test COMMAND $<TARGET_FILE:gradient-descent-test>)
  add_test(NAME l-bfgs-test COMMAND $<TARGET_FILE:l-bfgs-test>)
  add_test(NAME log-determinant-test COMMAND $<TARGET_FILE:log-determinant-test>)
  add_test(NAME matrix-inversion-test COMMAND $<TARGET_FILE:matrix-inversion-test>)
  add_test(NAME probability-distributions-test COMMAND $<TARGET_FILE:probability-distributions-test>)
  add_test(NAME rbf-interpolation-test COMMAND $<TARGET_FILE:rbf-interpolation-test>)
  add_test(NAME som-test COMMAND $<TARGET_FILE:som-test>)

  if(MATHTOOLBOX_PYTHON_BINDINGS)
    add_test(NAME python-classical-mds-test COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:pymathtoolbox> ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python-examples/classical-mds.py)
  endif()
endif()
