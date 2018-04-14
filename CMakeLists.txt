cmake_minimum_required(VERSION 3.1)

project(mathtoolbox CXX)

set(CMAKE_CXX_STANDARD 11)

set(HEADERS mathtoolbox/metric-mds.hpp)

install(FILES ${HEADERS} DESTINATION include/mathtoolbox)