# Kalman filter
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(CMAKE_BUILD_TYPE Debug)

project(filter)

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# hiredis
find_library(HIREDIS_LIBRARY hiredis)

add_executable(app test.cpp redis/RedisClient.cpp filter/ButterworthFilter.cpp ${HIREDIS_LIBRARY}))

target_link_libraries(app)