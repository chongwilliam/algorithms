# Kalman filter
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_FLAGS "-std=c++17 -O3")
set(CMAKE_BUILD_TYPE Release)

project(filter)

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(EIGEN3_INCLUDE_DIR /usr/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})

# hiredis
find_library(HIREDIS_LIBRARY hiredis)

# jsoncpp
if (CMAKE_SYSTEM_NAME MATCHES Darwin) 
	find_library(JSONCPP_LIBRARY jsoncpp)
elseif (CMAKE_SYSTEM_NAME MATCHES Linux)
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(JSONCPP jsoncpp)
	link_directories(${JSONCPP_LIBRARY_DIRS})
	set(JSONCPP_LIBRARY ${JSONCPP_LIBRARIES})
	include_directories("${JSONCPP_INCLUDE_DIRS}")
endif ()

# plotting (disable if running source)
# find_package(sciplot)

add_executable(app test.cpp ukf.cpp redis/RedisClient.cpp filter/ButterworthFilter.cpp)
# target_link_libraries(app ${HIREDIS_LIBRARY} ${JSONCPP_LIBRARIES} sciplot::sciplot)
target_link_libraries(app ${HIREDIS_LIBRARY} ${JSONCPP_LIBRARIES})

