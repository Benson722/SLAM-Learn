cmake_minimum_required(VERSION 2.8)
project(ch6)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
 
# 添加cmake模块以使用ceres库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})
  
# 寻找G2O
find_package( G2O REQUIRED )
include_directories( 
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)
 
# Eigen   gaussNewton.cpp
include_directories("/usr/include/eigen3")
 
add_executable(Try_guassNewton  Try_guassNewton.cpp)
target_link_libraries(Try_guassNewton 
	${OpenCV_LIBS} 
	g2o_core g2o_stuff
)

