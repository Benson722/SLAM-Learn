cmake_minimum_required( VERSION 2.8 )
# Cmake最低版本，如果加入“FATAL_ERROR”，将发出致命错误。
project( vo1 )
# 声明项目的名称

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_CXX_FLAGS "-std=c++14 -mfma")

# 添加cmake模块以使用 g2o
# 使用这种方法会自动在系统的 cmake 路径，以及前面文件设置的 cmake 的包含路径中查找 g2o；
# 可以通过LIST命令向 CMAKE_MODULE_PATH 的 list 中添加指定的路径。
# PROJECT_SOURCE_DIR=全路径/src
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
find_package( FMT  REQUIRED )

# find_package(Eigen3 REQUIRED) 为什么不使用这种引用方式？🤪
# 答：使用这种方法会自动在系统的cmake路径，以及前面文件设置的cmake的包含路径中查找Eigen；
# 

# include_directories(${EIGEN3_INCLUDE_DIR}) # 为什么不用这种方式？🤪
# 答：因为

# Eigen 使用：find_package / include_dir.. ? list
# link problem?

# Eigen3 库的引用方式与注意：🤪
# 1. Eigen是纯用头文件搭建起来的库，采用模板编程；
#      所以，在使用的时候，只需引用头文件即可，只需要设定一个包含路径即可，不需要链接库文件（因为没有）。
# 2. 库文件可以包含头文件，头文件不可包含库文件；头文件可视，库文件不可视。
# 3. 两种库的区别在于静态库被调用时直接加载到内存，而动态库再是在需要的时候加载到内存，不使用的时候再从内存释放。
# 4. 

# 加入所有头文件
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${FMT_INCLUDE_DIRS}    
    "/usr/include/eigen3/"
)

# 指示CMake创建一个新目标：可执行文件 orb_cv
# 这个可执行文件是通过编译和链接源文件 orb_cv.cpp生成的。
# CMake将为编译器使用默认设置，并自动选择生成工具

add_executable( orb_cv orb_cv.cpp  ) 

# 将目标库链接到可执行目标
# add_library 是静态库
# target_link_libraries 将动态库链接到目标
target_link_libraries( orb_cv ${OpenCV_LIBS} )

add_executable( orb_self orb_self.cpp  )
target_link_libraries( orb_self ${OpenCV_LIBS} )

add_executable( feature_extraction feature_extraction.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2 
add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2 
add_executable( triangulation triangulation.cpp )
target_link_libraries( triangulation ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
target_link_libraries( pose_estimation_3d2d 
   ${OpenCV_LIBS}
   ${CSPARSE_LIBRARY}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
)

add_executable( pose_estimation_hand_3d2d pose_estimation_hand_3d2d.cpp )
target_link_libraries( pose_estimation_hand_3d2d 
   ${OpenCV_LIBS}
   ${THIRD_PARTY_LIBS}
   fmt::fmt
)
