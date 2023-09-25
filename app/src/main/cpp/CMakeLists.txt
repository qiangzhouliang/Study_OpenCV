
cmake_minimum_required(VERSION 3.22.1)
# openCV version 3.4.4
project("study_opencv")

# 导入头文件
include_directories(${CMAKE_SOURCE_DIR}/../jniLibs/include)
# 批量导入c文件
file(GLOB openCV ${CMAKE_SOURCE_DIR}/../jniLibs/*.hpp)


add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        ${openCV}
        native-lib.cpp
        native-transform.cpp
        native-sdk.cpp
        native-mat.cpp
        native-utils.cpp
        native-imgproc.cpp
        util/BitmapUtil.cpp
        util/BitmapMatUtils.cpp)

# 设置一个环境变量 目录为 arm64-v8a
set(my_lib_path ${CMAKE_SOURCE_DIR}/../jniLibs/${CMAKE_ANDROID_ARCH_ABI})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${my_lib_path}")

target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        android
        opencv_java3
        # included in the NDK.
        #加入该依赖库 解决 undefined reference to 'AndroidBitmap_getInfo'
        jnigraphics
        log)