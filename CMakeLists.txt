set (CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")


cmake_minimum_required(VERSION 3.13)
project(deformationTransfer)

set(CMAKE_CXX_STANDARD 14)

add_executable(deformationTransfer main.cpp)


include_directories (src)

target_link_libraries (deformationTransfer
        /usr/local/Cellar/open-mesh/8.0/lib/libOpenMeshTools.8.0.dylib
        /usr/local/Cellar/open-mesh/8.0/lib/libOpenMeshCore.8.0.dylib
        )
