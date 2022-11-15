cmake_minimum_required(VERSION 3.0)

project(ALTensor LANGUAGES CXX
    VERSION 0.1.0
    DESCRIPTION "ALTensor is a C++ library for tensor operations and machine learning"
    HOMEPAGE_URL "")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(DIVISIBLE_INSTALL_LIB_DIR ${PROJECT_SOURCE_DIR}/lib)


add_executable(${PROJECT_NAME} main.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)

target_include_directories(${PROJECT_NAME} PUBLIC
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
$<INSTALL_INTERFACE:include>
PRIVATE src)

add_subdirectory(src)
target_link_libraries(${PROJECT_NAME} PUBLIC srclib)
